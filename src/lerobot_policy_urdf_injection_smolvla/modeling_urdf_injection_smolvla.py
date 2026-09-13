from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, Unpack

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
)

from .configuration_urdf_injection_smolvla import (
    URDFInjectionSmolVLAConfig,
)
from .urdf_parser import MorphologyCache



class MorphologyFiLM(nn.Module):
    """
    Morphology-conditioned FiLM modulation.

    The projection is zero-initialized, so the module is an exact identity at
    initialization:

        y = x * (1 + scale * 0) + scale * 0 = x
    """

    def __init__(
        self,
        morphology_dim: int,
        hidden_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()

        self.scale = float(scale)
        self.norm = nn.LayerNorm(morphology_dim)
        self.proj = nn.Linear(morphology_dim, 2 * hidden_dim)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        morphology_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conditioning = self.proj(self.norm(morphology_embedding))
        gamma, beta = conditioning.chunk(2, dim=-1)

        gamma = torch.tanh(gamma)
        gamma = gamma.to(device=hidden.device, dtype=hidden.dtype)
        beta = beta.to(device=hidden.device, dtype=hidden.dtype)

        while gamma.ndim < hidden.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        output = hidden * (1.0 + self.scale * gamma) + self.scale * beta
        return output, gamma, beta


class MorphologyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class URDFInjectionSmolVLAPolicy(SmolVLAPolicy):
    """
    SmolVLA conditioned by URDF morphology only through FiLM modulation of the
    final action-expert transformer layers.

    The vanilla SmolVLA training/freezing policy is left unchanged. The custom
    path adds only:

      1. URDF -> morphology tokens -> pooled morphology embedding
      2. FiLM on the final N action-expert layers
      3. optional correct-vs-wrong-URDF counterfactual ranking loss
    """

    config_class = URDFInjectionSmolVLAConfig
    config: URDFInjectionSmolVLAConfig
    name = "urdf_injection_smolvla"

    def __init__(self, config: URDFInjectionSmolVLAConfig, **kwargs):
        #compatability patch
        try:
            super().__init__(config, **kwargs)
        except TypeError:
            super().__init__(config)

        # Morphology representation

        self.morphology_encoder = MorphologyMLP(
            input_dim=config.kinematic_input_dim,
            hidden_dim=config.morphology_hidden_dim,
            embed_dim=config.morphology_embedding_dim,
        )

        self.joint_index_embedding = nn.Embedding(
            config.max_joints,
            config.morphology_embedding_dim,
        )

        self.morphology_cache = MorphologyCache(max_joints=config.max_joints)

        # Runtime conditioning state used by the FiLM hooks.
        self._active_dof_mask: torch.Tensor | None = None
        self._active_morphology_pooled: torch.Tensor | None = None

        # Cache the MLP residual so FiLM can be applied to the full decoder layer output.
        self._film_residual_cache: dict[int, torch.Tensor] = {}
        self._film_hook_handles = []

        # Counterfactual-training diagnostics.
        self._last_correct_loss = None
        self._last_wrong_loss = None
        self._last_rank_loss = None
        self._last_total_loss = None
        self._morphology_rank_batch_count = 0
        self._last_negative_types: list[str] = []

        self._init_morphology_film()
        self._print_trainable_summary()

    # SmolVLA Action Expert discovery

    def _get_expert_wrapper(self):
        """Return SmolVLA's VLM+expert wrapper."""
        model = getattr(self, "model", None)
        if model is None:
            return None

        return getattr(model, "vlm_with_expert", None)

    def _get_expert_module(self):
        """Return SmolVLA's language-model action expert."""
        wrapper = self._get_expert_wrapper()
        if wrapper is None:
            return None

        return getattr(wrapper, "lm_expert", None)

    def _get_expert_layers(self):
        expert = self._get_expert_module()
        if expert is None:
            raise RuntimeError("Could not locate SmolVLA action expert (lm_expert).")

        layers = getattr(expert, "layers", None)
        if layers is None:
            # Compatibility patch for wrappers that expose layers one level deeper
            inner_model = getattr(expert, "model", None)
            layers = getattr(inner_model, "layers", None) if inner_model is not None else None

        if layers is None:
            raise RuntimeError("Could not locate SmolVLA action-expert layers.")

        return layers

    def _get_expert_hidden_dim(self) -> int:
        wrapper = self._get_expert_wrapper()

        if wrapper is not None and hasattr(wrapper, "expert_hidden_size"):
            return int(wrapper.expert_hidden_size)

        expert = self._get_expert_module()
        expert_config = getattr(expert, "config", None)
        if expert_config is not None and hasattr(expert_config, "hidden_size"):
            return int(expert_config.hidden_size)

        layers = self._get_expert_layers()
        first_layer = layers[0]
        norm = getattr(first_layer, "input_layernorm", None)
        weight = getattr(norm, "weight", None) if norm is not None else None
        if weight is not None:
            return int(weight.numel())

        raise RuntimeError("Could not infer SmolVLA action-expert hidden dimension.")

    # -----
    # FiLM hooks
    # -----

    def _init_morphology_film(self):
        self.morphology_film = nn.ModuleList()

        if not getattr(self.config, "use_morphology_film", False):
            return

        layers = self._get_expert_layers()

        requested_layers = int(
            getattr(self.config, "morphology_film_num_layers", 4)
        )
        requested_layers = max(1, min(requested_layers, len(layers)))

        hidden_dim = self._get_expert_hidden_dim()
        film_scale = float(getattr(self.config, "morphology_film_scale", 0.2))

        self._film_start_layer = len(layers) - requested_layers
        film_layers = layers[self._film_start_layer :]

        for _ in film_layers:
            self.morphology_film.append(
                MorphologyFiLM(
                    morphology_dim=self.config.morphology_embedding_dim,
                    hidden_dim=hidden_dim,
                    scale=film_scale,
                )
            )

        # SmolVLA executes expert layers manually, so hooks are used to apply FiLM
        # to the complete post-residual layer output.
        for film_index, layer in enumerate(film_layers):
            post_norm = getattr(layer, "post_attention_layernorm", None)
            mlp = getattr(layer, "mlp", None)

            if post_norm is None or mlp is None:
                raise RuntimeError(
                    "Selected SmolVLA expert layer does not expose "
                    "post_attention_layernorm and mlp; cannot install exact "
                    "post-layer FiLM hooks."
                )

            self._film_hook_handles.append(
                post_norm.register_forward_pre_hook(
                    self._make_capture_residual_hook(film_index)
                )
            )

            self._film_hook_handles.append(
                mlp.register_forward_hook(
                    self._make_mlp_film_hook(film_index)
                )
            )

        print(
            "Morphology FiLM: "
            f"last {requested_layers}/{len(layers)} SmolVLA expert layers, "
            f"expert_hidden_dim={hidden_dim}, "
            f"morphology_dim={self.config.morphology_embedding_dim}, "
            f"scale={film_scale}",
            flush=True,
        )

    def _make_capture_residual_hook(self, film_index: int):
        def hook(module, inputs):
            if self._active_morphology_pooled is None:
                return None

            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(
                    "Could not capture SmolVLA expert residual for FiLM."
                )

            self._film_residual_cache[film_index] = inputs[0]
            return None

        return hook

    @staticmethod
    def _repeat_condition_to_batch(
        tensor: torch.Tensor,
        target_batch_size: int,
        label: str,
    ) -> torch.Tensor:
        if tensor.shape[0] == target_batch_size:
            return tensor

        if target_batch_size % tensor.shape[0] != 0:
            raise RuntimeError(
                f"{label} batch mismatch: source={tensor.shape[0]}, "
                f"target={target_batch_size}."
            )

        repeats = target_batch_size // tensor.shape[0]
        return tensor.repeat_interleave(repeats, dim=0)

    def _apply_valid_urdf_mask(
        self,
        film_hidden: torch.Tensor,
        original_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Samples without a URDF remain exactly vanilla SmolVLA."""
        if self._active_dof_mask is None:
            return film_hidden

        valid = self._active_dof_mask.any(dim=1)
        valid = self._repeat_condition_to_batch(
            valid,
            original_hidden.shape[0],
            "FiLM DoF mask",
        )

        valid = valid.to(
            device=original_hidden.device,
            dtype=original_hidden.dtype,
        )

        while valid.ndim < original_hidden.ndim:
            valid = valid.unsqueeze(-1)

        return valid * film_hidden + (1.0 - valid) * original_hidden

    def _make_mlp_film_hook(self, film_index: int):
        def hook(module, inputs, output):
            morphology = self._active_morphology_pooled

            if morphology is None:
                # Defensive cleanup in case a previous interrupted pass left a
                # cached residual.
                self._film_residual_cache.pop(film_index, None)
                return output

            if not isinstance(output, torch.Tensor):
                return output

            residual = self._film_residual_cache.pop(film_index, None)
            if residual is None:
                raise RuntimeError(
                    "Morphology FiLM MLP hook fired without a captured "
                    f"residual for film layer {film_index}."
                )

            # Apply FiLM to the complete expert-layer output before the residual is added.
            original_layer_hidden = output + residual

            cond = self._repeat_condition_to_batch(
                morphology,
                original_layer_hidden.shape[0],
                "Morphology FiLM",
            )

            film_hidden, _, _ = self.morphology_film[film_index](
                original_layer_hidden,
                cond,
            )

            film_hidden = self._apply_valid_urdf_mask(
                film_hidden,
                original_layer_hidden,
            )

            # Adjust for SmolVLA's subsequent residual addition.
            return film_hidden - residual

        return hook

    # =========================================================================
    # Morphology encoding
    # =========================================================================

    def _pool_morphology_tokens(
        self,
        tokens: torch.Tensor,
        dof_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = dof_mask.unsqueeze(-1).to(tokens.dtype)
        summed = (tokens * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / count

        return torch.nn.functional.normalize(pooled, dim=-1)

    def _encode_morphology_tokens(
        self,
        joint_features: torch.Tensor,
        dof_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.morphology_encoder(joint_features)

        joint_indices = torch.arange(
            self.config.max_joints,
            device=tokens.device,
        )

        index_embedding = self.joint_index_embedding(joint_indices).unsqueeze(0)
        tokens = tokens + index_embedding.to(tokens.dtype)

        return tokens * dof_mask.unsqueeze(-1).to(tokens.dtype)

    @staticmethod
    def _as_metadata_list(
        value: Any,
        batch_size: int,
        default_value: str,
    ) -> list[str]:
        if value is None:
            return [default_value] * batch_size

        if isinstance(value, str):
            return [value] * batch_size

        if isinstance(value, torch.Tensor):
            flattened = value.detach().cpu().reshape(-1).tolist()
            result = [str(item) for item in flattened]
        elif isinstance(value, (list, tuple)):
            result = [str(item) for item in value]
        else:
            result = [str(value)]

        if len(result) < batch_size:
            result.extend([default_value] * (batch_size - len(result)))

        return result[:batch_size]

    def _build_morphology_batch(
        self,
        robot_ids: list[str],
        urdf_paths: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_features_list = []
        dof_mask_list = []

        for robot_id, urdf_path in zip(robot_ids, urdf_paths, strict=True):
            if urdf_path:
                morphology = self.morphology_cache.get_morphology(
                    robot_id,
                    urdf_path,
                    device,
                )
                joint_features_list.append(morphology["joint_features"])
                dof_mask_list.append(morphology["dof_mask"])
            else:
                joint_features_list.append(
                    torch.zeros(
                        (
                            self.config.max_joints,
                            self.config.kinematic_input_dim,
                        ),
                        device=device,
                    )
                )
                dof_mask_list.append(
                    torch.zeros(
                        (self.config.max_joints,),
                        dtype=torch.bool,
                        device=device,
                    )
                )

        joint_features = torch.stack(joint_features_list, dim=0)
        dof_mask = torch.stack(dof_mask_list, dim=0)

        morphology_tokens = self._encode_morphology_tokens(
            joint_features,
            dof_mask,
        )

        return morphology_tokens, dof_mask

    def _set_active_morphology(
        self,
        robot_ids: list[str],
        urdf_paths: list[str],
    ) -> None:
        device = self.morphology_encoder.mlp[0].weight.device

        morphology_tokens, dof_mask = self._build_morphology_batch(
            robot_ids=robot_ids,
            urdf_paths=urdf_paths,
            device=device,
        )

        self._active_dof_mask = dof_mask
        self._active_morphology_pooled = self._pool_morphology_tokens(
            morphology_tokens,
            dof_mask,
        )

    def _clear_active_morphology(self) -> None:
        self._active_dof_mask = None
        self._active_morphology_pooled = None
        self._film_residual_cache.clear()

    def _forward_with_metadata(
        self,
        batch: dict[str, Any],
        robot_ids: list[str],
        urdf_paths: list[str],
        **kwargs,
    ):
        self._set_active_morphology(robot_ids, urdf_paths)

        try:
            return super().forward(batch, **kwargs)
        finally:
            self._clear_active_morphology()


    # Counterfactual morphology sampling
    def _negative_urdf_pool(self) -> list[str]:
        raw_pool = getattr(self.config, "morphology_negative_urdf_paths", "")

        if isinstance(raw_pool, str):
            return [item.strip() for item in raw_pool.split(",") if item.strip()]

        return [str(item).strip() for item in raw_pool if str(item).strip()]

    def _hard_negative_candidates(self, correct_urdf_path: str) -> list[str]:
        root = Path(
            getattr(
                self.config,
                "morphology_hard_negative_root",
                "assets/robots/generated_hard_negatives",
            )
        )

        if not correct_urdf_path:
            return []

        robot_name = Path(correct_urdf_path).stem
        robot_dir = root / robot_name

        if not robot_dir.exists():
            return []

        return [
            str(path)
            for path in sorted(robot_dir.glob(f"{robot_name}_perturbed_*.urdf"))
            if path.is_file()
        ]

    def _make_wrong_morphology_metadata(
        self,
        correct_robot_ids: list[str],
        correct_urdf_paths: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Sample counterfactual morphologies:

          - morphology_hard_negative_ratio: perturbed same-robot URDF
          - remaining probability: nominal cross-robot URDF
        """
        del correct_robot_ids  # URDF path determines the negative cache key.

        cross_robot_pool = self._negative_urdf_pool()
        hard_ratio = float(
            getattr(self.config, "morphology_hard_negative_ratio", 0.70)
        )

        wrong_robot_ids = []
        wrong_urdf_paths = []
        self._last_negative_types = []

        for correct_path in correct_urdf_paths:
            hard_candidates = self._hard_negative_candidates(correct_path)

            use_hard = bool(hard_candidates) and random.random() < hard_ratio

            if use_hard:
                selected = random.choice(hard_candidates)
                negative_type = "hard"
            else:
                candidates = [path for path in cross_robot_pool if path != correct_path]

                if candidates:
                    selected = random.choice(candidates)
                    negative_type = "cross"
                elif hard_candidates:
                    selected = random.choice(hard_candidates)
                    negative_type = "hard"
                else:
                    raise ValueError(
                        "No valid counterfactual morphology available for "
                        f"{correct_path!r}."
                    )

            wrong_urdf_paths.append(selected)
            selected_name = Path(selected).stem
            wrong_robot_ids.append(
                f"counterfactual_{negative_type}_{selected_name}"
            )
            self._last_negative_types.append(negative_type)

        return wrong_robot_ids, wrong_urdf_paths

    # =========================================================================
    # Loss helpers
    # =========================================================================

    @staticmethod
    def _split_policy_output(
            output: Any,
    ) -> tuple[
        torch.Tensor,
        dict[str, Any],
        Literal["tensor", "tuple"],
    ]:
        """
        Normalize SmolVLA forward output to:

            loss,
            dict[str, Any],
            output kind
        """

        if isinstance(
                output,
                torch.Tensor,
        ):
            return (
                output,
                {},
                "tensor",
            )

        if (
                isinstance(output, tuple)
                and len(output) >= 1
                and isinstance(
                    output[0],
                    torch.Tensor,
                )
        ):
            output_dict: dict[str, Any] = {}

            if (
                    len(output) >= 2
                    and isinstance(
                    output[1],
                    Mapping,
                )
            ):
                for key, value in output[1].items():
                    if not isinstance(
                            key,
                            str,
                    ):
                        raise TypeError(
                            "SmolVLA loss dictionary "
                            "contains a non-string key: "
                            f"{key!r}"
                        )

                    output_dict[key] = value

            return (
                output[0],
                output_dict,
                "tuple",
            )

        raise TypeError(
            "Unexpected SmolVLA forward output. "
            "Expected a loss tensor or "
            "(loss, loss_dict), received "
            f"{type(output)}."
        )

    @staticmethod
    def _restore_policy_output(
            total_loss: torch.Tensor,
            output_dict: dict[str, Any],
            output_kind: Literal[
                "tensor",
                "tuple",
            ],
    ):
        if output_kind == "tensor":
            return total_loss

        return (
            total_loss,
            output_dict,
        )

    @staticmethod
    def _is_morphology_parameter_name(name: str) -> bool:
        return (
            name.startswith("morphology_encoder.")
            or name.startswith("joint_index_embedding.")
            or name.startswith("morphology_film.")
        )

    @contextmanager
    def _temporarily_freeze_non_morphology(self):
        """
        During the wrong-URDF branch, gradients should flow THROUGH vanilla
        SmolVLA into the morphology pathway, but vanilla SmolVLA itself must not
        learn to become deliberately worse for a wrong URDF.
        """
        changed: list[tuple[torch.nn.Parameter, bool]] = []

        try:
            for name, param in self.named_parameters():
                if self._is_morphology_parameter_name(name):
                    continue

                changed.append((param, param.requires_grad))
                param.requires_grad_(False)

            yield

        finally:
            for param, requires_grad in changed:
                param.requires_grad_(requires_grad)

    def _sample_shared_flow_matching_noise_time(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Explicitly sample noise/time once so the correct- and wrong-URDF
        branches see identical flow-matching stochasticity.
        """
        if "action" not in batch:
            raise KeyError("Training batch does not contain 'action'.")

        action = batch["action"]
        batch_size = int(action.shape[0])
        device = action.device

        noise_shape = (
            batch_size,
            int(self.config.chunk_size),
            int(self.config.max_action_dim),
        )

        noise = self.model.sample_noise(noise_shape, device)
        time = self.model.sample_time(batch_size, device)
        return noise, time


    # Training
    def forward(
            self,
            batch: dict[str, Tensor],
            noise: Tensor | None = None,
            time: Tensor | None = None,
            reduction: str = "mean",
    ):
        """
        Train with vanilla SmolVLA flow matching plus optional same-sample,
        same-noise counterfactual morphology ranking.

        Only the URDF assignment differs between correct and wrong branches.
        """
        if batch is None:
            raise ValueError("batch cannot be None during training.")

        batch_size = int(batch["observation.state"].shape[0])

        default_robot_id = (
                getattr(
                    self.config,
                    "robot_id",
                    "",
                )
                or "default_robot"
        )

        default_urdf_path = (
                getattr(
                    self.config,
                    "urdf_path",
                    "",
                )
                or ""
        )

        correct_robot_ids = self._as_metadata_list(
            batch.get("robot_id"),
            batch_size,
            default_robot_id,
        )

        correct_urdf_paths = self._as_metadata_list(
            batch.get("urdf_path"),
            batch_size,
            default_urdf_path,
        )

        self._last_correct_loss = None
        self._last_wrong_loss = None
        self._last_rank_loss = None
        self._last_total_loss = None

        use_ranking = self.training and getattr(
            self.config,
            "use_morphology_ranking_loss",
            False,
        )

        rank_probability = float(
            getattr(self.config, "morphology_rank_probability", 0.25)
        )

        all_have_urdf = all(bool(path) for path in correct_urdf_paths)
        run_counterfactual = (
            use_ranking
            and all_have_urdf
            and random.random() < rank_probability
        )

        # Normal SmolVLA + correct morphology
        if not run_counterfactual:
            output = self._forward_with_metadata(
                batch=batch,
                robot_ids=correct_robot_ids,
                urdf_paths=correct_urdf_paths,
                noise=noise,
                time=time,
                reduction=reduction,
            )

            correct_loss, output_dict, output_kind = self._split_policy_output(output)
            correct_loss_scalar = correct_loss.mean()

            self._last_correct_loss = correct_loss_scalar.detach()
            self._last_total_loss = correct_loss_scalar.detach()

            output_dict.update(
                {
                    "loss_correct": correct_loss_scalar.detach(),
                    "loss_wrong": torch.tensor(
                        float("nan"),
                        device=correct_loss_scalar.device,
                    ),
                    "loss_morphology_rank": torch.zeros(
                        (),
                        device=correct_loss_scalar.device,
                    ),
                    "used_morphology_rank": torch.zeros(
                        (),
                        device=correct_loss_scalar.device,
                    ),
                }
            )

            return self._restore_policy_output(
                total_loss=correct_loss_scalar,
                output_dict=output_dict,
                output_kind=output_kind,
            )


        # Counterfactual branch

        wrong_robot_ids, wrong_urdf_paths = self._make_wrong_morphology_metadata(
            correct_robot_ids=correct_robot_ids,
            correct_urdf_paths=correct_urdf_paths,
        )

        # Share flow-matching noise/time and replay Torch RNG for identical stochastic passes.
        if noise is None or time is None:
            sampled_noise, sampled_time = (
                self._sample_shared_flow_matching_noise_time(
                    batch
                )
            )

            if noise is None:
                noise = sampled_noise

            if time is None:
                time = sampled_time

        shared_noise = noise
        shared_time = time

        cpu_rng_state, cuda_rng_state = (
            self._capture_rng_state()
        )

        correct_output = self._forward_with_metadata(
            batch=batch,
            robot_ids=correct_robot_ids,
            urdf_paths=correct_urdf_paths,
            noise=shared_noise,
            time=shared_time,
            reduction=reduction,
        )

        correct_loss, output_dict, output_kind = self._split_policy_output(
            correct_output
        )
        correct_loss_scalar = correct_loss.mean()

        self._restore_rng_state(
            cpu_rng_state,
            cuda_rng_state,
        )

        with self._temporarily_freeze_non_morphology():
            wrong_output = self._forward_with_metadata(
                batch=batch,
                robot_ids=wrong_robot_ids,
                urdf_paths=wrong_urdf_paths,
                noise=shared_noise,
                time=shared_time,
                reduction=reduction,
            )

            wrong_loss, _, _ = (
                self._split_policy_output(
                    wrong_output
                )
            )

            wrong_loss_scalar = (
                wrong_loss.mean()
            )

        margin = float(getattr(self.config, "morphology_rank_margin", 0.01))
        rank_weight = float(getattr(self.config, "morphology_rank_weight", 0.10))

        rank_loss = torch.relu(
            margin + correct_loss_scalar - wrong_loss_scalar
        )

        total_loss = correct_loss_scalar + rank_weight * rank_loss

        self._last_correct_loss = correct_loss_scalar.detach()
        self._last_wrong_loss = wrong_loss_scalar.detach()
        self._last_rank_loss = rank_loss.detach()
        self._last_total_loss = total_loss.detach()

        output_dict.update(
            {
                "loss_correct": correct_loss_scalar.detach(),
                "loss_wrong": wrong_loss_scalar.detach(),
                "loss_morphology_rank": rank_loss.detach(),
                "morphology_loss_gap": (
                    wrong_loss_scalar.detach() - correct_loss_scalar.detach()
                ),
                "used_morphology_rank": torch.ones(
                    (),
                    device=total_loss.device,
                ),
            }
        )

        self._morphology_rank_batch_count += 1

        rank_log_every = int(
            getattr(self.config, "morphology_rank_log_every", 100)
        )

        if (
            rank_log_every > 0
            and self._morphology_rank_batch_count % rank_log_every == 0
        ):
            correct_value = float(correct_loss_scalar.detach().cpu().item())
            wrong_value = float(wrong_loss_scalar.detach().cpu().item())
            gap_value = wrong_value - correct_value
            rank_value = float(rank_loss.detach().cpu().item())
            total_value = float(total_loss.detach().cpu().item())

            hard_count = sum(
                negative_type == "hard"
                for negative_type in self._last_negative_types
            )
            cross_count = sum(
                negative_type == "cross"
                for negative_type in self._last_negative_types
            )

            print(
                "[MORPH-RANK] "
                f"rank_batch={self._morphology_rank_batch_count} "
                f"correct={correct_value:.6f} "
                f"wrong={wrong_value:.6f} "
                f"gap={gap_value:+.6f} "
                f"rank={rank_value:.6f} "
                f"neg=hard:{hard_count}/cross:{cross_count} "
                f"total={total_value:.6f}",
                flush=True,
            )

        return self._restore_policy_output(
            total_loss=total_loss,
            output_dict=output_dict,
            output_kind=output_kind,
        )


    # Inference

    @torch.no_grad()
    def select_action(
            self,
            batch: dict[str, Tensor],
            noise: Tensor | None = None,
            **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Run SmolVLA inference conditioned through morphology FiLM."""
        state = batch.get(
            "observation.state"
        )

        if isinstance(
                state,
                Tensor,
        ):
            batch_size = int(
                state.shape[0]
            )
        else:
            state_keys = [
                key
                for key in batch
                if "state" in key
            ]

            if (
                    state_keys
                    and isinstance(
                batch[state_keys[0]],
                Tensor,
            )
            ):
                batch_size = int(
                    batch[state_keys[0]].shape[0]
                )
            else:
                batch_size = 1

        default_robot_id = (
                getattr(
                    self.config,
                    "robot_id",
                    "",
                )
                or "default_robot"
        )

        default_urdf_path = (
                getattr(
                    self.config,
                    "urdf_path",
                    "",
                )
                or ""
        )

        robot_ids = self._as_metadata_list(
            batch.get("robot_id"),
            batch_size,
            default_robot_id,
        )

        urdf_paths = self._as_metadata_list(
            batch.get("urdf_path"),
            batch_size,
            default_urdf_path,
        )

        self._set_active_morphology(
            robot_ids,
            urdf_paths,
        )

        try:
            return super().select_action(
                batch,
                noise=noise,
                **kwargs,
            )
        finally:
            self._clear_active_morphology()

    @torch.no_grad()
    def predict_action_chunk(
            self,
            batch: dict[str, Tensor],
            noise: Tensor | None = None,
            **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Chunk inference including RTC-compatible keyword arguments."""

        state = batch.get(
            "observation.state"
        )

        batch_size = (
            int(state.shape[0])
            if isinstance(state, Tensor)
            else 1
        )

        default_robot_id = (
                getattr(
                    self.config,
                    "robot_id",
                    "",
                )
                or "default_robot"
        )

        default_urdf_path = (
                getattr(
                    self.config,
                    "urdf_path",
                    "",
                )
                or ""
        )

        robot_ids = self._as_metadata_list(
            batch.get("robot_id"),
            batch_size,
            default_robot_id,
        )

        urdf_paths = self._as_metadata_list(
            batch.get("urdf_path"),
            batch_size,
            default_urdf_path,
        )

        self._set_active_morphology(
            robot_ids,
            urdf_paths,
        )

        try:
            return super().predict_action_chunk(
                batch,
                noise=noise,
                **kwargs,
            )
        finally:
            self._clear_active_morphology()


    # Optimizer
    def get_optim_params(self):
        """
        Preserve vanilla SmolVLA's base LR and optionally give only the new
        morphology pathway its own LR.
        """
        if not getattr(self.config, "use_lr_split", True):
            return [param for param in self.parameters() if param.requires_grad]

        morphology_lr = float(getattr(self.config, "morphology_lr", 1e-4))

        base_params = []
        morphology_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if self._is_morphology_parameter_name(name):
                morphology_params.append(param)
            else:
                base_params.append(param)

        groups = []

        if base_params:
            groups.append({"params": base_params})

        if morphology_params:
            groups.append(
                {
                    "params": morphology_params,
                    "lr": morphology_lr,
                }
            )

        print(
            "get_optim_params:\n"
            f"  vanilla SmolVLA : {len(base_params)} tensors at base LR\n"
            f"  morphology      : {len(morphology_params)} tensors "
            f"at {morphology_lr:g}",
            flush=True,
        )

        return groups

    def _print_trainable_summary(self):
        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )

        expert = self._get_expert_module()
        expert_trainable = (
            sum(param.numel() for param in expert.parameters() if param.requires_grad)
            if expert is not None
            else 0
        )

        morphology_trainable = sum(
            param.numel()
            for name, param in self.named_parameters()
            if param.requires_grad and self._is_morphology_parameter_name(name)
        )

        print("=" * 80)
        print("URDF-INJECTION-SMOLVLA TRAINABLE MODULES")
        print("=" * 80)
        print(f"Expert trainable params : {expert_trainable:,}")
        print(f"Morphology trainable    : {morphology_trainable:,}")
        print(
            f"TOTAL trainable         : {trainable_params:,} / {total_params:,} "
            f"({100.0 * trainable_params / max(total_params, 1):.2f}%)"
        )
        print(
            "Vanilla SmolVLA freeze/train settings are preserved; only the "
            "morphology pathway is added."
        )
        print("=" * 80)


    # Checkpoint compatibility
    def load_state_dict(
            self,
            state_dict: Mapping[str, Tensor],
            strict: bool = True,
            assign: bool = False,
    ):
        """
        Load either:

          - vanilla SmolVLA base checkpoints (morphology parameters missing), or
          - URDF-injection SmolVLA checkpoints (morphology parameters present).
        """
        cleaned_state_dict: dict[str, Tensor] = {}

        custom_prefixes = (
            "morphology_encoder.",
            "joint_index_embedding.",
            "morphology_film.",
        )

        for key, value in state_dict.items():
            remapped_key = key

            # Support checkpoints saved with an extra `model.` prefix.
            for prefix in custom_prefixes:
                wrapped_prefix = f"model.{prefix}"
                if key.startswith(wrapped_prefix):
                    remapped_key = key.replace(wrapped_prefix, prefix, 1)
                    break

            cleaned_state_dict[remapped_key] = value

        try:
            result = super().load_state_dict(
                cleaned_state_dict,
                strict=False,
                assign=assign,
            )
        except TypeError:
            result = super().load_state_dict(
                cleaned_state_dict,
                strict=False,
            )

        #Missing params are expected when training from scratch
        allowed_missing_prefixes = custom_prefixes

        unexpected_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]

        if unexpected_missing:
            raise RuntimeError(
                "Unexpected missing SmolVLA checkpoint keys: "
                f"{unexpected_missing}"
            )

        if result.unexpected_keys:
            raise RuntimeError(
                "Unexpected SmolVLA checkpoint keys: "
                f"{result.unexpected_keys}"
            )

        if result.missing_keys:
            print(
                "Initialized new morphology/FiLM parameters not present in "
                f"the base checkpoint: {result.missing_keys}",
                flush=True,
            )

        return result

    @staticmethod
    def _capture_rng_state():
        cpu_state = torch.random.get_rng_state()

        cuda_state = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        return cpu_state, cuda_state

    @staticmethod
    def _restore_rng_state(
            cpu_state,
            cuda_state,
    ):
        torch.random.set_rng_state(
            cpu_state
        )

        if (
                cuda_state is not None
                and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                cuda_state
            )
