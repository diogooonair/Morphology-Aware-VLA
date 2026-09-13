from typing import Any

import torch

from lerobot.policies.smolvla.processor_smolvla import (
    make_smolvla_pre_post_processors,
)

from .configuration_urdf_injection_smolvla import (
    URDFInjectionSmolVLAConfig,
)


class URDFInjectionSmolVLAPreprocessorWrapper:
    """
    Standard SmolVLA preprocessing + preservation/injection of
    robot morphology metadata.

    The processor DOES NOT parse the URDF.

    It only ensures that:
        robot_id
        urdf_path

    reach URDFInjectionSmolVLAPolicy.

    URDF parsing/caching and morphology encoding remain inside
    the policy.
    """

    def __init__(
        self,
        base_preprocessor,
        robot_id: str,
        urdf_path: str,
    ):
        self.base_preprocessor = base_preprocessor

        # Defaults supplied by the policy config / CLI.
        self.robot_id = robot_id
        self.urdf_path = urdf_path

    @staticmethod
    def _normalize_metadata(
        value: Any,
        batch_size: int,
        default_value: str,
    ) -> list[str]:
        """
        Convert morphology metadata into one string per batch element.
        """

        if value is None:
            return [
                default_value
            ] * batch_size

        if isinstance(
            value,
            str,
        ):
            return [
                value
            ] * batch_size

        if isinstance(
            value,
            torch.Tensor,
        ):
            value = (
                value.detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

        elif isinstance(
            value,
            tuple,
        ):
            value = list(
                value
            )

        elif not isinstance(
            value,
            list,
        ):
            value = [
                value
            ]

        result = [
            str(item)
            for item in value
        ]

        # A single metadata value can describe the whole batch.
        if (
            len(result) == 1
            and batch_size > 1
        ):
            result = (
                result
                * batch_size
            )

        # Pad defensively if necessary.
        if len(result) < batch_size:
            result.extend(
                [
                    default_value
                ]
                * (
                    batch_size
                    - len(result)
                )
            )

        return result[
            :batch_size
        ]

    def __call__(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:

        #Capture explicit morphology metadata BEFORE standard  SmolVLA processing.

        robot_id = batch.get(
            "robot_id",
            None,
        )

        urdf_path = batch.get(
            "urdf_path",
            None,
        )



        processed_batch = (
            self.base_preprocessor(
                batch
            )
        )


        # Determine batch size

        if (
            "observation.state"
            not in processed_batch
        ):
            raise KeyError(
                "SmolVLA processed batch does not contain "
                "'observation.state'."
            )

        batch_size = int(
            processed_batch[
                "observation.state"
            ].shape[0]
        )


        #Dataset does not contain robot_id / urdf_path, so it is explicitly supplied through
        # --policy.robot_id and --policy.urdf_path

        robot_ids = (
            self._normalize_metadata(
                value=robot_id,
                batch_size=batch_size,
                default_value=self.robot_id,
            )
        )

        urdf_paths = (
            self._normalize_metadata(
                value=urdf_path,
                batch_size=batch_size,
                default_value=self.urdf_path,
            )
        )

        processed_batch[
            "robot_id"
        ] = robot_ids

        processed_batch[
            "urdf_path"
        ] = urdf_paths

        return processed_batch

    #delegate the rest to the default processor

    def state_dict(
        self,
    ):
        return (
            self.base_preprocessor
            .state_dict()
        )

    def load_state_dict(
        self,
        state_dict,
    ):
        return (
            self.base_preprocessor
            .load_state_dict(
                state_dict
            )
        )

    def reset(
        self,
    ):
        return (
            self.base_preprocessor
            .reset()
        )

    def __getattr__(
        self,
        name,
    ):
        return getattr(
            self.base_preprocessor,
            name,
        )


def make_urdf_injection_smolvla_pre_post_processors(
    config: URDFInjectionSmolVLAConfig,
    dataset_stats: dict[
        str,
        dict[
            str,
            torch.Tensor,
        ],
    ]
    | None = None,
):
    """
    Standard SmolVLA pre/post processing with morphology metadata.

    The processor only carries robot_id and urdf_path.

    URDF parsing, caching, morphology encoding, pooling and FiLM
    conditioning are all performed by URDFInjectionSmolVLAPolicy.
    """


    #default values
    base_preprocessor, postprocessor = (
        make_smolvla_pre_post_processors(
            config=config,
            dataset_stats=dataset_stats,
        )
    )

    #wrap preprocessor to keep morfology params
    wrapped_preprocessor = (
        URDFInjectionSmolVLAPreprocessorWrapper(
            base_preprocessor=base_preprocessor,
            robot_id=config.robot_id,
            urdf_path=config.urdf_path,
        )
    )

    return (
        wrapped_preprocessor,
        postprocessor,
    )