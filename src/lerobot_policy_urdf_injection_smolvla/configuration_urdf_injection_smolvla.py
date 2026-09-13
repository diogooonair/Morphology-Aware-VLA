from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("urdf_injection_smolvla")
@dataclass
class URDFInjectionSmolVLAConfig(SmolVLAConfig):
    """
    SmolVLA with URDF-conditioned FiLM modulation.

    The base SmolVLA architecture/training configuration is inherited
    unchanged. Morphology is used only through FiLM conditioning of
    the action expert.
    """
    # Maximum number of joints represented by the URDF encoder.
    max_joints: int = 18

    # Per-joint kinematic feature dimensionality.
    kinematic_input_dim: int = 12

    morphology_hidden_dim: int = 256

    # SmolVLA FiLM conditioning dimension.
    morphology_embedding_dim: int = 1024


    #Film params
    use_morphology_film: bool = True
    morphology_film_num_layers: int = 4

    # Multiplicative strength applied to the generated FiLM residual.
    morphology_film_scale: float = 0.2



    # Morphology specific training commands , smolVLA keep the defaults from config
    morphology_lr: float = 1e-4

    use_lr_split: bool = True


    #ranking loss
    use_morphology_ranking_loss: bool = True

    morphology_rank_probability: float = 0.25
    morphology_rank_weight: float = 0.10
    morphology_rank_margin: float = 0.01

    # Same-robot perturbed URDF negatives.
    morphology_hard_negative_root: str = (
        "assets/robots/generated_hard_negatives"
    )

    #Perturbed ratio
    morphology_hard_negative_ratio: float = 0.70

    morphology_negative_urdf_paths: str = (
        "assets/robots/so100.urdf,"
        "assets/robots/panda.urdf,"
        "assets/robots/xarm7.urdf,"
        "assets/robots/vx300s.urdf"
    )

    morphology_rank_log_every: int = 100

    robot_id: str = ""

    urdf_path: str = ""