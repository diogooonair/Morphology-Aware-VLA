try:
    from lerobot_policy_urdf_injection_smolvla.processor_urdf_injection_smolvla import (
        make_urdf_injection_smolvla_pre_post_processors,
    )

except ValueError as e:
    if "already registered as" in str(e):
        import sys

        make_urdf_injection_smolvla_pre_post_processors = None

        for module_name, module in sys.modules.items():
            if (
                module_name.endswith("processor_urdf_injection_smolvla")
                and hasattr(
                    module,
                    "make_urdf_injection_smolvla_pre_post_processors",
                )
            ):
                make_urdf_injection_smolvla_pre_post_processors = getattr(
                    module,
                    "make_urdf_injection_smolvla_pre_post_processors",
                )
                break

        if make_urdf_injection_smolvla_pre_post_processors is None:
            raise RuntimeError(
                "URDF-Injection SmolVLA processor should be imported but can t be located "
            ) from e
    else:
        raise

stats = (
    dataset_stats
    if "dataset_stats" in locals()
    else kwargs.get("dataset_stats")
)

processors = make_urdf_injection_smolvla_pre_post_processors(
    config=policy_cfg,
    dataset_stats=stats,
)

return processors