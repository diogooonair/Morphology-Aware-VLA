# Morphology-Aware-VLA

Official repository for morphology-conditioned SmolVLA. Integrates
URDF-based FiLM conditioning and counterfactual training to improve
robotic manipulation and motion quality.

This repository contains:

-   Our conditioned SmolVLA;
-   the URDF parser;
-   the policy configuration;
-   the trajectories recorded during the evaluation trials;
-   the engagement-labeled trials;
-   the script used to calculate the metrics based on the recorded
    trajectories.

## Repository Structure

``` text
.
├── src/
│   └── lerobot_policy_urdf_injection_smolvla/
│       ├── configuration_urdf_injection_smolvla.py
│       ├── __init__.py
│       ├── modeling_urdf_injection_smolvla.py
│       ├── processor_urdf_injection_smolvla.py
│       └── urdf_parser.py
│
├── assets/
│   └── robots/
│       ├── so101.urdf
│       ├── so100.urdf
│       ├── panda.urdf
│       ├── xarm7.urdf
│       ├── vx300s.urdf
│       └── generated_hard_negatives/
│
├── trajectory_results/
│   ├── Baseline_20k_b64_black_solo_n30/
│   ├── Baseline_20k_b64_pink_solo_n30/
│   ├── Baseline_20k_b64_black_both_n30/
│   ├── Baseline_20k_b64_pink_both_n30/
│   ├── Ours_20k_b64_black_solo_n30/
│   ├── Ours_20k_b64_pink_solo_n30/
│   ├── Ours_20k_b64_black_both_n30/
│   ├── Ours_20k_b64_pink_both_n30/
│   └── engagement_labels.csv
│
├── evaluate_movement_quality.py
└── README.md
```

## Installation and Usage

A small change is required in LeRobot's processor factory so that
`urdf_injection_smolvla` can instantiate its custom pre/post-processors.

In `lerobot/src/lerobot/policies/factory.py`, before the SmolVLA
processor case, add the `elif` for our custom processor wrapper. The
required patch is provided in `lerobot.patch.py`.

To install the policy, clone the repository and run the following
command from inside the repository folder:

``` bash
pip install -e .
```

### Training

For training, we used the following configuration. Replace the robotic
arm name and URDF path if fine-tuning for another arm, and replace the
dataset and output paths as needed.

``` bash
lerobot-train \
  --policy.type=urdf_injection_smolvla \
  --policy.pretrained_path="lerobot/smolvla_base" \
  --policy.robot_id="so101" \
  --policy.urdf_path="assets/robots/so101.urdf" \
  --dataset.repo_id="local/dataset" \
  --dataset.root="local_datasets/dataset" \
  --batch_size=64 \
  --steps=20000 \
  --output_dir="outputs/output_dir" \
  --policy.use_morphology_film=true \
  --policy.morphology_film_num_layers=4 \
  --policy.morphology_film_scale=0.2 \
  --policy.morphology_lr=1e-4 \
  --policy.use_lr_split=true \
  --policy.use_morphology_ranking_loss=true \
  --policy.morphology_rank_probability=0.25 \
  --policy.morphology_rank_weight=0.10 \
  --policy.morphology_rank_margin=0.01 \
  --policy.morphology_hard_negative_ratio=0.70 \
  --policy.morphology_hard_negative_root="assets/robots/generated_hard_negatives" \
  --policy.morphology_negative_urdf_paths="assets/robots/so100.urdf,assets/robots/panda.urdf,assets/robots/xarm7.urdf,assets/robots/vx300s.urdf" \
  --policy.morphology_rank_log_every=100 \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  2>&1 | tee training_log.log
```

The ranking probability and weight can be lowered to reduce training
overhead and should not significantly affect inference performance.

## Dataset

The dataset should use absolute joint positions in radians so that it
follows the same convention as the URDF.

The proprioceptive data should not contain joint-limit violations, which
can occur due to incorrect calibration or other data-collection issues.

For the remaining dataset requirements and recommendations, refer to
SmolVLA.
