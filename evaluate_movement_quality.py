# USAGE  python evaluate_movement_quality.py --root trajectory_results
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
    from scipy.stats import mannwhitneyu
except ImportError as exc:
    raise SystemExit("This script requires scipy: pip install scipy") from exc


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

CONDITIONS = [
    ("pink", "solo"),
    ("black", "solo"),
    ("pink", "both"),
    ("black", "both"),
]

VALID_TRIALS = {
    ("pink", "solo"): 10,
    ("black", "solo"): 10,
    ("pink", "both"): 10,
    ("black", "both"): 4,
}

MODEL_PREFIXES = {
    "Ours": "Ours_20k_b64",
    "Baseline": "Baseline_20k_b64",
}

PAPER_METRICS = [
    (
        "active_motion_fraction",
        "Active motion fraction",
    ),
    (
        "max_displacement_from_start_rad",
        "Maximum displacement from start",
    ),
    (
        "progress_efficiency",
        "Progress efficiency",
    ),
    (
        "endpoint_directness",
        "Endpoint directness",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the movement metrics reported in the paper."
    )
    parser.add_argument(
        "--root",
        default="trajectory_results",
        help="Directory containing the evaluation condition folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="trajectory_results/paper_movement_metrics",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--active-speed-threshold-rad-s",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260912,
    )
    return parser.parse_args()


def trial_number(path: Path) -> int | None:
    parts = path.stem.split("_")

    if len(parts) >= 3 and parts[0] == "trial":
        try:
            return int(parts[1])
        except ValueError:
            return None

    return None


def discover_sources(condition_dir: Path):
    sources = []

    direct = sorted(condition_dir.glob("trial_*_trajectory.csv"))
    if direct:
        sources.append((condition_dir, direct))

    for run_dir in sorted(condition_dir.glob("so101_rollout_*")):
        if not run_dir.is_dir():
            continue

        files = sorted(run_dir.glob("trial_*_trajectory.csv"))
        if files:
            sources.append((run_dir, files))

    return sources


def select_trial_csvs(
    condition_dir: Path,
    expected: int,
    first_n_only: bool,
):
    if not condition_dir.exists():
        return [], "missing"

    sources = discover_sources(condition_dir)

    if not sources:
        return [], "no_csv"

    required = set(range(1, expected + 1))

    # For black-both, only trials 1-4 are valid for the paper.
    if first_n_only:
        usable = []

        for source, files in sources:
            by_number = {
                trial_number(path): path
                for path in files
                if trial_number(path) is not None
            }

            if required.issubset(by_number):
                usable.append(
                    (
                        source,
                        [by_number[i] for i in range(1, expected + 1)],
                    )
                )

        if not usable:
            return [], "missing_required_trials"

        source, files = sorted(
            usable,
            key=lambda item: item[0].name,
        )[-1]

        return files, f"trials_001_to_{expected:03d}:{source.name}"

    # Prefer a complete run containing exactly the expected number.
    exact = [
        (source, files)
        for source, files in sources
        if len(files) == expected
    ]

    if exact:
        source, files = sorted(
            exact,
            key=lambda item: item[0].name,
        )[-1]

        return files, f"newest_complete:{source.name}"

    # Otherwise combine trials across restarted runs.
    by_number = {}
    by_source = {}

    for source, files in sources:
        for path in files:
            number = trial_number(path)

            if number is None:
                continue

            if (
                number not in by_number
                or source.name > by_source[number].name
            ):
                by_number[number] = path
                by_source[number] = source

    if required.issubset(by_number):
        return (
            [by_number[i] for i in range(1, expected + 1)],
            "combined_restarts",
        )

    return [], "ambiguous_or_incomplete"


def load_trial(path: Path) -> np.ndarray:
    df = pd.read_csv(path)

    joint_columns = [
        f"measured_{joint}_rad"
        for joint in ARM_JOINTS
    ]

    missing = [
        column
        for column in joint_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: missing measured joint columns: {missing}"
        )

    q_df = df[joint_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid = q_df.notna().all(axis=1)
    q = q_df.loc[valid].to_numpy(dtype=float)

    finite = np.isfinite(q).all(axis=1)
    return q[finite]


def smooth_q(
    q: np.ndarray,
    window: int,
    polyorder: int,
) -> np.ndarray:
    if len(q) < 5:
        return q.copy()

    window = int(window)

    if window % 2 == 0:
        window += 1

    max_window = (
        len(q)
        if len(q) % 2 == 1
        else len(q) - 1
    )

    window = min(window, max_window)

    if window <= polyorder:
        return q.copy()

    return savgol_filter(
        q,
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )


def analyze_trial(
    path: Path,
    model: str,
    color: str,
    scene: str,
    args,
) -> dict:
    q = load_trial(path)

    if len(q) < 2:
        raise ValueError(f"Too few samples in {path}")

    q = smooth_q(
        q,
        args.smooth_window,
        args.smooth_polyorder,
    )

    # Joint-space displacement from the initial configuration.
    displacement = np.linalg.norm(
        q - q[0],
        axis=1,
    )

    max_displacement = float(
        np.max(displacement)
    )

    final_displacement = float(
        np.linalg.norm(q[-1] - q[0])
    )

    # Total joint-space path length.
    step_distance = np.linalg.norm(
        np.diff(q, axis=0),
        axis=1,
    )

    path_length = float(
        np.sum(step_distance)
    )

    progress_efficiency = (
        max_displacement / path_length
        if path_length > 1e-12
        else np.nan
    )

    endpoint_directness = (
        final_displacement / path_length
        if path_length > 1e-12
        else np.nan
    )

    # Active-motion fraction.
    dt = 1.0 / args.control_hz

    velocity = np.diff(q, axis=0) / dt

    speed = np.linalg.norm(
        velocity,
        axis=1,
    )

    active_motion_fraction = (
        float(
            np.mean(
                speed >= args.active_speed_threshold_rad_s
            )
        )
        if len(speed)
        else 0.0
    )

    return {
        "model": model,
        "color": color,
        "scene": scene,
        "condition": f"{color}_{scene}",
        "trial_number": trial_number(path),
        "trajectory_csv": str(path),

        "active_motion_fraction": active_motion_fraction,
        "max_displacement_from_start_rad": max_displacement,
        "progress_efficiency": progress_efficiency,
        "endpoint_directness": endpoint_directness,
    }


def cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    greater = 0
    lower = 0

    for value in x:
        greater += int(np.sum(value > y))
        lower += int(np.sum(value < y))

    return float(
        (greater - lower) / (len(x) * len(y))
    )


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full_like(
        p_values,
        np.nan,
    )

    finite = np.where(
        np.isfinite(p_values)
    )[0]

    if len(finite) == 0:
        return adjusted.tolist()

    order = finite[
        np.argsort(p_values[finite])
    ]

    m = len(order)
    running = 0.0

    for rank, index in enumerate(order):
        value = (
            (m - rank)
            * p_values[index]
        )

        running = max(
            running,
            value,
        )

        adjusted[index] = min(
            1.0,
            running,
        )

    return adjusted.tolist()


def bootstrap_median_diff(
    x,
    y,
    samples,
    rng,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan

    observed = float(
        np.median(x) - np.median(y)
    )

    x_indices = rng.integers(
        0,
        len(x),
        size=(samples, len(x)),
    )

    y_indices = rng.integers(
        0,
        len(y),
        size=(samples, len(y)),
    )

    differences = (
        np.median(x[x_indices], axis=1)
        - np.median(y[y_indices], axis=1)
    )

    return (
        observed,
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )


def compare_metrics(
    df: pd.DataFrame,
    rng,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []

    for column, label in PAPER_METRICS:
        ours = pd.to_numeric(
            df.loc[
                df["model"] == "Ours",
                column,
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        baseline = pd.to_numeric(
            df.loc[
                df["model"] == "Baseline",
                column,
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        test = mannwhitneyu(
            ours,
            baseline,
            alternative="two-sided",
            method="auto",
        )

        median_diff, ci_low, ci_high = (
            bootstrap_median_diff(
                ours,
                baseline,
                bootstrap_samples,
                rng,
            )
        )

        rows.append(
            {
                "metric": column,
                "label": label,

                "ours_n": len(ours),
                "ours_median": float(np.median(ours)),
                "ours_q1": float(np.percentile(ours, 25)),
                "ours_q3": float(np.percentile(ours, 75)),

                "baseline_n": len(baseline),
                "baseline_median": float(np.median(baseline)),
                "baseline_q1": float(
                    np.percentile(baseline, 25)
                ),
                "baseline_q3": float(
                    np.percentile(baseline, 75)
                ),

                "median_difference_ours_minus_baseline": median_diff,
                "median_difference_ci_low": ci_low,
                "median_difference_ci_high": ci_high,

                "cliffs_delta": cliffs_delta(
                    ours,
                    baseline,
                ),

                "mann_whitney_u": float(
                    test.statistic
                ),

                "p_value": float(
                    test.pvalue
                ),
            }
        )

    adjusted = holm_adjust(
        [row["p_value"] for row in rows]
    )

    for row, adjusted_p in zip(
        rows,
        adjusted,
    ):
        row["holm_adjusted_p"] = adjusted_p

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")

    root = Path(args.root).expanduser()
    output_dir = Path(
        args.output_dir
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    preflight = []

    for model, prefix in MODEL_PREFIXES.items():
        for color, scene in CONDITIONS:
            expected = VALID_TRIALS[
                (color, scene)
            ]

            condition_dir = (
                root
                / f"{prefix}_{color}_{scene}_n30"
            )

            files, selection = select_trial_csvs(
                condition_dir,
                expected,
                first_n_only=(
                    color == "black"
                    and scene == "both"
                ),
            )

            preflight.append(
                {
                    "model": model,
                    "color": color,
                    "scene": scene,
                    "expected": expected,
                    "selected": len(files),
                    "selection": selection,
                    "condition_dir": str(condition_dir),
                }
            )

            if len(files) != expected:
                raise RuntimeError(
                    f"{model} {color}/{scene}: "
                    f"expected {expected}, "
                    f"found {len(files)} "
                    f"({selection})"
                )

            for path in files:
                rows.append(
                    analyze_trial(
                        path,
                        model,
                        color,
                        scene,
                        args,
                    )
                )

    preflight_df = pd.DataFrame(
        preflight
    )

    preflight_df.to_csv(
        output_dir / "paper_metrics_preflight.csv",
        index=False,
    )

    trials = pd.DataFrame(rows)

    counts = (
        trials.groupby("model")
        .size()
        .to_dict()
    )

    if (
        counts.get("Ours") != 34
        or counts.get("Baseline") != 34
    ):
        raise RuntimeError(
            f"Expected 34 trials/model, got {counts}"
        )

    trials.to_csv(
        output_dir / "paper_metrics_per_trial.csv",
        index=False,
    )

    rng = np.random.default_rng(
        args.seed
    )

    comparison = compare_metrics(
        trials,
        rng,
        args.bootstrap_samples,
    )

    comparison.to_csv(
        output_dir / "paper_metrics_comparison.csv",
        index=False,
    )

    print()
    print("=" * 84)
    print("PAPER MOVEMENT METRICS")
    print("=" * 84)
    print(
        "34 trials/model: "
        "10 pink-solo + 10 black-solo + "
        "10 pink-both + 4 black-both"
    )
    print()

    for _, row in comparison.iterrows():
        print(row["label"])
        print(
            f"  Ours     : "
            f"{row['ours_median']:.4f}"
        )
        print(
            f"  Baseline : "
            f"{row['baseline_median']:.4f}"
        )
        print(
            f"  Cliff's delta : "
            f"{row['cliffs_delta']:.3f}"
        )
        print(
            f"  Mann-Whitney p: "
            f"{row['p_value']:.8g}"
        )
        print(
            f"  Holm p        : "
            f"{row['holm_adjusted_p']:.8g}"
        )
        print()

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()