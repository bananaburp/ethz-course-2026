"""
Plot per-run training curves for all PPO runs under logs/ppo/.

Each run gets its own figure with subplots for every logged metric.
Also prints the current PPO config from ex3_ppo_config.py.

Usage:
    python scripts/plot_training_curves.py [--log_dir logs/ppo] [--save_dir logs/plots]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from exercises.ex3_ppo_config import PPO_PARAMETERS


# Metrics to plot, grouped by subplot row
METRIC_GROUPS = [
    # (tag, y-label, title)
    ("eval/return",             "Return",         "Eval Return"),
    ("eval/ee_tracking_error",  "Error (m)",      "EE Tracking Error"),
    ("eval/length",             "Steps",          "Eval Episode Length"),
    ("train/mean_kl",           "KL",             "Mean KL Divergence"),
    ("train/mean_surrogate_loss","Loss",           "Surrogate Loss"),
    ("train/mean_value_loss",   "Loss",           "Value Loss"),
    ("train/mean_entropy",      "Entropy",        "Policy Entropy"),
    ("train/global_action_std", "Std",            "Action Std"),
    ("train/learning_rate",     "LR",             "Learning Rate"),
]

NCOLS = 3


def load_scalars(event_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {tag: (steps, values)} from the TensorBoard event file in event_dir."""
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag, _, _ in METRIC_GROUPS:
        if tag in available:
            events = ea.Scalars(tag)
            steps  = np.array([e.step  for e in events])
            values = np.array([e.value for e in events])
            result[tag] = (steps, values)
    return result


def infer_config_from_data(scalars: dict) -> dict:
    """Extract any hyperparams that were actually logged (learning_rate, std)."""
    inferred = {}
    if "train/learning_rate" in scalars:
        _, vals = scalars["train/learning_rate"]
        if len(vals):
            inferred["learning_rate (logged)"] = f"{vals[0]:.2e}"
    if "train/global_action_std" in scalars:
        _, vals = scalars["train/global_action_std"]
        if len(vals):
            inferred["init_action_std (logged)"] = f"{vals[0]:.4f}"
    return inferred


def plot_run(run_dir: Path, scalars: dict, save_dir: Path | None) -> None:
    nrows = int(np.ceil(len(METRIC_GROUPS) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(6 * NCOLS, 4 * nrows))
    axes = axes.flatten()

    fig.suptitle(f"Run: {run_dir.name}", fontsize=14, fontweight="bold", y=1.01)

    for idx, (tag, ylabel, title) in enumerate(METRIC_GROUPS):
        ax = axes[idx]
        if tag in scalars:
            steps, values = scalars[tag]
            ax.plot(steps, values, linewidth=1.5)
            # smoothed overlay
            if len(values) >= 10:
                kernel = max(1, len(values) // 20)
                smoothed = np.convolve(values, np.ones(kernel) / kernel, mode="valid")
                smooth_steps = steps[kernel - 1:]
                ax.plot(smooth_steps, smoothed, linewidth=2, color="red",
                        alpha=0.8, label="smoothed")
                ax.legend(fontsize=8)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
        else:
            ax.text(0.5, 0.5, "not logged", ha="center", va="center",
                    transform=ax.transAxes, color="grey")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(len(METRIC_GROUPS), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{run_dir.name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  Saved → {out}")
    else:
        plt.show()

    plt.close(fig)


def print_config(run_configs: list[tuple[str, dict]]) -> None:
    print("\n" + "=" * 60)
    print("CURRENT PPO CONFIG  (from ex3_ppo_config.py)")
    print("=" * 60)
    for key, val in PPO_PARAMETERS.items():
        print(f"  {key:<25} {val}")

    print("\n" + "=" * 60)
    print("PER-RUN LOGGED HYPERPARAMS")
    print("  (only params that TensorBoard actually recorded)")
    print("=" * 60)
    for run_name, inferred in run_configs:
        print(f"\n  {run_name}")
        if inferred:
            for k, v in inferred.items():
                print(f"    {k:<30} {v}")
        else:
            print("    (no hyperparams logged)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training curves per run.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/ppo",
        help="Root directory containing PPO run subdirectories.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/ppo/plots",
        help="Directory to save PNG plots. If omitted, shows interactive windows.",
    )
    args = parser.parse_args()

    log_dir = (ROOT_DIR / args.log_dir).resolve()
    save_dir = Path(args.save_dir).resolve() if args.save_dir else None

    if not log_dir.exists():
        print(f"[ERROR] Log directory not found: {log_dir}")
        sys.exit(1)

    run_dirs = sorted(
        [p for p in log_dir.iterdir() if p.is_dir() and p.name != "eval"],
        key=lambda p: p.name,
    )

    if not run_dirs:
        print(f"[ERROR] No run directories found under: {log_dir}")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run(s) under {log_dir}")

    run_configs = []
    for run_dir in run_dirs:
        print(f"\n  Processing: {run_dir.name}")
        try:
            scalars = load_scalars(run_dir)
        except Exception as exc:
            print(f"    [WARN] Could not read TensorBoard events: {exc}")
            run_configs.append((run_dir.name, {}))
            continue

        if not scalars:
            print("    [WARN] No scalar data found — skipping plot.")
            run_configs.append((run_dir.name, {}))
            continue

        inferred = infer_config_from_data(scalars)
        run_configs.append((run_dir.name, inferred))

        total_iters = max(steps[-1] for steps, _ in scalars.values() if len(steps))
        print(f"    Iterations logged: {int(total_iters)}")
        print(f"    Metrics available: {sorted(scalars.keys())}")

        plot_run(run_dir, scalars, save_dir)

    print_config(run_configs)


if __name__ == "__main__":
    main()
