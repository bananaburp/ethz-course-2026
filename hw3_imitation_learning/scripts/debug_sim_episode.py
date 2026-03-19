"""Run one closed-loop sim episode and compare predicted vs GT actions.

The model receives real physics states from the simulator (not GT zarr states),
so this is a true closed-loop rollout: each inference call sees the state that
resulted from executing the model's own previous predictions.

GT actions are loaded from the zarr dataset for the specified episode and
overlaid on the same plot for comparison.

Usage:
    python scripts/debug_sim_episode.py \
        --checkpoint ./checkpoints/multi_cube/best_model_ee_xyz_multitask.pt \
        --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
        --episode 0 \
        --n-infer 100 \
        --csv-save debug_sim.csv \
        --plot-dim 2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from so101_gym.constants import ASSETS_DIR

from hw3.dataset import load_zarr
from hw3.eval_utils import apply_action, infer_action_chunk, load_checkpoint
from hw3.sim_env import SO100MulticubeSimEnv, SO100SimEnv

XML_PATH = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"
XML_PATH_MULTICUBE = ASSETS_DIR / "so100_multicube_ee.xml"


def run_sim_episode(
    env,
    model: torch.nn.Module,
    normalizer,
    state_keys: list[str],
    action_keys: list[str],
    chunk_size: int,
    device: torch.device,
    n_infer: int,
) -> list[np.ndarray]:
    """Run a sim episode and return one predicted chunk per inference call.

    Returns a list of arrays each with shape (chunk_size, action_dim).
    """
    obs = env.reset()
    action_queue: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    step = 0

    while len(chunks) < n_infer:
        if not action_queue:
            if len(chunks) >= n_infer:
                break
            chunk = infer_action_chunk(
                model=model,
                normalizer=normalizer,
                obs=obs,
                state_keys=state_keys,
                device=device,
            )
            chunks.append(chunk)
            action_queue.extend(chunk)

        action = action_queue.pop(0)
        apply_action(env, action, action_keys)
        obs = env.step()
        step += 1

    return chunks


def build_records(
    pred_chunks: list[np.ndarray],
    gt_actions: np.ndarray,
    chunk_size: int,
) -> list[dict]:
    """Build per-inference-call stat records from sim predictions and GT zarr actions."""
    records: list[dict] = []
    T = gt_actions.shape[0]
    for i, chunk in enumerate(pred_chunks):
        t0 = i * chunk_size
        t1 = min(t0 + chunk_size, T)
        if t0 >= T:
            # No GT left — still record pred, GT will be zeros
            gt_chunk = np.zeros_like(chunk[:1])
        else:
            gt_chunk = gt_actions[t0:t1]

        records.append({
            "pred_min":  chunk.min(axis=0),
            "pred_mean": chunk.mean(axis=0),
            "pred_max":  chunk.max(axis=0),
            "gt_min":    gt_chunk.min(axis=0),
            "gt_mean":   gt_chunk.mean(axis=0),
            "gt_max":    gt_chunk.max(axis=0),
        })
    return records


def save_records_csv(records: list[dict], path: Path) -> None:
    action_dim = len(records[0]["pred_min"])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["infer_call", "action_dim",
                         "pred_min", "pred_mean", "pred_max",
                         "gt_min", "gt_mean", "gt_max"])
        for call_i, rec in enumerate(records, start=1):
            for d in range(action_dim):
                writer.writerow([
                    call_i, d,
                    rec["pred_min"][d], rec["pred_mean"][d], rec["pred_max"][d],
                    rec["gt_min"][d],   rec["gt_mean"][d],   rec["gt_max"][d],
                ])
    print(f"CSV saved: {path}")


def plot_action_dim(
    records: list[dict],
    dim: int,
    save_path: Path | None = None,
) -> None:
    calls = np.arange(1, len(records) + 1)
    pred_mean = np.array([r["pred_mean"][dim] for r in records])
    pred_min  = np.array([r["pred_min"][dim]  for r in records])
    pred_max  = np.array([r["pred_max"][dim]  for r in records])
    gt_mean   = np.array([r["gt_mean"][dim]   for r in records])
    gt_min    = np.array([r["gt_min"][dim]    for r in records])
    gt_max    = np.array([r["gt_max"][dim]    for r in records])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(calls, pred_min, pred_max, alpha=0.2, color="tomato", label="pred range")
    ax.plot(calls, pred_mean, color="tomato", marker="o", label="pred mean (sim rollout)")
    ax.fill_between(calls, gt_min, gt_max, alpha=0.2, color="steelblue", label="GT range")
    ax.plot(calls, gt_mean, color="steelblue", marker="s", label="GT mean (zarr)")

    ax.set_xlabel("Inference call")
    ax.set_ylabel("Action value (denorm)")
    ax.set_title(f"Action dim {dim} — closed-loop sim vs GT zarr")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop sim episode debug vs GT zarr.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zarr", type=Path, required=True, help="Zarr dataset for GT overlay.")
    parser.add_argument("--episode", type=int, default=0, help="Zarr episode index for GT (default: 0).")
    parser.add_argument("--n-infer", type=int, default=20, help="Max inference calls to run (default: 20).")
    parser.add_argument("--multicube", action="store_true", help="Use multicube scene.")
    parser.add_argument("--goal-cube", type=str, default="red", choices=["red", "green", "blue"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot-dim", type=int, default=None, help="Action dim to plot.")
    parser.add_argument("--plot-save", type=Path, default=None)
    parser.add_argument("--csv-save", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalizer, chunk_size, state_keys, action_keys = load_checkpoint(args.checkpoint, device)

    # Load GT from zarr
    states, actions, ep_ends = load_zarr(args.zarr, state_keys=state_keys, action_keys=action_keys, debug=False)
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_eps = len(ep_ends)
    ep_idx = min(args.episode, n_eps - 1)
    t0, t1 = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
    gt_actions = actions[t0:t1]
    print(f"GT episode {ep_idx}: {t1 - t0} steps, {len(gt_actions) // chunk_size} full chunks")

    # Build sim env — auto-detect multicube from state keys
    _MULTICUBE_KEYS = {"original_pos_cube_red", "original_pos_cube_green", "original_pos_cube_blue", "state_goal"}
    is_multicube = args.multicube or any(
        k.split("[")[0] in _MULTICUBE_KEYS for k in state_keys
    )
    use_mocap = not any("action_joints" in k for k in action_keys)
    if is_multicube:
        env = SO100MulticubeSimEnv(
            xml_path=XML_PATH_MULTICUBE,
            render_w=640, render_h=480,
            use_mocap=use_mocap,
            goal_cube=args.goal_cube,
            shuffle_cubes=False,
            seed=args.seed,
        )
    else:
        env = SO100SimEnv(
            xml_path=XML_PATH,
            render_w=640, render_h=480,
            use_mocap=use_mocap,
            seed=args.seed,
        )
    print(f"Sim env: {'multicube' if is_multicube else 'single-cube'}, use_mocap={use_mocap}")

    print(f"\nRunning closed-loop sim episode (up to {args.n_infer} inference calls)...")
    pred_chunks = run_sim_episode(
        env=env,
        model=model,
        normalizer=normalizer,
        state_keys=state_keys,
        action_keys=action_keys,
        chunk_size=chunk_size,
        device=device,
        n_infer=args.n_infer,
    )
    print(f"Collected {len(pred_chunks)} inference calls.")

    records = build_records(pred_chunks, gt_actions, chunk_size)

    if args.csv_save is not None:
        save_records_csv(records, args.csv_save)

    if args.plot_dim is not None:
        plot_action_dim(records, dim=args.plot_dim, save_path=args.plot_save)


if __name__ == "__main__":
    main()
