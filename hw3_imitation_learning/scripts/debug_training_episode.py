"""Debug print for one episode from training data.

Loads a checkpoint, picks the first episode from a zarr, and prints
the same per-inference-call debug output that eval.py emits at runtime.

Usage:
    python scripts/debug_training_episode.py \
        --checkpoint ./checkpoints/multi_cube/best_model_ee_xyz_multitask.pt \
        --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
        --n-infer 100 --csv-save debug_all_eps.csv
        --episode 50 --csv-save debug_eps0.csv --n-infer 10 --plot-dim 2

    Omit --episode to plot the mean over all episodes.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hw3.dataset import Normalizer, load_zarr
from hw3.eval_utils import load_checkpoint, parse_key_spec


def _expand_key_labels(state_keys: list[str]) -> list[str]:
    """Return one label per state dimension, e.g. 'state_ee_xyz[0]'."""
    labels: list[str] = []
    for spec in state_keys:
        name, col_slice = parse_key_spec(spec)
        # Determine width from a dummy zero vector — we just need the slice size.
        # We'll fill actual sizes from the data itself.
        labels.append(spec)  # placeholder replaced below
    return labels


def plot_action_dim(
    records: list[dict],
    dim: int,
    save_path: Path | None = None,
) -> None:
    """Plot denorm min/mean/max for GT and prediction across inference calls.

    ``records`` is the list returned by ``print_episode_debug``.
    Each entry has keys: ``pred_min``, ``pred_mean``, ``pred_max``,
    ``gt_min``, ``gt_mean``, ``gt_max`` — all arrays of shape (action_dim,).
    """
    calls = np.arange(1, len(records) + 1)
    pred_min  = np.array([r["pred_min"][dim]  for r in records])
    pred_mean = np.array([r["pred_mean"][dim] for r in records])
    pred_max  = np.array([r["pred_max"][dim]  for r in records])
    gt_min    = np.array([r["gt_min"][dim]    for r in records])
    gt_mean   = np.array([r["gt_mean"][dim]   for r in records])
    gt_max    = np.array([r["gt_max"][dim]    for r in records])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(calls, pred_min, pred_max, alpha=0.2, color="tomato", label="pred range")
    ax.plot(calls, pred_mean, color="tomato", marker="o", label="pred mean")
    ax.fill_between(calls, gt_min, gt_max, alpha=0.2, color="steelblue", label="GT range")
    ax.plot(calls, gt_mean, color="steelblue", marker="s", label="GT mean")

    ax.set_xlabel("Inference call")
    ax.set_ylabel("Denorm action value")
    ax.set_title(f"Action dim {dim} — predicted vs GT (denorm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def save_records_csv(records: list[dict], path: Path) -> None:
    """Write per-inference-call stats to a tidy CSV.

    Columns: infer_call, action_dim, pred_min, pred_mean, pred_max,
             gt_min, gt_mean, gt_max
    """
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


def print_episode_debug(
    states_raw: np.ndarray,       # (T, state_dim) raw (unnormalised)
    actions_raw: np.ndarray,      # (T, action_dim) raw (unnormalised)
    model: torch.nn.Module,
    normalizer: Normalizer,
    chunk_size: int,
    state_keys: list[str],
    device: torch.device,
    n_infer: int = 3,
    verbose: bool = True,
) -> list[dict]:
    T = states_raw.shape[0]
    action_dim = actions_raw.shape[1]

    # Build per-dim state labels
    dim_labels: list[str] = []
    for spec in state_keys:
        _, col_slice = parse_key_spec(spec)
        # figure out width: normalize one row to get state_dim, then count cols
        # We just run through the specs and count:
        dummy = np.zeros(states_raw.shape[1], dtype=np.float32)
        # We'll build labels by scanning specs — need widths
        # Instead, count how many dims each spec contributes from the raw data
        # We do this lazily when we know the total dim
        dim_labels.append(spec)

    # Rebuild proper labels with per-element suffixes
    per_dim_labels: list[str] = []
    col = 0
    state_dim = states_raw.shape[1]
    # We need widths per spec. Use a greedy scan matching total dim.
    # Strategy: run specs and try to infer width from state_dim / spec count.
    # Since we can't re-read zarr here, we infer widths from known key dims.
    _KEY_DIMS: dict[str, int] = {
        "state_ee_xyz": 3,
        "state_ee_full": 7,
        "state_joints": 5,
        "state_gripper": 1,
        "state_cube": 7,
        "state_obstacle": 7,
        "goal_pos": 3,
        "original_pos_cube_red": 7,
        "original_pos_cube_green": 7,
        "original_pos_cube_blue": 7,
        "state_goal": 3,
        "_rel_goal_pos": 3,
        "_rel_cube_pos": 3,
    }
    for spec in state_keys:
        name, col_slice = parse_key_spec(spec)
        full_dim = _KEY_DIMS.get(name, 1)
        indices = np.arange(full_dim)[col_slice]
        for i in indices:
            per_dim_labels.append(f"{spec}[{i}]")

    # If inferred labels don't match state_dim, fall back to plain indices
    if len(per_dim_labels) != state_dim:
        per_dim_labels = [f"dim_{i}" for i in range(state_dim)]

    if verbose:
        print(f"  action normalizer per dim (mean / std):")
        for d in range(action_dim):
            print(f"    dim {d:2d}: mean={normalizer.action_mean[d]:+.5f}  std={normalizer.action_std[d]:.5f}")

    records: list[dict] = []
    infer_count = 0
    step = 0
    while step < T and infer_count < n_infer:
        state_raw = states_raw[step]
        state_norm = normalizer.normalize_state(state_raw)
        state_t = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.sample_actions(state_t)

        chunk_norm = pred.squeeze(0).cpu().numpy()   # (chunk_size, action_dim)
        chunk_denorm = chunk_norm.copy()
        for i in range(chunk_norm.shape[0]):
            chunk_denorm[i] = normalizer.denormalize_action(chunk_norm[i])

        infer_count += 1
        gt_end = min(step + chunk_size, T)
        gt_chunk = actions_raw[step:gt_end]   # raw (unnormalised)
        gt_norm = np.stack([normalizer.normalize_action(gt_chunk[i]) for i in range(len(gt_chunk))])

        if verbose:
            print(f"\n[infer call {infer_count}]  state: min={state_raw.min():.3f}  max={state_raw.max():.3f}"
                  f"  norm: min={state_norm.min():.3f}  max={state_norm.max():.3f}  mean={state_norm.mean():.3f}")
            # print(f"  --- state input ---")
            # for dim_i, label in enumerate(per_dim_labels):
            #     print(f"    [{dim_i:3d}] {label:<42s}  raw={state_raw[dim_i]:+.6f}  norm={state_norm[dim_i]:+.6f}")

            print(f"  --- predicted action chunk ---")
            print(f"  {'dim':<4}  {'norm_min':>9}  {'norm_mean':>9}  {'norm_max':>9}  |"
                  f"  {'denorm_min':>11}  {'denorm_mean':>11}  {'denorm_max':>11}")
            for d in range(action_dim):
                nv = chunk_norm[:, d]
                dv = chunk_denorm[:, d]
                print(f"  {d:<4}  {nv.min():>+9.4f}  {nv.mean():>+9.4f}  {nv.max():>+9.4f}  |"
                      f"  {dv.min():>+11.5f}  {dv.mean():>+11.5f}  {dv.max():>+11.5f}")

            print(f"  --- ground truth action chunk (steps {step}–{gt_end-1}) ---")
            print(f"  {'dim':<4}  {'norm_min':>9}  {'norm_mean':>9}  {'norm_max':>9}  |"
                  f"  {'raw_min':>11}  {'raw_mean':>11}  {'raw_max':>11}")
            for d in range(action_dim):
                nv = gt_norm[:, d]
                rv = gt_chunk[:, d]
                print(f"  {d:<4}  {nv.min():>+9.4f}  {nv.mean():>+9.4f}  {nv.max():>+9.4f}  |"
                      f"  {rv.min():>+11.5f}  {rv.mean():>+11.5f}  {rv.max():>+11.5f}")

        records.append({
            "pred_min":  chunk_denorm.min(axis=0),
            "pred_mean": chunk_denorm.mean(axis=0),
            "pred_max":  chunk_denorm.max(axis=0),
            "gt_min":    gt_chunk.min(axis=0),
            "gt_mean":   gt_chunk.mean(axis=0),
            "gt_max":    gt_chunk.max(axis=0),
        })

        step += chunk_size

    return records


def mean_records(all_records: list[list[dict]]) -> list[dict]:
    """Average per-inference-call records across episodes.

    Episodes with fewer inference calls than the maximum are skipped for the
    later calls (i.e. we only average over episodes that have a given call).
    """
    max_calls = max(len(r) for r in all_records)
    averaged: list[dict] = []
    for i in range(max_calls):
        ep_recs = [r[i] for r in all_records if i < len(r)]
        keys = list(ep_recs[0].keys())
        averaged.append({k: np.mean(np.stack([r[k] for r in ep_recs]), axis=0) for k in keys})
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug print from training data episode.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=None, help="Episode index (default: mean over all episodes)")
    parser.add_argument("--n-infer", type=int, default=5, help="Number of inference calls to print (default: 3)")
    parser.add_argument("--plot-dim", type=int, default=None, help="Action dim to plot (omit to skip plotting)")
    parser.add_argument("--plot-save", type=Path, default=None, help="Save plot to path instead of showing")
    parser.add_argument("--csv-save", type=Path, default=None, help="Save per-inference-call stats to a CSV file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalizer, chunk_size, state_keys, action_keys = load_checkpoint(args.checkpoint, device)

    states, actions, ep_ends = load_zarr(
        args.zarr,
        state_keys=state_keys,
        action_keys=action_keys,
        debug=False,
    )

    # Extract episode slice(s)
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_eps = len(ep_ends)

    if args.episode is None:
        # Mean over all episodes
        print(f"\n═══ Mean over all {n_eps} episodes ═══\n")
        all_records: list[list[dict]] = []
        for ep_idx in range(n_eps):
            t0, t1 = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
            ep_records = print_episode_debug(
                states[t0:t1], actions[t0:t1],
                model, normalizer, chunk_size, state_keys, device,
                n_infer=args.n_infer,
                verbose=False,
            )
            if ep_records:
                all_records.append(ep_records)
        records = mean_records(all_records)
        print(f"Averaged {len(all_records)} episodes, {len(records)} inference calls each (max).")
    else:
        if args.episode >= n_eps:
            print(f"Episode {args.episode} out of range (dataset has {n_eps} episodes). Using 0.")
            ep_idx = 0
        else:
            ep_idx = args.episode
        t0, t1 = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
        print(f"\n═══ Episode {ep_idx} ({t1 - t0} steps) ═══\n")
        records = print_episode_debug(
            states[t0:t1], actions[t0:t1],
            model, normalizer, chunk_size, state_keys, device,
            n_infer=args.n_infer,
        )

    if args.csv_save is not None:
        save_records_csv(records, args.csv_save)

    if args.plot_dim is not None:
        plot_action_dim(records, dim=args.plot_dim, save_path=args.plot_save)


if __name__ == "__main__":
    main()
