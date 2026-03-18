"""Remove still (zero-action) steps from raw teleop zarr datasets.

A step is considered "still" if both the arm joint delta and gripper action delta
are below a threshold (default: 1e-5). These steps are filtered out and the
remaining states are saved to a new zarr dataset with updated episode_ends.

The last state of each episode is always kept (no future state to compute a delta from).

Usage:
    python scripts/filter_still_steps.py --src datasets/raw/multi_cube --dst datasets/raw/multi_cube_filtered
    python scripts/filter_still_steps.py --src datasets/raw/multi_cube --dst datasets/raw/multi_cube_filtered --threshold 1e-4
    python scripts/filter_still_steps.py --src datasets/raw/multi_cube --dst datasets/raw/multi_cube_filtered --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr


def find_keep_indices(
    state_joints: np.ndarray,
    action_gripper: np.ndarray,
    episode_ends: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean mask of states to keep and updated episode_ends.

    Keeps state[t] if the action at t (state[t+1] - state[t]) has any component
    above threshold. Always keeps the last state of each episode.
    """
    keep = np.zeros(len(state_joints), dtype=bool)
    new_episode_ends: list[int] = []
    ep_start = 0
    running = 0

    for ep_end in episode_ends:
        ep_joints = state_joints[ep_start:ep_end]
        ep_gripper = action_gripper[ep_start:ep_end]

        delta_joints = np.abs(ep_joints[1:] - ep_joints[:-1])   # (T-1, 6)
        delta_gripper = np.abs(ep_gripper[1:] - ep_gripper[:-1]) # (T-1, 1)

        # step t is "moving" if any joint or gripper delta exceeds threshold
        moving = (delta_joints.max(axis=1) >= threshold) | (delta_gripper[:, 0] >= threshold)

        # keep indices within episode: all moving steps + last step
        ep_keep = np.concatenate([moving, [True]])  # last state always kept
        keep[ep_start:ep_end] = ep_keep

        running += ep_keep.sum()
        new_episode_ends.append(running)
        ep_start = ep_end

    return keep, np.array(new_episode_ends, dtype=np.int64)


def filter_zarr(src_path: Path, dst_path: Path, threshold: float, dry_run: bool) -> dict:
    """Filter a single zarr file and write to dst_path. Returns stats dict."""
    src = zarr.open(src_path)
    data = src["data"]
    state_joints = data["state_joints"][:]
    action_gripper = data["action_gripper"][:]
    episode_ends = src["meta"]["episode_ends"][:]

    keep, new_episode_ends = find_keep_indices(
        state_joints, action_gripper, episode_ends, threshold
    )

    n_before = len(state_joints)
    n_after = int(keep.sum())
    n_removed = n_before - n_after

    stats = {
        "src": str(src_path),
        "dst": str(dst_path),
        "before": n_before,
        "after": n_after,
        "removed": n_removed,
        "pct_removed": 100.0 * n_removed / n_before if n_before > 0 else 0.0,
    }

    if dry_run:
        return stats

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open(dst_path, mode="w")
    dst_data = dst.require_group("data")
    dst_meta = dst.require_group("meta")

    # copy all data arrays with the keep mask applied
    for key in data.keys():
        src_arr = data[key]
        filtered = src_arr[:][keep]
        dst_data.create_array(
            key,
            data=filtered,
            chunks=src_arr.chunks,
            compressors=src_arr.compressors,
        )

    dst_meta.create_array("episode_ends", data=new_episode_ends)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter still steps from raw teleop zarr datasets.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("datasets/raw/multi_cube"),
        help="Source dataset root directory (default: datasets/raw/multi_cube)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination root directory (default: <src>_filtered)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-5,
        help="Min delta to consider a step as moving (default: 1e-5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing any files",
    )
    args = parser.parse_args()

    src_root = args.src
    dst_root = args.dst if args.dst is not None else src_root.parent / (src_root.name + "_filtered")

    zarr_files = sorted(src_root.rglob("*.zarr"))
    if not zarr_files:
        print(f"No .zarr files found under {src_root}")
        return

    if args.dry_run:
        print(f"[dry-run] threshold={args.threshold}")
    else:
        print(f"src:       {src_root}")
        print(f"dst:       {dst_root}")
        print(f"threshold: {args.threshold}")
        print()

    total_before = total_after = 0

    for src_path in zarr_files:
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel

        stats = filter_zarr(src_path, dst_path, args.threshold, args.dry_run)
        total_before += stats["before"]
        total_after += stats["after"]

        tag = "[dry-run] " if args.dry_run else ""
        print(
            f"{tag}{rel}: "
            f"{stats['removed']} removed / {stats['before']} "
            f"→ {stats['after']} steps ({stats['pct_removed']:.1f}% removed)"
        )

    total_removed = total_before - total_after
    pct = 100.0 * total_removed / total_before if total_before > 0 else 0.0
    print()
    print(f"Total: {total_removed} removed / {total_before} → {total_after} steps ({pct:.1f}% removed)")

    if not args.dry_run:
        print(f"\nFiltered dataset written to: {dst_root}")


if __name__ == "__main__":
    main()
