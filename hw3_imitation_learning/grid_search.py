#!/usr/bin/env python3
"""Overnight grid search over d_model × depth × dropout for HW3 ex1.

Run from hw3_imitation_learning/:
    python grid_search.py

Fixed: chunk_size=16, EPOCHS=200, BATCH_SIZE=64, LR=1e-3, VAL_SPLIT=0.3
Varied: d_model in [384, 512, 640], depth in [4, 5, 6], dropout in [0.0, 0.05, 0.1]
Combos exceeding PARAM_LIMIT (1M) are skipped automatically.
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

D_MODELS = [384, 512, 640]
DEPTHS = [4, 5, 6]
DROPOUTS = [0.0, 0.05, 0.1]

# Soft parameter limit — combos exceeding this are skipped.
PARAM_LIMIT = 1_200_000
# Known dims from the ex1 training command.
_STATE_DIM = 9   # state_ee_xyz(3) + state_gripper(1) + state_cube[:5](5)
_ACTION_DIM = 4  # action_ee_xyz(3) + action_gripper(1)
_CHUNK_SIZE = 16


def param_count(d: int, depth: int) -> int:
    output_dim = _CHUNK_SIZE * _ACTION_DIM
    return (_STATE_DIM + 1) * d + (depth - 1) * (d * d + d) + (d + 1) * output_dim

RESULTS_CSV = Path("grid_search_results.csv")
GRID_CKPT_DIR = Path("checkpoints/grid_search")
SRC_CKPT = Path("checkpoints/single_cube/best_model_ee_xyz_obstacle.pt")

ZARR_PATH = "/Volumes/T9/DevSpace/Github/robot-learning/hw3_imitation_learning/datasets/processed/single_cube/processed_ee_xyz.zarr"

TRAIN_BASE = [
    sys.executable, "scripts/train.py",
    "--zarr", ZARR_PATH,
    "--state-keys", "state_ee_xyz", "state_gripper", "state_cube[:5]",
    "--action-keys", "action_ee_xyz", "action_gripper",
    "--policy", "obstacle",
    "--chunk-size", "16",
]

FIELDNAMES = ["d_model", "depth", "dropout", "success_rate", "score"]


def load_completed() -> set[tuple[int, int, float]]:
    if not RESULTS_CSV.exists():
        return set()
    completed: set[tuple[int, int, float]] = set()
    with RESULTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                completed.add((int(row["d_model"]), int(row["depth"]), float(row["dropout"])))
            except (KeyError, ValueError):
                pass
    return completed


def append_result(d: int, depth: int, dropout: float, success_rate: float | None, score: float | None) -> None:
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "d_model": d,
            "depth": depth,
            "dropout": dropout,
            "success_rate": success_rate,
            "score": score,
        })


def run_combo(d: int, depth: int, dropout: float) -> tuple[float | None, float | None]:
    tag = f"d{d}_l{depth}_dr{dropout}"
    dest_ckpt = GRID_CKPT_DIR / f"{tag}.pt"

    print(f"\n{'='*60}")
    print(f"  RUN: d_model={d}  depth={depth}  dropout={dropout}")
    print(f"{'='*60}", flush=True)

    # Train
    train_cmd = TRAIN_BASE + [
        "--d-model", str(d),
        "--depth", str(depth),
        "--dropout", str(dropout),
    ]
    result = subprocess.run(train_cmd, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] train.py exited with code {result.returncode} — skipping run.", flush=True)
        return None, None

    # Copy checkpoint
    if not SRC_CKPT.exists():
        print(f"  [ERROR] source checkpoint not found: {SRC_CKPT}", flush=True)
        return None, None

    GRID_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_CKPT, dest_ckpt)
    print(f"  Saved checkpoint → {dest_ckpt}", flush=True)

    # Eval
    eval_cmd = [
        sys.executable, "student_eval/run_eval.py",
        "--exercise", "1",
        "--checkpoint", str(dest_ckpt),
    ]
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
    output = eval_result.stdout + eval_result.stderr
    print(output, end="", flush=True)

    success_rate: float | None = None
    score: float | None = None

    m = re.search(r"Success rate:\s*(\d+)/(\d+)", output, re.IGNORECASE)
    if m:
        success_rate = int(m.group(1)) / int(m.group(2))

    m = re.search(r"Score\s*:\s*([\d.]+)", output, re.IGNORECASE)
    if m:
        score = float(m.group(1))

    print(f"  → success_rate={success_rate}  score={score}", flush=True)
    return success_rate, score


def print_leaderboard() -> None:
    if not RESULTS_CSV.exists():
        return
    rows: list[dict] = []
    with RESULTS_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    rows.sort(key=lambda r: float(r["success_rate"]) if r["success_rate"] else -1, reverse=True)
    print(f"\n{'='*60}")
    print("  LEADERBOARD (by success_rate desc)")
    print(f"{'='*60}")
    print(f"  {'d_model':>7}  {'depth':>5}  {'dropout':>7}  {'success':>7}  {'score':>6}")
    print(f"  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}")
    for r in rows:
        sr = f"{float(r['success_rate']):.2%}" if r["success_rate"] else "N/A"
        sc = r["score"] if r["score"] else "N/A"
        print(f"  {r['d_model']:>7}  {r['depth']:>5}  {r['dropout']:>7}  {sr:>7}  {sc:>6}")
    print(f"{'='*60}", flush=True)


def main() -> None:
    completed = load_completed()
    all_combos = list(product(D_MODELS, DEPTHS, DROPOUTS))

    eligible = []
    skipped = []
    for d, dep, dr in all_combos:
        pc = param_count(d, dep)
        if pc > PARAM_LIMIT:
            skipped.append((d, dep, dr, pc))
        else:
            eligible.append((d, dep, dr, pc))

    if skipped:
        print(f"Skipping {len(skipped)} combos exceeding {PARAM_LIMIT:,} params:")
        for d, dep, dr, pc in skipped:
            print(f"  d_model={d} depth={dep} dropout={dr}  ({pc:,} params)")

    remaining = [(d, dep, dr) for d, dep, dr, _ in eligible if (d, dep, dr) not in completed]
    print(f"\nGrid search: {len(eligible)} eligible, {len(completed)} already done, {len(remaining)} to run.")
    for d, dep, dr, pc in eligible:
        tag = "DONE" if (d, dep, dr) in completed else f"{pc:,} params"
        print(f"  d_model={d} depth={dep} dropout={dr}  {tag}")

    for i, (d, depth, dropout) in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Starting run...", flush=True)
        success_rate, score = run_combo(d, depth, dropout)
        append_result(d, depth, dropout, success_rate, score)

    print_leaderboard()


if __name__ == "__main__":
    main()
