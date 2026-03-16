#!/usr/bin/env python3
"""Student-facing evaluation script for HW3 Exercises 1–3.

Usage
-----
    python student_eval/run_eval.py --exercise 1 --checkpoint ./checkpoints/single_cube/best_model_ee_full_obstacle.pt
    python student_eval/run_eval.py --exercise 2 --checkpoint ./checkpoints/single_cube/best_model_ee_full_obstacle.pt
    python student_eval/run_eval.py --exercise 3 --checkpoint ./checkpoints/multicube/best_model_multicube.pt

The script expects your ``model.py`` at ``hw3/model.py`` relative to the
project root (i.e. the parent directory of ``student_eval/``).

This script imports the **compiled** ``eval_harness`` module (.so / .pyd)
which lives in the same directory.  Do NOT modify or replace it.

The script will:
  1. Load your model definition from ``./model.py``
  2. Load the trained weights from the checkpoint
  3. Run 100 headless simulation episodes (seed=42)
  4. Print your success rate and score
  5. Write a signed ``ex{N}_result.hwresult`` file

Upload the ``.hwresult`` file(s) to Gradescope.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

# Force MPS on Apple Silicon before the compiled harness selects a device.
# The harness uses:  device = "cuda" if cuda else "cpu"
# We intercept every path that could pin tensors/modules to CPU:
#   1. torch.load  – override map_location to MPS (even when harness passes "cpu" explicitly)
#   2. Tensor.to   – redirect .to("cpu") → .to("mps")
#   3. Module.to   – redirect .to("cpu") → .to("mps")
# PYTORCH_ENABLE_MPS_FALLBACK lets unsupported MPS ops fall back silently.
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _mps = torch.device("mps")

    def _is_cpu(d: object) -> bool:
        return (isinstance(d, str) and d.startswith("cpu")) or (
            isinstance(d, torch.device) and d.type == "cpu"
        )

    # 1. torch.load – force map_location=mps regardless of what the harness passes
    _orig_load = torch.load

    def _mps_load(*args, **kwargs):
        kwargs["map_location"] = _mps
        return _orig_load(*args, **kwargs)

    torch.load = _mps_load  # type: ignore[assignment]

    # 2. Tensor.to – redirect cpu device to mps
    _orig_tensor_to = torch.Tensor.to

    def _mps_tensor_to(self, *args, **kwargs):  # type: ignore[misc]
        if args and _is_cpu(args[0]):
            args = (_mps,) + args[1:]
        if "device" in kwargs and _is_cpu(kwargs["device"]):
            kwargs["device"] = _mps
        return _orig_tensor_to(self, *args, **kwargs)

    torch.Tensor.to = _mps_tensor_to  # type: ignore[method-assign]

    # 3. Module.to – redirect cpu device to mps
    _orig_module_to = torch.nn.Module.to

    def _mps_module_to(self, *args, **kwargs):  # type: ignore[misc]
        if args and _is_cpu(args[0]):
            args = (_mps,) + args[1:]
        if "device" in kwargs and _is_cpu(kwargs["device"]):
            kwargs["device"] = _mps
        return _orig_module_to(self, *args, **kwargs)

    torch.nn.Module.to = _mps_module_to  # type: ignore[method-assign]

_EX_INFO = {
    1: {"name": "Single-Cube Obstacle (train)", "default_ckpt": "ex1.pt"},
    2: {"name": "Single-Cube Obstacle (adversarial)", "default_ckpt": "ex2.pt"},
    3: {"name": "Multicube Goal-Conditioned", "default_ckpt": "ex3.pt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HW3 – Local Evaluation (Exercises 1–3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exercise",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Exercise number to evaluate (1, 2, or 3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to your checkpoint (default: ./ex{N}.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the signed result file "
        "(default: ./ex{N}_result.hwresult)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Do NOT change for official submission.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress output.",
    )
    args = parser.parse_args()

    ex = args.exercise
    info = _EX_INFO[ex]

    # Defaults that depend on exercise number
    ckpt = args.checkpoint or info["default_ckpt"]
    output = args.output or f"ex{ex}_result.hwresult"

    # Resolve paths – model.py is always at <project_root>/hw3/model.py
    # project_root = parent of student_eval/ (where this script lives)
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "hw3" / "model.py"
    ckpt_path = Path(ckpt).resolve()
    output_path = Path(output).resolve()

    if not model_path.exists():
        print(
            f"ERROR: model.py not found at {model_path}\n"
            "       Expected hw3/model.py relative to the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Import the compiled harness (the .so / .pyd in this directory)
    harness_dir = Path(__file__).resolve().parent
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    try:
        import eval_harness  # noqa: E402  — compiled .so
    except ImportError as e:
        print(
            "ERROR: Could not import eval_harness.\n"
            "Make sure the compiled eval_harness*.so (or .pyd on Windows)\n"
            "is in the same directory as this script.\n"
            f"\nDetails: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("=" * 55)
    print(f"  HW3 Exercise {ex} – {info['name']}")
    print("=" * 55)
    print(f"  Model      : {model_path}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {output_path}")
    print(f"  Episodes   : {args.num_episodes}")
    print(f"  Seed       : {args.seed}")
    print("=" * 55)

    class _Tee(io.TextIOBase):
        """Write to both real stdout and a buffer simultaneously."""
        def __init__(self, real: io.TextIOWrapper) -> None:
            self._real = real
            self._buf = io.StringIO()
        def write(self, s: str) -> int:
            self._real.write(s)
            self._real.flush()
            return self._buf.write(s)
        def flush(self) -> None:
            self._real.flush()

    tee = _Tee(sys.__stdout__)
    with redirect_stdout(tee):
        eval_harness.run_eval(
            exercise=ex,
            model_py=str(model_path),
            checkpoint=str(ckpt_path),
            output_path=str(output_path),
            num_episodes=args.num_episodes,
            seed=args.seed,
            verbose=not args.quiet,
        )
    harness_output = tee._buf.getvalue()

    # Parse success rate and score from harness summary lines:
    #   "  Success rate: 59/100 (59.0%)"
    #   "  Score       : 40/100"
    success_rate: float | None = None
    score: float | None = None
    m = re.search(r"Success rate:\s*(\d+)/(\d+)", harness_output, re.IGNORECASE)
    if m:
        success_rate = int(m.group(1)) / int(m.group(2))
    m = re.search(r"Score\s*:\s*([\d.]+)", harness_output, re.IGNORECASE)
    if m:
        score = float(m.group(1))

    # Load checkpoint hyperparams (best-effort)
    ckpt_meta: dict = {}
    try:
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        for key in ("policy_type", "chunk_size", "d_model", "depth", "dropout",
                    "state_dim", "action_dim", "state_keys", "action_keys",
                    "epoch", "val_loss", "epochs", "batch_size", "lr", "val_split"):
            if key in ckpt_data:
                ckpt_meta[key] = ckpt_data[key]
    except Exception:
        pass

    # Append row to eval_log.csv (never overwritten)
    log_path = Path(__file__).resolve().parent / "eval_log.csv"
    fieldnames = [
        "timestamp", "exercise",
        "success_rate", "score",
        "chunk_size", "d_model", "depth", "dropout",
        "best_epoch", "best_val_loss",
        "epochs", "batch_size", "lr", "val_split",
    ]
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "exercise": ex,
            "success_rate": success_rate,
            "score": score,
            "chunk_size": ckpt_meta.get("chunk_size"),
            "d_model": ckpt_meta.get("d_model"),
            "depth": ckpt_meta.get("depth"),
            "dropout": ckpt_meta.get("dropout"),
            "best_epoch": ckpt_meta.get("epoch"),
            "best_val_loss": ckpt_meta.get("val_loss"),
            "epochs": ckpt_meta.get("epochs"),
            "batch_size": ckpt_meta.get("batch_size"),
            "lr": ckpt_meta.get("lr"),
            "val_split": ckpt_meta.get("val_split"),
        })
    print(f"\n  Log appended : {log_path}")


if __name__ == "__main__":
    main()
