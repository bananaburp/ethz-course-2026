"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Prerequisites:
    Raw teleop recordings must first be processed into action-labelled zarrs:
        python scripts/compute_actions.py --action-space ee
    This writes to datasets/processed/single_cube/processed_ee_xyz.zarr

Usage (obstacle policy, EE action space — recommended for ex1/ex2):
    python scripts/train_v0.py \
        --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys state_ee_xyz state_gripper "state_cube[:5]"  \
        --action-keys action_ee_xyz action_gripper \
        --policy obstacle --chunk-size 8

Optional flags:
    --chunk-size 8      action chunk horizon H (default: 16)
    --d-model 128        MLP hidden dimension (default: 128)
    --depth 2            number of hidden layers (default: 2)
    --epochs 200         training epochs (default: 200)
    --batch-size 64      batch size (default: 64)
    --lr 1e-3            learning rate (default: 1e-3)
    --seed 42            random seed (default: 42)

Checkpoints are saved to:
    checkpoints/single_cube/best_model_<action_space>_<policy>.pt
    checkpoints/multi_cube/best_model_<action_space>_<policy>.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    audit_zarr_keys,
    episode_train_val_split,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy
from torch.utils.data import DataLoader

EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.1


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--extra-zarr",
        nargs="*",
        type=Path,
        default=None,
        dest="extra_zarr",
        help="Additional zarr paths to merge.",
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--d-model", type=int, default=128, help="MLP hidden dim.")
    parser.add_argument("--depth", type=int, default=2, help="Number of MLP hidden layers.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    # ── improvement 1: audit available zarr keys ──────────────────────
    audit_zarr_keys(args.zarr, args.state_keys, args.action_keys)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )

    print(f"Dataset: {states.shape[0]} timesteps across {len(ep_ends)} episodes")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── improvement 2: episode-aware train / val split ────────────────
    (train_states, train_actions, train_ep_ends), (
        val_states,
        val_actions,
        val_ep_ends,
    ) = episode_train_val_split(states, actions, ep_ends, val_ratio=VAL_SPLIT, seed=args.seed)

    n_train_ep = len(train_ep_ends)
    n_val_ep = len(val_ep_ends)
    print(f"Split: {n_train_ep} train episodes / {n_val_ep} val episodes")

    # fit normalizer on train data only — no val stat leakage
    normalizer = Normalizer.from_data(train_states, train_actions)

    train_ds = SO100ChunkDataset(
        train_states, train_actions, train_ep_ends,
        chunk_size=args.chunk_size, normalizer=normalizer,
    )
    val_ds = SO100ChunkDataset(
        val_states, val_actions, val_ep_ends,
        chunk_size=args.chunk_size, normalizer=normalizer,
    )
    print(f"Samples: {len(train_ds)} train / {len(val_ds)} val  (chunk_size={args.chunk_size})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                    "d_model": args.d_model,
                    "depth": args.depth,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")

    # ── save loss CSV ──────────────────────────────────────────────────
    csv_path = save_path.with_suffix(".csv")
    with csv_path.open("w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
            f.write(f"{i},{tl:.8f},{vl:.8f}\n")
    print(f"Loss CSV:  {csv_path}")

    # ── plot loss curves ───────────────────────────────────────────────
    epochs = range(1, args.epochs + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curves — {action_space} / {args.policy}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = save_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot: {plot_path}")


if __name__ == "__main__":
    main()
