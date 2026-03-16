"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:  
    python scripts/train.py \
    --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper "state_cube[:5]"  \
    --action-keys action_ee_xyz action_gripper \
    --policy obstacle --chunk-size 16 --d-model 512 --depth 4
    
    python scripts/train.py \
        --zarr datasets/processed/single_cube/processed_ee_full.zarr \
        --state-keys state_ee_full state_gripper "state_cube[:5]"  \
        --action-keys action_ee_full action_gripper \
        --policy obstacle --chunk-size 16 --d-model 512 --depth 4
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
    episode_train_val_split,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 200 
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.3


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
        # TODO: Implement the training step for one batch here.
        # This mostly: Get states and action_chunks onto the correct device, compute the loss, and step the optimizer.
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
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
        # TODO: Implement the evaluation step for one batch here.
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Transformer d_model dimension (default: 256).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Transformer depth (number of layers) (default: 3).",
    )
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
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability after each hidden ReLU (default: 0.1). Set to 0 to disable.",
    )
    parser.add_argument(
        "--episode-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split by whole episodes (default). Use --no-episode-split for random timestep split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

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
    normalizer = Normalizer.from_data(states, actions)

    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    if args.episode_split:
        (tr_states, tr_actions, tr_ends), (va_states, va_actions, va_ends) = (
            episode_train_val_split(states, actions, ep_ends, val_ratio=VAL_SPLIT, seed=args.seed)
        )
        train_ds = SO100ChunkDataset(tr_states, tr_actions, tr_ends, chunk_size=args.chunk_size, normalizer=normalizer)
        val_ds   = SO100ChunkDataset(va_states, va_actions, va_ends, chunk_size=args.chunk_size, normalizer=normalizer)
    else:
        full_ds = SO100ChunkDataset(states, actions, ep_ends, chunk_size=args.chunk_size, normalizer=normalizer)
        n_val = max(1, int(len(full_ds) * VAL_SPLIT))
        train_ds, val_ds = random_split(
            full_ds, [len(full_ds) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples, chunk_size={args.chunk_size}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        # TODO: build with your desired specifications
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # TODO: implement an optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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

    for epoch in range(1, EPOCHS + 1):
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
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "dropout": args.dropout,
                    "val_loss": val_loss,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "val_split": VAL_SPLIT,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
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
    epochs_range = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_losses, label="train")
    ax.plot(epochs_range, val_losses, label="val")
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