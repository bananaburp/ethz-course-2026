"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:  

    python scripts/train.py \
        --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys state_ee_xyz state_gripper "state_cube[:5]" state_obstacle \
        --action-keys action_ee_xyz action_gripper \
        --policy obstacle --chunk-size 16 --d-model 384 --depth 4 --epochs 100 \
        --layer-norm --residual

    python scripts/train.py \
        --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
        --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
        --action-keys action_ee_xyz action_gripper \
        --policy multitask --chunk-size 16 --d-model 512 --depth 4 --epochs 200 \
        --rel-coords \
        --layer-norm --residual

    # BESO score-based diffusion policy (single cube)
    python scripts/train.py \
        --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys state_ee_xyz state_gripper "state_cube[:5]" state_obstacle \
        --action-keys action_ee_xyz action_gripper \
        --policy beso --chunk-size 16 --d-model 256 --depth 4 --epochs 200 \
        --sigma-data 0.5 --sigma-min 0.002 --sigma-max 80.0 \
        --n-timesteps 10 --cond-mask-prob 0.1 \
        --layer-norm

    # BESO score-based diffusion policy (multi cube, goal-conditioned)
    python scripts/train.py \
        --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
        --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
        --action-keys action_ee_xyz action_gripper \
        --policy beso --chunk-size 16 --d-model 384 --depth 4 --epochs 300 \
        --sigma-data 0.5 --sigma-min 0.002 --sigma-max 80.0 \
        --n-timesteps 20 --cond-mask-prob 0.1 \
        --rel-coords --layer-norm


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
    filter_episodes_by_goal,
    load_and_merge_zarrs,
    load_zarr,
    rel_coords_transform,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.15


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
        choices=["obstacle", "multitask", "beso"],
        default="obstacle",
        help="Policy type: 'obstacle', 'multitask', or 'beso' (score-based diffusion) (default: obstacle).",
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
        "--layer-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Insert LayerNorm after each hidden linear layer (default: off).",
    )
    parser.add_argument(
        "--residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add residual skip connections in hidden blocks (default: off).",
    )
    # ── BESO diffusion hyperparameters ────────────────────────────────
    parser.add_argument(
        "--sigma-data",
        type=float,
        default=0.5,
        help="EDM sigma_data: expected std of clean (normalised) actions (default: 0.5).",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.002,
        help="BESO minimum noise level (default: 0.002).",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=80.0,
        help="BESO maximum noise level (default: 80.0).",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=10,
        help="BESO denoising steps at inference (default: 10).",
    )
    parser.add_argument(
        "--cond-mask-prob",
        type=float,
        default=0.1,
        help="Classifier-free guidance dropout probability (default: 0.1).",
    )
    parser.add_argument(
        "--rel-coords",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Express all XYZ positions relative to the target cube (multicube only).",
    )
    parser.add_argument(
        "--episode-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split by whole episodes (default). Use --no-episode-split for random timestep split.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs (default: 400).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--filter-goal",
        type=str,
        default=None,
        choices=["red", "green", "blue"],
        help="Keep only episodes whose goal matches this colour (multicube per-color training).",
    )
    args = parser.parse_args()
    EPOCHS = args.epochs

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
    if args.filter_goal:
        states, actions, ep_ends = filter_episodes_by_goal(
            states, actions, ep_ends, args.state_keys, args.filter_goal
        )
    if args.rel_coords:
        states = rel_coords_transform(states, args.state_keys)
        print("  Relative-coordinate transform applied.")

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
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
        residual=args.residual,
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        n_timesteps=args.n_timesteps,
        cond_mask_prob=args.cond_mask_prob,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # TODO: implement an optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curves — {action_space} / {args.policy}")
    ax.grid(True, alpha=0.3)
    (train_line,) = ax.plot([], [], label="train")
    (val_line,) = ax.plot([], [], label="val")
    ax.legend()
    plt.show(block=False)

    color_tag = f"_{args.filter_goal}" if args.filter_goal else ""
    save_name = f"best_model_{action_space}_{args.policy}{color_tag}.pt"

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

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train {train_loss:.6f} | val {val_loss:.6f}"
            )
            xs = range(1, len(train_losses) + 1)
            train_line.set_data(xs, train_losses)
            val_line.set_data(xs, val_losses)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

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
                    "layer_norm": args.layer_norm,
                    "residual": args.residual,
                    "sigma_data": args.sigma_data,
                    "sigma_min": args.sigma_min,
                    "sigma_max": args.sigma_max,
                    "n_timesteps": args.n_timesteps,
                    "cond_mask_prob": args.cond_mask_prob,
                    "val_loss": val_loss,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "val_split": VAL_SPLIT,
                    "rel_coords": args.rel_coords,
                },
                save_path,
            )
            tag = " ✓ saved"

        if tag and epoch % 10 != 0:
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
    xs = range(1, EPOCHS + 1)
    train_line.set_data(xs, train_losses)
    val_line.set_data(xs, val_losses)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.ioff()
    plot_path = save_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot: {plot_path}")


if __name__ == "__main__":
    main()