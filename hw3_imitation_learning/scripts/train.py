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

MULTI-CUBE example (goal_pos MUST come before the cube blocks so --goal-permutation works):
XYZ action space:
    python scripts/train.py \
        --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
        --state-keys state_ee_xyz state_gripper goal_pos \
            "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" \
            state_goal \
        --action-keys action_ee_xyz action_gripper \
        --policy multitask --chunk-size 16 --d-model 512 --depth 4 \
        --goal-permutation --episode-split

FULL action space:
    python scripts/train.py \
        --zarr datasets/processed/multi_cube/processed_ee_full.zarr \
        --state-keys state_ee_full state_gripper goal_pos \
            "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" \
            state_goal \
        --action-keys action_ee_full action_gripper \
        --policy multitask --chunk-size 16 --d-model 512 --depth 4 \
        --goal-permutation --episode-split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    audit_zarr_keys,
    episode_train_val_split,
    get_state_key_layout,
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


def visualize_inputs(
    train_states: np.ndarray,
    val_states: np.ndarray,
    save_dir: Path,
    state_keys: list[str] | None = None,
) -> None:
    """Plot state dimension histograms and a PCA train/val scatter, saved once before training."""
    # ── per-dim histograms ────────────────────────────────────────────
    state_dim = train_states.shape[1]
    ncols = min(8, state_dim)
    nrows = (state_dim + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 1.8), squeeze=False)
    for d in range(state_dim):
        ax = axes[d // ncols][d % ncols]
        ax.hist(train_states[:, d], bins=40, alpha=0.6, color="steelblue", label="train", density=True)
        ax.hist(val_states[:, d], bins=40, alpha=0.6, color="tomato", label="val", density=True)
        ax.set_title(f"dim {d}", fontsize=7)
        ax.tick_params(labelsize=5)
    # hide empty subplots
    for d in range(state_dim, nrows * ncols):
        axes[d // ncols][d % ncols].set_visible(False)
    axes[0][-1].legend(fontsize=6)
    fig.suptitle("Normalised state distributions — train (blue) vs val (red)", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "input_histograms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── PCA scatter ───────────────────────────────────────────────────
    if state_dim >= 2:
        all_states = np.concatenate([train_states, val_states], axis=0)
        mean = all_states.mean(0)
        centered = all_states - mean
        # manual 2-component PCA via SVD (no sklearn needed)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T  # (N, 2)
        n_tr = len(train_states)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.scatter(proj[:n_tr, 0], proj[:n_tr, 1], s=4, alpha=0.3, color="steelblue", label=f"train ({n_tr})")
        ax2.scatter(proj[n_tr:, 0], proj[n_tr:, 1], s=4, alpha=0.5, color="tomato", label=f"val ({len(val_states)})")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("PCA of normalised state — train vs val")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(save_dir / "input_pca.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)


@torch.no_grad()
def evaluate_per_dim(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return per-action-dimension MSE averaged over the validation set."""
    model.eval()
    total_sq_err = None
    n_batches = 0
    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        preds = model.sample_actions(states)  # (B, chunk_size, action_dim)
        sq_err = ((preds - action_chunks) ** 2).mean(dim=(0, 1))  # (action_dim,)
        total_sq_err = sq_err if total_sq_err is None else total_sq_err + sq_err
        n_batches += 1
    return (total_sq_err / max(n_batches, 1)).cpu().numpy()


@torch.no_grad()
def visualize_predictions(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    epoch: int,
    n_samples: int = 4,
) -> None:
    """Plot predicted vs. ground-truth action chunks for a few val samples."""
    model.eval()
    states, action_chunks = next(iter(loader))
    states = states[:n_samples].to(device)
    action_chunks = action_chunks[:n_samples]

    preds = model.sample_actions(states).cpu().numpy()  # (n_samples, chunk_size, action_dim)
    gt = action_chunks.numpy()
    action_dim = gt.shape[-1]

    fig, axes = plt.subplots(
        n_samples, action_dim, figsize=(max(action_dim * 2.5, 6), n_samples * 2), squeeze=False
    )
    for i in range(n_samples):
        for d in range(action_dim):
            ax = axes[i][d]
            ax.plot(gt[i, :, d], label="GT", color="steelblue")
            ax.plot(preds[i, :, d], label="Pred", color="tomato", linestyle="--")
            if i == 0:
                ax.set_title(f"dim {d}")
            if d == 0:
                ax.set_ylabel(f"sample {i}")
            ax.tick_params(labelsize=6)
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Epoch {epoch}: predicted vs GT action chunks", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path / f"pred_epoch{epoch:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _find_perm_col_starts(
    state_keys: list[str], zarr_path: Path
) -> tuple[int | None, int | None]:
    """Return (cube_col_start, goal_col_start) by walking state_keys column widths.

    Reads one row from each key in the zarr to determine the column width after
    slicing, then accumulates offsets to find where original_pos_cube_red and
    state_goal begin in the concatenated state vector.
    """
    root = zarr_lib.open_group(str(zarr_path), mode="r")
    data = root["data"]
    from hw3.dataset import _parse_key_spec

    offset = 0
    cube_col_start: int | None = None
    goal_col_start: int | None = None
    for spec in state_keys:
        name, col_slice = _parse_key_spec(spec)
        arr = np.asarray(data[name][:1], dtype=np.float32)
        width = (arr[:, col_slice] if col_slice != slice(None) else arr).shape[1]
        if name == "original_pos_cube_red" and cube_col_start is None:
            cube_col_start = offset
        if name == "state_goal" and goal_col_start is None:
            goal_col_start = offset
        offset += width
    return cube_col_start, goal_col_start


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
        "--goal-permutation",
        action="store_true",
        default=False,
        help="Augment training data by randomly permuting goal and cube positions (multitask only).",
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
    parser.add_argument(
        "--vis-every",
        type=int,
        default=20,
        help="Save prediction plots every N epochs (0 to disable).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    # Debug: audit zarr keys before loading
    audit_zarr_keys(zarr_paths[0], args.state_keys, args.action_keys)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
            debug=True,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
            debug=True,
        )
    # Build active-step mask: steps where the end-effector actually moves.
    # Fit ee action std on active steps only to avoid the bimodal zero/active distribution
    # from collapsing the std (84% near-zero, 16% near the ±10mm cap).
    # Gripper and joint stats are computed from all steps (already well-scaled).
    _ee_dims = {"action_ee_xyz": 3, "action_ee_full": 6, "action_gripper": 1, "action_joints": 5}
    _ee_col_start, _ee_col_end = 0, 0
    _col = 0
    for _spec in (args.action_keys or []):
        _name = _spec.split("[")[0]
        _d = _ee_dims.get(_name, 0)
        if _name in ("action_ee_xyz", "action_ee_full"):
            _ee_col_start, _ee_col_end = _col, _col + _d
            break
        _col += _d

    EE_XYZ_THRESH = 0.001   # 1 mm — cleanly splits zero cluster from active cluster
    if _ee_col_end > _ee_col_start:
        ee_mask = np.linalg.norm(actions[:, _ee_col_start:_ee_col_start + 3], axis=1) > EE_XYZ_THRESH
        n_active = int(ee_mask.sum())
        print(f"  Active steps for ee normalizer: {n_active}/{len(actions)} "
              f"({100*n_active/len(actions):.1f}%)  [cols {_ee_col_start}:{_ee_col_end}]")
        action_mean = actions.mean(axis=0)
        action_std  = actions.std(axis=0)
        active_ee = actions[ee_mask, _ee_col_start:_ee_col_end]
        action_mean[_ee_col_start:_ee_col_end] = active_ee.mean(axis=0)
        action_std[_ee_col_start:_ee_col_end]  = active_ee.std(axis=0)
        print(f"  EE active-step mean: {action_mean[_ee_col_start:_ee_col_end]}")
        print(f"  EE active-step std:  {action_std[_ee_col_start:_ee_col_end]}")
    else:
        print("  No ee action key found — using standard normalization for all action dims.")
        action_mean = actions.mean(axis=0)
        action_std  = actions.std(axis=0)
    action_std  = np.maximum(action_std, 1e-6)
    state_mean  = states.mean(axis=0)
    state_std   = np.maximum(states.std(axis=0), 1e-6)
    normalizer  = Normalizer(state_mean, state_std, action_mean, action_std)

    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # Debug: normalizer stats per state key block
    if args.state_keys:
        layout = get_state_key_layout(zarr_paths[0], args.state_keys)
        print("\nNormalizer stats per state key:")
        for spec, c0, c1 in layout:
            m = normalizer.state_mean[c0:c1]
            s = normalizer.state_std[c0:c1]
            print(f"  {spec:<42s} cols {c0:3d}-{c1-1:3d} | "
                  f"mean={np.array2string(m, precision=3, suppress_small=True, max_line_width=120)} "
                  f"std={np.array2string(s, precision=3, suppress_small=True, max_line_width=120)}")
    if args.action_keys:
        layout_a = get_state_key_layout(zarr_paths[0], args.action_keys)
        print("Normalizer stats per action key:")
        for spec, c0, c1 in layout_a:
            m = normalizer.action_mean[c0:c1]
            s = normalizer.action_std[c0:c1]
            print(f"  {spec:<42s} cols {c0:3d}-{c1-1:3d} | "
                  f"mean={np.array2string(m, precision=3, suppress_small=True, max_line_width=120)} "
                  f"std={np.array2string(s, precision=3, suppress_small=True, max_line_width=120)}")
    print()

    # ── compute permutation offsets (robust regardless of state_keys ordering) ──
    cube_col_start: int | None = None
    goal_col_start: int | None = None
    if args.goal_permutation and args.state_keys:
        cube_col_start, goal_col_start = _find_perm_col_starts(args.state_keys, zarr_paths[0])
        print(f"  Permutation indices: cube_col_start={cube_col_start}, goal_col_start={goal_col_start}")

    # ── train / val split ─────────────────────────────────────────────
    if args.episode_split:
        (tr_states, tr_actions, tr_ends), (va_states, va_actions, va_ends) = (
            episode_train_val_split(states, actions, ep_ends, val_ratio=VAL_SPLIT, seed=args.seed)
        )
        train_ds = SO100ChunkDataset(tr_states, tr_actions, tr_ends, chunk_size=args.chunk_size, normalizer=normalizer, goal_permutation=args.goal_permutation, cube_col_start=cube_col_start, goal_col_start=goal_col_start)
        val_ds   = SO100ChunkDataset(va_states, va_actions, va_ends, chunk_size=args.chunk_size, normalizer=normalizer)
    else:
        if args.goal_permutation:
            print("Warning: --goal-permutation requires --episode-split. Disabling augmentation.")
            # args.goal_permutation = False
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

    # Debug: inspect first batch shapes and value ranges
    _s, _a = next(iter(train_loader))
    print(f"[DataLoader sample] state shape={tuple(_s.shape)}, action_chunk shape={tuple(_a.shape)}")
    print(f"  state   min={_s.min().item():.3f}  max={_s.max().item():.3f}  mean={_s.mean().item():.3f}")
    print(f"  actions min={_a.min().item():.3f}  max={_a.max().item():.3f}  mean={_a.mean().item():.3f}")
    del _s, _a

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
    patience = EPOCHS // 8
    no_improve_count = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    per_dim_log: list[np.ndarray] = []

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
    vis_dir = save_path.parent / f"vis_{save_path.stem}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    # pull normalised states from the train/val datasets for input inspection
    if args.episode_split:
        norm_tr = normalizer.normalize_state(tr_states)
        norm_va = normalizer.normalize_state(va_states)
    else:
        norm_tr = normalizer.normalize_state(states)
        norm_va = norm_tr  # same pool, split is random per-step
    visualize_inputs(norm_tr, norm_va, vis_dir, state_keys=args.state_keys)
    print(f"Input vis:  {vis_dir}/input_histograms.png  input_pca.png")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        per_dim = evaluate_per_dim(model, val_loader, device)
        per_dim_log.append(per_dim)

        if args.vis_every > 0 and epoch % args.vis_every == 0:
            visualize_predictions(model, val_loader, device, vis_dir, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tag = ""
        if best_val - val_loss > 0.001:
            best_val = val_loss
            no_improve_count = 0
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
        else:
            no_improve_count += 1

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement > 0.001 for {patience} epochs)")
            break

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
    epochs_range = range(1, len(train_losses) + 1)
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

    # ── per-dimension error plot ───────────────────────────────────────
    if per_dim_log:
        per_dim_arr = np.stack(per_dim_log)  # (epochs, action_dim)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for d in range(per_dim_arr.shape[1]):
            ax2.plot(epochs_range, per_dim_arr[:, d], label=f"dim {d}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE")
        ax2.set_title(f"Per-dim val MSE — {action_space} / {args.policy}")
        ax2.legend(ncol=max(1, per_dim_arr.shape[1] // 4), fontsize=7)
        ax2.grid(True, alpha=0.3)
        perdim_path = save_path.with_suffix("").parent / f"{save_path.stem}_perdim.png"
        fig2.savefig(perdim_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Per-dim plot: {perdim_path}")


if __name__ == "__main__":
    main()