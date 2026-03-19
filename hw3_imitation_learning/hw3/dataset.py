"""Dataset utilities for SO-100 teleop imitation learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        active_mask: np.ndarray | None = None,
        method: str = "zscore",
    ) -> "Normalizer":
        # Fit action statistics only on active steps when a mask is provided.
        # This prevents near-zero "stationary" steps from collapsing the std.
        action_rows = actions[active_mask] if active_mask is not None else actions
        if method == "minmax":
            # Normalize each dim to [-1, 1] using 1st/99th percentile bounds.
            # Equivalent to z-score with mean=(p99+p1)/2 and std=(p99-p1)/2.
            s_lo = np.percentile(states, 1, axis=0)
            s_hi = np.percentile(states, 99, axis=0)
            state_mean = (s_hi + s_lo) / 2.0
            state_std = cls._safe_std((s_hi - s_lo) / 2.0)
            a_lo = np.percentile(action_rows, 1, axis=0)
            a_hi = np.percentile(action_rows, 99, axis=0)
            action_mean = (a_hi + a_lo) / 2.0
            action_std = cls._safe_std((a_hi - a_lo) / 2.0)
        else:
            state_mean = states.mean(axis=0)
            state_std = cls._safe_std(states.std(axis=0))
            action_mean = action_rows.mean(axis=0)
            action_std = cls._safe_std(action_rows.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def _parse_key_spec(spec: str) -> tuple[str, slice]:
    """Parse a key spec like ``"state_cube[:3]"`` into (key, col_slice).

    Supports slicing notations: ``key``, ``key[:N]``, ``key[M:]``, ``key[M:N]``.
    Returns the array name and a column slice to apply on axis=1.
    """
    if "[" not in spec:
        return spec, slice(None)
    name, rest = spec.split("[", 1)
    rest = rest.rstrip("]")
    parts = rest.split(":")
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return name, slice(start, stop)
    raise ValueError(
        f"Invalid key spec: {spec!r}  (expected 'key', 'key[:N]', 'key[M:]', or 'key[M:N]')"
    )


def get_state_key_layout(
    zarr_path: Path, key_specs: list[str]
) -> list[tuple[str, int, int]]:
    """Return [(spec, col_start, col_end), ...] for each key spec against a zarr."""
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]
    layout: list[tuple[str, int, int]] = []
    offset = 0
    for spec in key_specs:
        name, col_slice = _parse_key_spec(spec)
        arr = np.asarray(data[name][:1], dtype=np.float32)
        sliced = arr[:, col_slice] if col_slice != slice(None) else arr
        width = sliced.shape[1]
        layout.append((spec, offset, offset + width))
        offset += width
    return layout


def load_zarr(
    zarr_path: Path,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load states, actions, and episode_ends from a processed .zarr.

    Args:
        zarr_path: Path to the processed .zarr store.
        state_keys: List of data array key specs to concatenate as the state.
            Each entry can include an optional column slice, e.g.
            ``["state_ee_xyz", "state_cube[:3]"]``.
            If ``None``, falls back to the ``state_key`` attribute in the zarr metadata.
        action_keys: List of data array key specs to concatenate as the action.
            Supports column slicing, e.g. ``["action_ee_xyz", "action_gripper"]``.
            If ``None``, falls back to the ``action_key`` attribute in the zarr metadata.

    Returns:
        states, actions, episode_ends
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]

    # ── states: concatenate one or more arrays ────────────────────────
    if state_keys is None:
        sk = root.attrs.get("state_key", "state")
        state_keys = [sk]

    state_parts: list[np.ndarray] = []
    for spec in state_keys:
        name, col_slice = _parse_key_spec(spec)
        arr = np.asarray(data[name][:], dtype=np.float32)
        state_parts.append(arr[:, col_slice] if col_slice != slice(None) else arr)
    states = (
        np.concatenate(state_parts, axis=1) if len(state_parts) > 1 else state_parts[0]
    )

    # ── actions: concatenate one or more arrays ───────────────────────
    if action_keys is None:
        ak = root.attrs.get("action_key", "action")
        action_keys = [ak]

    action_parts: list[np.ndarray] = []
    for spec in action_keys:
        act_name, act_slice = _parse_key_spec(spec)
        arr = np.asarray(data[act_name][:], dtype=np.float32)
        action_parts.append(arr[:, act_slice] if act_slice != slice(None) else arr)
    actions = (
        np.concatenate(action_parts, axis=1)
        if len(action_parts) > 1
        else action_parts[0]
    )

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    if debug:
        print(f"\n[load_zarr] {zarr_path.name}")
        print("  State key layout:")
        offset = 0
        for spec, part in zip(state_keys, state_parts):
            w = part.shape[1]
            print(f"    col {offset:3d}-{offset+w-1:3d} | {spec}")
            offset += w
        print(f"    => total state_dim: {states.shape[1]}")
        print("  Action key layout:")
        offset = 0
        for spec, part in zip(action_keys, action_parts):
            w = part.shape[1]
            print(f"    col {offset:3d}-{offset+w-1:3d} | {spec}")
            offset += w
        print(f"    => total action_dim: {actions.shape[1]}")
        print(f"  Timesteps: {states.shape[0]}, Episodes: {len(episode_ends)}")

    return states, actions, episode_ends


def load_and_merge_zarrs(
    zarr_paths: list[Path],
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate data from multiple processed .zarr stores.

    Each zarr store is loaded independently via :func:`load_zarr` and the
    results are concatenated.  Episode-end indices are shifted so they remain
    globally correct after concatenation.

    Returns the same ``(states, actions, episode_ends)`` tuple
    as :func:`load_zarr`.
    """
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_ep_ends: list[np.ndarray] = []
    offset = 0

    for i, zp in enumerate(zarr_paths):
        states, actions, ep_ends = load_zarr(
            zp, state_keys=state_keys, action_keys=action_keys, debug=(debug and i == 0),
        )
        all_states.append(states)
        all_actions.append(actions)
        all_ep_ends.append(ep_ends + offset)
        offset += states.shape[0]

    merged_states = np.concatenate(all_states, axis=0)
    merged_actions = np.concatenate(all_actions, axis=0)
    merged_ep_ends = np.concatenate(all_ep_ends, axis=0)

    return merged_states, merged_actions, merged_ep_ends


def audit_zarr_keys(
    zarr_path: Path,
    state_keys: list[str] | None,
    action_keys: list[str] | None,
) -> None:
    """Print available state/action arrays in a zarr and warn about unused ones."""
    root = zarr.open_group(str(zarr_path), mode="r")
    available = sorted(root["data"].keys())
    used = set()
    if state_keys:
        used.update(_parse_key_spec(k)[0] for k in state_keys)
    if action_keys:
        used.update(_parse_key_spec(k)[0] for k in action_keys)

    state_avail = [k for k in available if k.startswith("state_")]
    action_avail = [k for k in available if k.startswith("action_")]
    unused_state = [k for k in state_avail if k not in used]
    unused_action = [k for k in action_avail if k not in used]

    print(f"Zarr key audit ({zarr_path.name}):")
    print(f"  available state  arrays: {state_avail}")
    print(f"  available action arrays: {action_avail}")
    print(f"  used keys: {sorted(used)}")
    if unused_state:
        print(f"  WARNING: unused state  arrays: {unused_state}")
    if unused_action:
        print(f"  WARNING: unused action arrays: {unused_action}")


def episode_train_val_split(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Split data into train/val by whole episodes (no episode bleeds across splits).

    Episodes are shuffled then divided so the last ``val_ratio`` fraction of the
    shuffled list becomes validation.  Episode-end indices in each returned split
    are renumbered to be locally correct.

    Returns:
        (train_states, train_actions, train_ep_ends),
        (val_states,   val_actions,   val_ep_ends)
    """
    n_episodes = len(episode_ends)
    n_val = max(1, int(n_episodes * val_ratio))

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_episodes)
    train_idx = sorted(shuffled[n_val:].tolist())
    val_idx = sorted(shuffled[:n_val].tolist())

    starts = np.concatenate(([0], episode_ends[:-1]))

    def _extract(
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        new_ends: list[int] = []
        offset = 0
        for i in indices:
            s, e = int(starts[i]), int(episode_ends[i])
            state_parts.append(states[s:e])
            action_parts.append(actions[s:e])
            offset += e - s
            new_ends.append(offset)
        return (
            np.concatenate(state_parts, axis=0),
            np.concatenate(action_parts, axis=0),
            np.asarray(new_ends, dtype=np.int64),
        )

    return _extract(train_idx), _extract(val_idx)


def build_valid_indices(
    episode_ends: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Return flat indices where a full action chunk of length ``chunk_size`` fits.

    For each episode [start, end) we keep indices start … (end - chunk_size).
    """
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        for t in range(start, last_start + 1):
            indices.append(t)
    return np.asarray(indices, dtype=np.int64)


class SO100ChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs with a sliding window of size H.

    Each sample consists of:
        state:        (state_dim,)             - state at timestep t
        action_chunk: (chunk_size, action_dim) - actions [t, t+1, …, t+H-1]
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
        goal_permutation: bool = False,
        cube_col_start: int | None = None,
        goal_col_start: int | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)
        self.goal_permutation = goal_permutation
        self.goal_col_start = goal_col_start if goal_col_start is not None else states.shape[1] - 3
        self.cube_col_start = cube_col_start if cube_col_start is not None else states.shape[1] - 12

        print(f"\n[SO100ChunkDataset] state_dim={states.shape[1]}, action_dim={actions.shape[1]}, "
              f"chunk_size={chunk_size}, n_valid={len(self.indices)}")
        print(f"  cube_col_start={self.cube_col_start}  (cols {self.cube_col_start}-{self.cube_col_start+8}: 3 cube xyz blocks)")
        print(f"  goal_col_start={self.goal_col_start}  (cols {self.goal_col_start}-{self.goal_col_start+2}: goal one-hot)")
        if self.goal_permutation and len(self.indices) > 0:
            t0 = int(self.indices[0])
            s0 = states[t0]
            c, g = self.cube_col_start, self.goal_col_start
            print(f"  [perm debug] sample t={t0} raw (before norm):")
            print(f"    cube_red  cols {c  }-{c+2 }: {s0[c  :c+3 ]}")
            print(f"    cube_grn  cols {c+3}-{c+5 }: {s0[c+3:c+6 ]}")
            print(f"    cube_blu  cols {c+6}-{c+8 }: {s0[c+6:c+9 ]}")
            print(f"    goal      cols {g  }-{g+2 }: {s0[g  :g+3 ]}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        state = self.states[t].copy()
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        if self.goal_permutation:
            perm = np.random.permutation(3)
            # permute the three cube position blocks (each 3 dims)
            c = self.cube_col_start
            cube_blocks = state[c : c + 9].reshape(3, 3)
            state[c : c + 9] = cube_blocks[perm].reshape(9)
            # permute the goal one-hot to match
            g = self.goal_col_start
            state[g : g + 3] = state[g : g + 3][perm]

        state_t = torch.from_numpy(state).float()
        action_t = torch.from_numpy(action_chunk).float()

        return state_t, action_t
