"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class _MLPBlock(nn.Module):
    """Linear → optional LayerNorm → ReLU, with optional residual skip."""

    def __init__(self, d_model: int, layer_norm: bool, residual: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model) if layer_norm else None
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = F.relu(out)
        if self.residual:
            out = out + x
        return out


class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.dropout_p = dropout
        # Stored as buffers so they survive checkpoint round-trips.
        self.register_buffer("_layer_norm", torch.tensor(layer_norm))
        self.register_buffer("_residual", torch.tensor(residual))
        self._build(layer_norm, residual)

    def _build(self, layer_norm: bool, residual: bool) -> None:
        output_dim = self.chunk_size * self.action_dim
        input_layers: list[nn.Module] = [nn.Linear(self.state_dim, self.d_model)]
        if layer_norm:
            input_layers.append(nn.LayerNorm(self.d_model))
        input_layers.append(nn.ReLU())
        self.input_proj = nn.Sequential(*input_layers)
        self.hidden_blocks = nn.ModuleList([
            _MLPBlock(self.d_model, layer_norm=layer_norm, residual=residual)
            for _ in range(self.depth - 1)
        ])
        self.output_proj = nn.Linear(self.d_model, output_dim)

    @staticmethod
    def _migrate_old_net_state_dict(state_dict: dict) -> dict:
        """Remap old flat net.N keys to input_proj/hidden_blocks/output_proj."""
        import re
        indices = sorted({
            int(m.group(1))
            for k in state_dict
            for m in [re.match(r"^net\.(\d+)\.", k)]
            if m
        })
        if not indices:
            return state_dict

        new_sd = {k: v for k, v in state_dict.items() if not k.startswith("net.")}

        # Group consecutive pairs: consecutive index → (Linear, LayerNorm)
        groups: list[tuple[int, int | None]] = []
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and indices[i + 1] == indices[i] + 1:
                groups.append((indices[i], indices[i + 1]))
                i += 2
            else:
                groups.append((indices[i], None))
                i += 1

        input_group = groups[0]
        hidden_groups = groups[1:-1]
        output_lin_idx = groups[-1][0]
        has_layer_norm = input_group[1] is not None

        for sfx in ("weight", "bias"):
            if (k := f"net.{input_group[0]}.{sfx}") in state_dict:
                new_sd[f"input_proj.0.{sfx}"] = state_dict[k]
        if input_group[1] is not None:
            for sfx in ("weight", "bias"):
                if (k := f"net.{input_group[1]}.{sfx}") in state_dict:
                    new_sd[f"input_proj.1.{sfx}"] = state_dict[k]

        for blk, (lin_idx, norm_idx) in enumerate(hidden_groups):
            for sfx in ("weight", "bias"):
                if (k := f"net.{lin_idx}.{sfx}") in state_dict:
                    new_sd[f"hidden_blocks.{blk}.linear.{sfx}"] = state_dict[k]
            if norm_idx is not None:
                for sfx in ("weight", "bias"):
                    if (k := f"net.{norm_idx}.{sfx}") in state_dict:
                        new_sd[f"hidden_blocks.{blk}.norm.{sfx}"] = state_dict[k]

        for sfx in ("weight", "bias"):
            if (k := f"net.{output_lin_idx}.{sfx}") in state_dict:
                new_sd[f"output_proj.{sfx}"] = state_dict[k]

        new_sd.setdefault("_layer_norm", torch.tensor(has_layer_norm))
        new_sd.setdefault("_residual", torch.tensor(False))
        return new_sd

    def load_state_dict(self, state_dict: dict, strict: bool = True, **kwargs):
        state_dict = self._migrate_old_net_state_dict(state_dict)
        # Infer architecture from the incoming state dict so the model
        # self-configures even when the eval harness omits layer_norm/residual.
        needs_ln = any(
            k.startswith("hidden_blocks.") and k.endswith(".norm.weight")
            for k in state_dict
        )
        needs_res = bool(state_dict.get("_residual", torch.tensor(False)).item())
        cur_ln = bool(self._layer_norm.item())
        cur_res = bool(self._residual.item())
        if needs_ln != cur_ln or needs_res != cur_res:
            self._layer_norm.fill_(needs_ln)
            self._residual.fill_(needs_res)
            self._build(needs_ln, needs_res)
        state_dict.setdefault("_layer_norm", torch.tensor(bool(needs_ln)))
        state_dict.setdefault("_residual", torch.tensor(bool(needs_res)))
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = self.input_proj(state)
        if self.training and self.dropout_p > 0.0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        for block in self.hidden_blocks:
            x = block(x)
            if self.training and self.dropout_p > 0.0:
                x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.output_proj(x)
        return x.view(x.size(0), self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)
        return F.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state)


class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.

    The goal conditioning (state_goal one-hot, goal_pos, per-cube positions)
    is concatenated into the state vector at data-loading time, so this MLP
    operates on the full goal-conditioned state directly.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.dropout_p = dropout
        self.register_buffer("_layer_norm", torch.tensor(layer_norm))
        self.register_buffer("_residual", torch.tensor(residual))
        self._build(layer_norm, residual)

    def _build(self, layer_norm: bool, residual: bool) -> None:
        output_dim = self.chunk_size * self.action_dim
        input_layers: list[nn.Module] = [nn.Linear(self.state_dim, self.d_model)]
        if layer_norm:
            input_layers.append(nn.LayerNorm(self.d_model))
        input_layers.append(nn.ReLU())
        self.input_proj = nn.Sequential(*input_layers)
        self.hidden_blocks = nn.ModuleList([
            _MLPBlock(self.d_model, layer_norm=layer_norm, residual=residual)
            for _ in range(self.depth - 1)
        ])
        self.output_proj = nn.Linear(self.d_model, output_dim)

    def load_state_dict(self, state_dict: dict, strict: bool = True, **kwargs):
        needs_ln = any(
            k.startswith("hidden_blocks.") and k.endswith(".norm.weight")
            for k in state_dict
        )
        needs_res = bool(state_dict.get("_residual", torch.tensor(False)).item())
        cur_ln = bool(self._layer_norm.item())
        cur_res = bool(self._residual.item())
        if needs_ln != cur_ln or needs_res != cur_res:
            self._layer_norm.fill_(needs_ln)
            self._residual.fill_(needs_res)
            self._build(needs_ln, needs_res)
        state_dict.setdefault("_layer_norm", torch.tensor(bool(needs_ln)))
        state_dict.setdefault("_residual", torch.tensor(bool(needs_res)))
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = self.input_proj(state)
        if self.training and self.dropout_p > 0.0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        for block in self.hidden_blocks:
            x = block(x)
            if self.training and self.dropout_p > 0.0:
                x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.output_proj(x)
        return x.view(x.size(0), self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


class CVAEPolicy(BasePolicy):
    """CVAE policy with learned conditional prior p_θ(z|s).

    Training: encoder q_φ(z|s,a) → z, KL(q_φ || p_θ(z|s)) + recon loss.
    Inference: sample z ~ p_θ(z|s), decode (s, z) → action_chunk.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
        layer_norm: bool = False,
        residual: bool = False,
        latent_dim: int = 64,
        beta: float = 1e-4,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.dropout_p = dropout
        self.latent_dim = latent_dim
        self.beta = beta
        self.register_buffer("_layer_norm", torch.tensor(layer_norm))
        self.register_buffer("_residual", torch.tensor(residual))
        self._build(layer_norm, residual)

    def _build(self, layer_norm: bool, residual: bool) -> None:
        flat_action_dim = self.chunk_size * self.action_dim

        # Encoder: (state || action_chunk_flat) → (mu, log_var)
        enc_layers: list[nn.Module] = [nn.Linear(self.state_dim + flat_action_dim, self.d_model)]
        if layer_norm:
            enc_layers.append(nn.LayerNorm(self.d_model))
        enc_layers.append(nn.ReLU())
        self.enc_input = nn.Sequential(*enc_layers)
        self.enc_hidden = nn.ModuleList([
            _MLPBlock(self.d_model, layer_norm=layer_norm, residual=residual)
            for _ in range(self.depth - 1)
        ])
        self.enc_out = nn.Linear(self.d_model, 2 * self.latent_dim)  # mu + log_var

        # Decoder: (state || z) → action_chunk_flat
        dec_layers: list[nn.Module] = [nn.Linear(self.state_dim + self.latent_dim, self.d_model)]
        if layer_norm:
            dec_layers.append(nn.LayerNorm(self.d_model))
        dec_layers.append(nn.ReLU())
        self.dec_input = nn.Sequential(*dec_layers)
        self.dec_hidden = nn.ModuleList([
            _MLPBlock(self.d_model, layer_norm=layer_norm, residual=residual)
            for _ in range(self.depth - 1)
        ])
        self.dec_out = nn.Linear(self.d_model, flat_action_dim)

        # Prior network: state → (mu_prior, log_var_prior)
        prior_layers: list[nn.Module] = [nn.Linear(self.state_dim, self.d_model)]
        if layer_norm:
            prior_layers.append(nn.LayerNorm(self.d_model))
        prior_layers.append(nn.ReLU())
        self.prior_net = nn.Sequential(*prior_layers)
        self.prior_hidden = nn.ModuleList([
            _MLPBlock(self.d_model, layer_norm=layer_norm, residual=residual)
            for _ in range(max(1, self.depth - 2))
        ])
        self.prior_out = nn.Linear(self.d_model, 2 * self.latent_dim)

    def _encode(self, state: torch.Tensor, action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = action_chunk.view(action_chunk.size(0), -1)
        x = self.enc_input(torch.cat([state, flat], dim=-1))
        if self.training and self.dropout_p > 0.0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        for block in self.enc_hidden:
            x = block(x)
            if self.training and self.dropout_p > 0.0:
                x = F.dropout(x, p=self.dropout_p, training=True)
        mu, log_var = self.enc_out(x).chunk(2, dim=-1)
        log_var = log_var.clamp(-10, 4)
        return mu, log_var

    def _prior(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.prior_net(state)
        for block in self.prior_hidden:
            x = block(x)
        mu_p, log_var_p = self.prior_out(x).chunk(2, dim=-1)
        log_var_p = log_var_p.clamp(-10, 4)
        return mu_p, log_var_p

    def _decode(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_input(torch.cat([state, z], dim=-1))
        if self.training and self.dropout_p > 0.0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        for block in self.dec_hidden:
            x = block(x)
            if self.training and self.dropout_p > 0.0:
                x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.dec_out(x)
        return x.view(x.size(0), self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        mu, log_var = self._encode(state, action_chunk)
        # Reparameterization trick
        std = (0.5 * log_var).exp()
        z = mu + std * torch.randn_like(std)
        pred = self._decode(state, z)
        recon = F.mse_loss(pred, action_chunk)
        mu_p, log_var_p = self._prior(state)
        var_p = log_var_p.exp()
        kl = 0.5 * (
            log_var_p - log_var
            + (log_var.exp() + (mu - mu_p).pow(2)) / var_p
            - 1
        ).mean(dim=-1).mean()
        return recon + self.beta * kl

    def sample_actions(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu_p, log_var_p = self._prior(state)
        if deterministic:
            z = mu_p
        else:
            std_p = (0.5 * log_var_p).exp()
            z = mu_p + std_p * torch.randn_like(std_p)
        return self._decode(state, z)

    def load_state_dict(self, state_dict: dict, strict: bool = True, **kwargs):
        needs_ln = any(
            k.endswith(".norm.weight")
            for k in state_dict
            if k.startswith("enc_hidden.") or k.startswith("dec_hidden.")
        )
        needs_res = bool(state_dict.get("_residual", torch.tensor(False)).item())
        cur_ln = bool(self._layer_norm.item())
        cur_res = bool(self._residual.item())
        if needs_ln != cur_ln or needs_res != cur_res:
            self._layer_norm.fill_(needs_ln)
            self._residual.fill_(needs_res)
            self._build(needs_ln, needs_res)
        state_dict.setdefault("_layer_norm", torch.tensor(bool(needs_ln)))
        state_dict.setdefault("_residual", torch.tensor(bool(needs_res)))
        # Old checkpoints lack prior_net weights — fall back to non-strict load.
        has_prior = any(k.startswith("prior_") for k in state_dict)
        if not has_prior:
            strict = False
        return super().load_state_dict(state_dict, strict=strict, **kwargs)


PolicyType: TypeAlias = Literal["obstacle", "multitask", "cvae"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 128,
    depth: int = 2,
    dropout: float = 0.0,
    layer_norm: bool = False,
    residual: bool = False,
    latent_dim: int = 64,
    beta: float = 1.0,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            layer_norm=layer_norm,
            residual=residual,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            layer_norm=layer_norm,
            residual=residual,
        )
    if policy_type == "cvae":
        return CVAEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            layer_norm=layer_norm,
            residual=residual,
            latent_dim=latent_dim,
            beta=beta,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
