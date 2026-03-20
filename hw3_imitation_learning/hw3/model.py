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
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        output_dim = chunk_size * action_dim

        self.dropout_p = dropout
        layers: list[nn.Module] = [nn.Linear(state_dim, d_model), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(d_model, d_model), nn.ReLU()]
        layers.append(nn.Linear(d_model, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = state
        for layer in self.net:
            x = layer(x)
            if self.training and self.dropout_p > 0.0 and isinstance(layer, nn.ReLU):
                x = F.dropout(x, p=self.dropout_p, training=True)
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


# # TODO: Students implement MultiTaskPolicy here.
# class MultiTaskPolicy(BasePolicy):
#     """Goal-conditioned policy for the multicube scene."""

#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         chunk_size: int,
#         d_model: int = 128,
#         depth: int = 2,
#         dropout: float = 0.0,
#     ) -> None:
#         super().__init__(state_dim, action_dim, chunk_size)
#         self.dropout_p = dropout
#         output_dim = chunk_size * action_dim
#         layers: list[nn.Module] = [nn.Linear(state_dim, d_model), nn.ReLU()]
#         for _ in range(depth - 1):
#             layers += [nn.Linear(d_model, d_model), nn.ReLU()]
#         layers.append(nn.Linear(d_model, output_dim))
#         self.net = nn.Sequential(*layers)

#     def compute_loss(
#         self, state: torch.Tensor, action_chunk: torch.Tensor
#     ) -> torch.Tensor:
#         pred = self.forward(state)
#         return F.mse_loss(pred, action_chunk)

#     def sample_actions(
#         self,
#         state: torch.Tensor,
#     ) -> torch.Tensor:
#         return self.forward(state)

#     def forward(
#         self,
#         state: torch.Tensor,
#     ) -> torch.Tensor:
#         """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
#         x = state
#         for layer in self.net:
#             x = layer(x)
#             if self.training and self.dropout_p > 0.0 and isinstance(layer, nn.ReLU):
#                 x = F.dropout(x, p=self.dropout_p, training=True)
#         return x.view(x.size(0), self.chunk_size, self.action_dim)

class _ResidualBlock(nn.Module):
    """Linear -> LayerNorm -> ReLU -> Dropout with a skip connection."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(torch.relu(self.norm(self.linear(x))))

class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth

        self.input = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(
            *[_ResidualBlock(d_model, dropout) for _ in range(depth)]
        )
        self.output = nn.Linear(d_model, chunk_size * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = self.input(state)
        x = self.blocks(x)
        x = self.output(x)
        return x.reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)

PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 128,
    depth: int = 2,
    dropout: float = 0.0,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
