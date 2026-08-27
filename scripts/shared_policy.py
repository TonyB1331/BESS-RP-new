"""Custom policy network with shared per-slot encoder for XBID PPO.

Architecture
------------
Observation:  flat vector of shape (n_slots * N_FEATURES,) = (96 * 23 = 2208,)
              reshaped internally to (batch, n_slots, N_FEATURES).

Shared encoder (applied identically to every slot):
    Linear(23 → 64) → Tanh → Linear(64 → 64) → Tanh

Aggregation across slots:
    Mean-pooling  + Max-pooling  → concat → (batch, 128)

Policy / Value heads:
    policy: Linear(128 → 64) → Tanh → Linear(64 → action_dim)
    value:  Linear(128 → 64) → Tanh → Linear(64 → 1)

Motivation
----------
Each of the 96 slots has the same semantics (23 features describing the
same market state); sharing encoder weights gives:
  * ~28× fewer parameters in the trunk vs a flat MLP over 2208 inputs
  * much better sample efficiency — one step updates the encoder using
    gradients from all 96 slots
  * permutation-equivariance baked into the slot dimension, aligning
    with the physics of the problem (what matters is the *distribution*
    of slot states, not their arbitrary order).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Matches xbid_trader.market.rl_observation.N_FEATURES
N_FEATURES = 23


class SharedSlotEncoder(BaseFeaturesExtractor):
    """Features extractor that applies a shared MLP to every slot then
    pools across slots."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_features: int = N_FEATURES,
        hidden_dim: int = 64,
        encoded_dim: int = 64,
    ) -> None:
        # Output dim is 2*encoded_dim because we concatenate mean+max pooling
        super().__init__(observation_space, features_dim=2 * encoded_dim)

        flat_dim = int(observation_space.shape[0])
        if flat_dim % n_features != 0:
            raise ValueError(
                f"Observation dim {flat_dim} is not a multiple of "
                f"n_features {n_features}"
            )
        self.n_slots    = flat_dim // n_features
        self.n_features = n_features

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, encoded_dim),
            nn.Tanh(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # (batch, n_slots*n_features) → (batch, n_slots, n_features)
        batch = observations.shape[0]
        x = observations.view(batch, self.n_slots, self.n_features)
        # Apply shared encoder to every slot
        x = self.encoder(x)                               # (batch, n_slots, encoded_dim)
        mean_pool = x.mean(dim=1)                         # (batch, encoded_dim)
        max_pool  = x.max(dim=1).values                   # (batch, encoded_dim)
        return th.cat([mean_pool, max_pool], dim=1)       # (batch, 2*encoded_dim)


class SharedEncoderPolicy(ActorCriticPolicy):
    """Actor-critic policy using the shared per-slot encoder.

    Drop-in replacement for SB3's default MlpPolicy — works with the
    standard PPO algorithm and VecNormalize.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ) -> None:
        # Small net_arch because most of the work is done by the encoder
        kwargs.setdefault("features_extractor_class",  SharedSlotEncoder)
        kwargs.setdefault("features_extractor_kwargs", dict(
            n_features=N_FEATURES, hidden_dim=64, encoded_dim=64,
        ))
        kwargs.setdefault("net_arch", dict(pi=[64], vf=[64]))
        kwargs.setdefault("activation_fn", nn.Tanh)
        super().__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs,
        )
