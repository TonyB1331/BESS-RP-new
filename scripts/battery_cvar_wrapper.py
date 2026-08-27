"""CVaR reward shaping for the battery agent — downside-of-PnL variant.

The supplier version penalised the CVaR of *residual settlement cost*.  A
battery has no involuntary residual; its risk is the **downside of daily
arbitrage profit** — days where prices moved against the committed schedule.

Loss per episode:
    L = − economic_pnl = −(ID_cash_flow − imbalance_cost − degradation)

Shaping (Rockafellar–Uryasev):
    r_extra = − λ_cvar · (1/α) · max(L − VaR_α, 0)

where ``VaR_α`` is the running (1−α)-quantile of past-episode losses, so the
agent is pushed to reduce the worst-α% loss days (risk-averse arbitrage).
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import gymnasium as gym
import numpy as np


class BatteryCVaRRewardWrapper(gym.Wrapper):
    """Episode-level CVaR shaping on the downside of battery economic PnL."""

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.10,
        lambda_cvar: float = 1.0,
        window: int = 200,
        warmup_episodes: int = 50,
    ) -> None:
        super().__init__(env)
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        self.alpha           = float(alpha)
        self.lambda_cvar     = float(lambda_cvar)
        self.warmup_episodes = int(warmup_episodes)
        self._losses: Deque[float] = deque(maxlen=int(window))
        self._episode_return: float = 0.0

    def reset(self, **kwargs):
        self._episode_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        done = bool(terminated or truncated)

        if done:
            economic_pnl = float(info.get("ep_economic_pnl", self._episode_return))
            loss = -economic_pnl

            if self.lambda_cvar > 0.0 and len(self._losses) >= self.warmup_episodes:
                var_estimate = float(np.quantile(np.asarray(self._losses), 1.0 - self.alpha))
                excess = max(loss - var_estimate, 0.0)
                penalty = self.lambda_cvar * excess / self.alpha
                reward -= penalty
                info["cvar_var_loss"] = var_estimate
                info["cvar_excess"]   = excess
                info["cvar_penalty"]  = penalty

            self._losses.append(loss)
            info["episode_return_raw"] = self._episode_return
            info["episode_loss"]       = loss

        return obs, reward, terminated, truncated, info
