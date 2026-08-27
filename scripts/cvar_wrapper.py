"""CVaR reward shaping for episodic RL — battery downside-loss variant.

Penalises episodes with the worst (most negative) arbitrage results, making
the policy risk-averse in the Rockafellar–Uryasev CVaR sense.

    L = −economic_pnl_of_the_episode        (€ loss; profit → negative loss)

so that a day that *lost* money contributes a large positive ``L``.

    r_extra = −λ_cvar × (1/α) × max(L − VaR_L, 0)

where VaR_L = running (1−α)-quantile of past episodes' losses.  Only the
worst ``α`` tail of days is penalised, shaping the policy toward avoiding
large drawdowns (e.g. days where it over-commits and pays imbalance, or buys
high / sells low).

Theoretical basis: Rockafellar & Uryasev (2000), CVaR_α(L).
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import gymnasium as gym
import numpy as np


class CVaRRewardWrapper(gym.Wrapper):
    """Episode-level CVaR shaping on the battery's downside loss (€).

    Parameters
    ----------
    env:
        The base XBID environment.
    alpha:
        Risk level in (0, 1].  0.1 = worst 10%.
    lambda_cvar:
        Weight on the CVaR penalty.
    window:
        Sliding window size for quantile estimation.
    warmup_episodes:
        Skip shaping until this many episodes observed.
    """

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

        # Track per-episode LOSS (€) = −economic_pnl
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
            # Per-episode loss (€) = −economic_pnl.  Falls back to the raw
            # (shaped) episode return if the economic metric is unavailable.
            loss = float(info.get("ep_loss", -self._episode_return))

            if (
                self.lambda_cvar > 0.0
                and len(self._losses) >= self.warmup_episodes
            ):
                # Upper tail: (1-α) quantile — large loss = bad
                var_estimate = float(
                    np.quantile(np.asarray(self._losses), 1.0 - self.alpha)
                )
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
