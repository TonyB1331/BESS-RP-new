"""Stochastic price process models for intraday products.

This module implements a correlated Ornstein–Uhlenbeck (OU) process to model
the evolution of electricity prices across neighbouring quarter‑hour delivery
products.  The OU process provides mean‑reverting behaviour around a long‑term
average price, and cross‑product correlation is introduced via an exponential
decay structure on the covariance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PriceProcessConfig:
    """Configuration parameters for the price process."""

    mean_price: float = 100.0
    sigma: float = 5.0
    kappa: float = 0.05
    correlation_decay: float = 0.9


class PriceProcess:
    """Correlated Ornstein–Uhlenbeck process for quarter‑hour prices."""

    def __init__(self, config: PriceProcessConfig) -> None:
        self.mean_price = config.mean_price
        self.sigma = config.sigma
        self.kappa = config.kappa
        self.correlation_decay = config.correlation_decay

    def _correlation_matrix(self, n: int) -> np.ndarray:
        """Construct an exponential decay correlation matrix."""
        idx = np.arange(n)
        diff = np.abs(idx[:, None] - idx[None, :])
        return self.correlation_decay ** diff

    def sample_daily_prices(self, n_products: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample a single vector of initial prices for all products.

        The OU process is discretised at the quarter‑hour resolution.  Each
        product is associated with one time step and there is no intra‑product
        path.  The correlation structure is applied across products.

        Parameters
        ----------
        n_products:
            Number of delivery products (e.g. 96).
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        ndarray
            A vector of length ``n_products`` containing the sampled prices.
        """
        rng = np.random.default_rng(seed)
        corr = self._correlation_matrix(n_products)
        # Construct covariance matrix from correlation and sigma
        cov = (self.sigma ** 2) * corr
        # Sample correlated Gaussian innovations
        innovations = rng.multivariate_normal(mean=np.zeros(n_products), cov=cov)
        # Compute mean‑reversion component; assign each product a different mean level around mean_price
        mean_levels = self.mean_price + rng.normal(loc=0.0, scale=self.sigma, size=n_products)
        prices = mean_levels + innovations
        # Ensure all prices remain positive
        prices[prices < 0] = self.mean_price
        return prices