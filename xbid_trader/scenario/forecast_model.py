"""Forecast error models for renewable energy and load.

Redesigned to produce realistic signals for the RL agent:

1. ``imbalance_prices`` are anchored to DAM reference prices plus a
   spread and noise term — so the signal γt,i = E[pBM] - mid is
   meaningful and centered near zero rather than always negative.

2. RES and load forecast errors have **temporal autocorrelation** via an
   AR(1) process — neighbouring slots are correlated, reflecting the
   persistence of forecast errors in practice.

3. ``Rt`` (net system imbalance) is derived from the errors in a
   physically consistent way: Rt = load_error - res_error, so positive
   Rt means the system needs more power (deficit).

4. The imbalance price spread over DAM depends on Rt — when the system
   is short (Rt > 0) the imbalance price tends to be above DAM, and
   vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class ForecastErrorConfig:
    """Configuration for the forecast error and imbalance price models.

    Parameters
    ----------
    res_sigma:
        Standard deviation of the RES forecast error innovations (MW).
    load_sigma:
        Standard deviation of the load forecast error innovations (MW).
    ar1_coef:
        AR(1) autocorrelation coefficient for both error processes.
        Higher values → smoother, more persistent error paths.
        Must be in [0, 1).  Default 0.7.
    imbalance_spread_mean:
        Mean spread between imbalance price and DAM price in €/MWh.
        Positive → imbalance price tends to be above DAM on average.
        Default 5.0.
    imbalance_spread_sigma:
        Standard deviation of the imbalance price spread innovations.
        Default 3.0.
    imbalance_rt_sensitivity:
        How strongly the imbalance price responds to Rt.
        Units: €/MWh per MW of imbalance.  Default 0.05.
    imbalance_ar1_coef:
        AR(1) coefficient for the imbalance price spread path.
        Default 0.8.
    """

    res_sigma:                  float = 30.0   # MW — realistic RES forecast error
    load_sigma:                 float = 20.0   # MW — realistic load forecast error
    ar1_coef:                   float = 0.7    # temporal autocorrelation
    imbalance_spread_mean:      float = 5.0    # €/MWh above DAM on average
    imbalance_spread_sigma:     float = 3.0    # €/MWh noise
    imbalance_rt_sensitivity:   float = 0.05   # €/MWh per MW of Rt
    imbalance_ar1_coef:         float = 0.8    # smoothness of imbalance path


class ForecastErrorModel:
    """Generates RES/load forecast errors and imbalance prices.

    All outputs are correlated with the DAM reference prices and exhibit
    temporal autocorrelation via AR(1) processes.
    """

    def __init__(self, config: ForecastErrorConfig) -> None:
        self.cfg = config

    def sample_errors(
        self,
        n_products: int,
        seed: Optional[int] = None,
        dam_reference_prices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample forecast error trajectories for all products.

        Parameters
        ----------
        n_products:
            Number of quarter-hour products (typically 96).
        seed:
            Optional random seed for reproducibility.
        dam_reference_prices:
            DAM reference prices in €/MWh, shape ``(n_products,)``.
            If provided, imbalance prices are anchored to these.
            If ``None``, a flat 100 €/MWh baseline is used.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(res_errors, load_errors, imbalance_prices)`` — each shape
            ``(n_products,)``.

            - ``res_errors``: RES generation forecast error in MW.
              Positive = more RES than forecast (surplus contribution).
            - ``load_errors``: Load forecast error in MW.
              Positive = more load than forecast (deficit contribution).
            - ``imbalance_prices``: Expected imbalance settlement price
              in €/MWh, correlated with DAM prices.
        """
        rng = np.random.default_rng(seed)

        if dam_reference_prices is None:
            dam_ref = np.full(n_products, 100.0)
        else:
            dam_ref = np.asarray(dam_reference_prices, dtype=float)

        # ── AR(1) RES forecast errors ─────────────────────────────────
        res_errors = self._ar1_path(
            n=n_products,
            sigma=self.cfg.res_sigma,
            phi=self.cfg.ar1_coef,
            rng=rng,
        )

        # ── AR(1) load forecast errors ────────────────────────────────
        load_errors = self._ar1_path(
            n=n_products,
            sigma=self.cfg.load_sigma,
            phi=self.cfg.ar1_coef,
            rng=rng,
        )

        # ── Net system imbalance Rt = load_error - res_error ──────────
        # Positive → system deficit (load higher or RES lower than expected)
        # Negative → system surplus
        rt = load_errors - res_errors

        # ── Imbalance price = DAM + spread(Rt) + AR(1) noise ──────────
        # The spread has three components:
        #   1. Fixed mean premium above DAM
        #   2. Rt-dependent component (scarcity pricing)
        #   3. AR(1) noise path
        rt_component = self.cfg.imbalance_rt_sensitivity * rt

        spread_noise = self._ar1_path(
            n=n_products,
            sigma=self.cfg.imbalance_spread_sigma,
            phi=self.cfg.imbalance_ar1_coef,
            rng=rng,
        )

        imbalance_prices = (
            dam_ref
            + self.cfg.imbalance_spread_mean
            + rt_component
            + spread_noise
        )

        return res_errors, load_errors, imbalance_prices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ar1_path(
        n: int,
        sigma: float,
        phi: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a zero-mean AR(1) path of length n.

        x_t = phi * x_{t-1} + eps_t,  eps_t ~ N(0, sigma * sqrt(1 - phi^2))

        The innovation std is scaled so the stationary variance equals sigma^2
        regardless of phi.
        """
        innovation_std = sigma * np.sqrt(max(1.0 - phi ** 2, 1e-6))
        eps = rng.normal(0.0, innovation_std, size=n)
        x = np.empty(n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]
        return x
