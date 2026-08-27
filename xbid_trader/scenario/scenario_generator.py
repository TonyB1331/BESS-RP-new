"""Full day scenario generation for the XBID hybrid trader.

Updated to pass DAM reference prices into the ForecastErrorModel so that
imbalance prices are anchored to real market levels rather than centred
at zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .price_process import PriceProcess, PriceProcessConfig
from .forecast_model import ForecastErrorModel, ForecastErrorConfig


@dataclass
class Scenario:
    """Data class holding a single day scenario for all products."""

    prices: np.ndarray
    res_errors: np.ndarray
    load_errors: np.ndarray
    imbalance_prices: np.ndarray

    @property
    def n_products(self) -> int:
        return len(self.prices)

    @property
    def rt(self) -> np.ndarray:
        """Net system imbalance: load_error − res_error."""
        return self.load_errors - self.res_errors


class ScenarioGenerator:
    """Generates daily scenarios for price and forecast error processes."""

    def __init__(
        self,
        n_products: int = 96,
        price_config: Optional[PriceProcessConfig] = None,
        error_config: Optional[ForecastErrorConfig] = None,
    ) -> None:
        self.n_products = n_products
        self.price_process = PriceProcess(price_config or PriceProcessConfig())
        self.error_model = ForecastErrorModel(error_config or ForecastErrorConfig())

    def generate_day_scenario(
        self,
        seed: Optional[int] = None,
        dam_reference_prices: Optional[np.ndarray] = None,
    ) -> Scenario:
        """Sample a complete daily scenario.

        Parameters
        ----------
        seed:
            Optional random seed.
        dam_reference_prices:
            Real DAM prices to anchor imbalance prices.  If ``None``,
            prices are sampled from the stochastic price process and
            used as the anchor.

        Returns
        -------
        Scenario
        """
        if seed is not None:
            price_seed = seed
            error_seed = seed + 1
        else:
            price_seed = None
            error_seed = None

        # Use provided DAM prices or sample from price process
        if dam_reference_prices is not None:
            prices = np.asarray(dam_reference_prices, dtype=float)
        else:
            prices = self.price_process.sample_daily_prices(
                self.n_products, seed=price_seed
            )

        # Pass DAM prices to forecast model so imbalance is realistic
        res_errors, load_errors, imbalance_prices = self.error_model.sample_errors(
            self.n_products,
            seed=error_seed,
            dam_reference_prices=prices,
        )

        return Scenario(
            prices=prices,
            res_errors=res_errors,
            load_errors=load_errors,
            imbalance_prices=imbalance_prices,
        )
