"""Market simulation components for the XBID hybrid trader.

The :mod:`xbid_trader.market` package implements a multi‑product limit order
book simulator.  It provides classes for order books with price–time priority,
matching logic, event‑driven market dynamics, state tracking and configurable
background trading agents.

Key classes include:

* :class:`SingleProductOrderBook` – stores resting orders for a single delivery
  product and matches incoming orders using price–time priority.  Supports
  limit, market and cancellation orders and returns trade events.
* :class:`MarketState` – aggregates the state of all product order books and
  exposes convenience accessors for prices and spreads.
* :class:`MarketEngine` – orchestrates the evolution of the market between
  decision epochs.  It integrates background trading agents, updates the order
  books and advances the global clock.
* Background traders such as :class:`NoiseTrader`, :class:`LiquidityProvider`,
  :class:`UrgencyTrader` and :class:`ForecastInformedTrader` that inject
  realistic liquidity and informational flow into the simulation.
"""

from .order_book import SingleProductOrderBook  # noqa: F401
from .market_state import MarketState  # noqa: F401
from .market_engine import MarketEngine  # noqa: F401
from .battery import (  # noqa: F401,E402
    BatteryConfig,
    soc_dispatch,
    soc_after_slots,
    make_dam_arbitrage_schedule,
    cycle_usage,
)
