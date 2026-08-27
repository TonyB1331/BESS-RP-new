"""Top‑level package for the XBID hybrid trader.

This package contains the core classes and utilities used throughout the hybrid
intraday electricity trading system.  The code is organised into several
subpackages:

* :mod:`xbid_trader.market` implements the multi‑product continuous double auction
  simulator, including order objects, order book mechanics, matching logic,
  event‑driven market engine and background participant models.
* :mod:`xbid_trader.scenario` defines the historical scenario provider (Rt,
  DAM position and settlement prices from real data) plus a synthetic
  fallback generator for offline experimentation.
* :mod:`xbid_trader.utils` contains utility functions for reproducibility and
  configuration loading.

The reinforcement-learning system (Gymnasium environment, shared-encoder
PPO policy, CVaR reward shaping and the Optuna meta-layer) lives in
``xbid_trader.market`` and the ``scripts/`` package.  See the README for an
overview of the system architecture.
"""

from .types import OrderSide, OrderType, Order, Trade  # noqa: F401