"""Event‑driven market engine for the XBID hybrid trader.

The :class:`MarketEngine` manages multiple product order books, invokes
background trading agents and advances the simulation clock.  It exposes a
high‑level interface to add orders, run the market until a specified time
increment and query the resulting trades.  All order IDs and trade IDs are
assigned centrally to guarantee uniqueness across products and agents.
"""

from __future__ import annotations

import logging

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import Order, OrderSide, OrderType, Trade
from .order_book import SingleProductOrderBook
from .market_state import MarketState
from .background_traders import (
    BackgroundTrader,
    NoiseTrader,
    LiquidityProvider,
    UrgencyTrader,
    ForecastInformedTrader,
)


class MarketEngine:
    """Simulates a continuous double auction for multiple delivery products.

    Parameters
    ----------
    num_products:
        Number of quarter‑hour delivery products traded in the day (normally 96).
    event_granularity_seconds:
        Time resolution of the event loop in seconds.  Smaller values yield
        finer simulation at the cost of computational overhead.  A typical
        choice is 60 seconds.
    background_agent_configs:
        Optional dictionary containing parameters for each background trader
        type.  Keys should include ``"noise_trader"``, ``"liquidity_provider"``,
        ``"urgency_trader"`` and ``"forecast_informed_trader"``; values are
        dictionaries of keyword arguments forwarded to the respective class
        constructors.
    gate_closure_offset_minutes:
        Number of minutes before the start of the delivery period when trading
        stops for a product.  In European intraday markets gate closure is
        typically 5 or 30 minutes before delivery depending on the bidding zone.
    """

    def __init__(
        self,
        num_products: int = 96,
        event_granularity_seconds: float = 60.0,
        background_agent_configs: Optional[Dict[str, Dict[str, float]]] = None,
        gate_closure_offset_minutes: float = 0.0,
        reference_prices: Optional[np.ndarray] = None,
        seed_books_on_init: bool = True,
        initial_half_spread: float = 0.5,
        initial_levels: int = 3,
        initial_level_qty: float = 5.0,
    ) -> None:
        if reference_prices is None:
            self.reference_prices = np.full(num_products, 100.0, dtype=float)
        else:
            self.reference_prices = np.asarray(reference_prices, dtype=float)
            if self.reference_prices.shape != (num_products,):
                raise ValueError(
                    f"reference_prices must have shape ({num_products},), "
                    f"got {self.reference_prices.shape}"
                )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.num_products = num_products
        self.delta_minutes = event_granularity_seconds / 60.0
        # Create one order book per product
        self.order_books: Dict[int, SingleProductOrderBook] = {
            pid: SingleProductOrderBook(pid) for pid in range(num_products)
        }
        # Compute gate closure times: product p stops trading at (p × 15) - offset
        # Gate closure times: each product stops trading shortly before its delivery
        # period begins.  For the p‑th product (0‑indexed) the gate closure is
        # (p + 1) × 15 minutes minus the offset.  This means that product 0
        # trades until 15 minutes, product 1 until 30 minutes, etc.
        self.gate_closures: List[float] = [
            (p + 1) * 15.0 - gate_closure_offset_minutes for p in range(num_products)
        ]
        # Initialize market state
        self.state = MarketState(order_books=self.order_books, current_time=0.0, gate_closures=self.gate_closures, reference_prices=self.reference_prices,)
        # ID counters
        self._next_order_id: int = 1
        self._next_trade_id: int = 1
        # Create background traders
        self.background_agents: List[BackgroundTrader] = []
        if background_agent_configs is None:
            background_agent_configs = {}
        # Instantiate each agent type based on provided config or defaults
        cfg = background_agent_configs.get("noise_trader", {})
        self.background_agents.append(NoiseTrader(**cfg))
        cfg = background_agent_configs.get("liquidity_provider", {})
        self.background_agents.append(LiquidityProvider(**cfg))
        cfg = background_agent_configs.get("urgency_trader", {})
        self.background_agents.append(UrgencyTrader(**cfg))
        cfg = background_agent_configs.get("forecast_informed_trader", {})
        self.forecast_agent = ForecastInformedTrader(**cfg)
        self.background_agents.append(self.forecast_agent)

        for agent in self.background_agents:
            if hasattr(agent, "update_reference_prices"):
                agent.update_reference_prices(self.reference_prices)
        self.seed_books_on_init = seed_books_on_init
        self.initial_half_spread = initial_half_spread
        self.initial_levels = initial_levels
        self.initial_level_qty = initial_level_qty
        if seed_books_on_init:
            self._seed_initial_books(
                half_spread=initial_half_spread,
                levels=initial_levels,
                qty=initial_level_qty,
    )
    # ------------------------------------------------------------------
    # Order and trade ID assignment
    # ------------------------------------------------------------------
    def _seed_initial_books(
        self,
        half_spread: float,
        levels: int,
        qty: float
    ) -> None:
        for pid in range(self.num_products):
            if not self.state.is_active(pid):
                continue

            mid = float(self.reference_prices[pid])

            for level in range(levels):
                offset = half_spread * (level + 1)

                bid = Order(
                    id=-1,
                    product_id=pid,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=max(mid - offset, 1e-3),
                    quantity=qty,
                    timestamp=self.state.current_time,
                    trader_id=f"seed_bid_{level}",
                )
                ask = Order(
                    id=-1,
                    product_id=pid,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=mid + offset,
                    quantity=qty,
                    timestamp=self.state.current_time,
                    trader_id=f"seed_ask_{level}",
                )

                self.add_external_order(bid)
                self.add_external_order(ask)
    
    def _assign_order_id(self, order: Order) -> Order:
        order.id = self._next_order_id
        self._next_order_id += 1
        return order

    def _assign_trade_ids(self, trades: List[Trade]) -> None:
        for trade in trades:
            trade.trade_id = self._next_trade_id
            self._next_trade_id += 1

    # ------------------------------------------------------------------
    # Simulation functions
    # ------------------------------------------------------------------
    def add_external_order(self, order: Order) -> List[Trade]:
        """Insert an external order (e.g. from the RL agent) into the market.

        The order is assigned a unique identifier and matched against the
        relevant product's order book.  Generated trades receive unique trade
        identifiers.
        """
        # Only add to active products
        if not self.state.is_active(order.product_id):
            self.logger.debug("Ignoring order for expired product %s", order.product_id)
            return []
        order = self._assign_order_id(order)
        ob = self.order_books[order.product_id]
        trades = ob.add_order(order)
        self._assign_trade_ids(trades)
        return trades

    def run_until(self, target_time: float) -> List[Trade]:
        """Advance the simulation until target_time and return all trades."""
        all_trades: List[Trade] = []

        debug_counts = {
            "noise": 0,
            "lp": 0,
            "urgency": 0,
            "forecast": 0,
        }

        while self.state.current_time < target_time:
            self.state.current_time += self.delta_minutes
            current_time = self.state.current_time

            step_orders: List[Order] = []
            for agent in self.background_agents:
                new_orders = agent.generate_orders(self.state, current_time)
                step_orders.extend(new_orders)
                debug_counts[agent.trader_id] = debug_counts.get(agent.trader_id, 0) + len(new_orders)

            for order in step_orders:
                order = self._assign_order_id(order)
                ob = self.order_books[order.product_id]
                trades = ob.add_order(order)
                self._assign_trade_ids(trades)
                all_trades.extend(trades)

        self.logger.debug("Order counts per agent: %s", debug_counts)
        self.logger.debug("Total trades this run: %d", len(all_trades))
        return all_trades

    def set_expected_imbalance_prices(self, signal: np.ndarray) -> None:
        self.forecast_agent.update_signals(signal)

    def set_reference_prices(self, reference_prices: np.ndarray) -> None:
        self.reference_prices = np.asarray(reference_prices, dtype=float)
        self.state.reference_prices = self.reference_prices
        for agent in self.background_agents:
            if hasattr(agent, "update_reference_prices"):
                agent.update_reference_prices(self.reference_prices)
    def reset(self, reference_prices: Optional[np.ndarray] = None) -> None:
        """Reinitialise the engine for a new trading day.

        All order books are cleared and ID counters reset.  Background agents
        keep their configuration but their internal state resets implicitly
        since it is all derived from the fresh MarketState.

        Parameters
        ----------
        reference_prices:
            Optional new reference price vector.  If ``None`` the existing
            ``self.reference_prices`` is reused.
        """
        if reference_prices is not None:
            self.reference_prices = np.asarray(reference_prices, dtype=float)
            for agent in self.background_agents:
                if hasattr(agent, "update_reference_prices"):
                    agent.update_reference_prices(self.reference_prices)

        self.order_books = {
            pid: SingleProductOrderBook(pid) for pid in range(self.num_products)
        }
        self.state = MarketState(
            order_books=self.order_books,
            current_time=0.0,
            gate_closures=self.gate_closures,
            reference_prices=self.reference_prices,
        )
        self._next_order_id = 1
        self._next_trade_id = 1

        if self.seed_books_on_init:
            self._seed_initial_books(
                half_spread=self.initial_half_spread,
                levels=self.initial_levels,
                qty=self.initial_level_qty,
            )
        self.logger.info("MarketEngine reset for new day.")
    def snapshot_order_books(self) -> Dict[int, Dict[str, any]]:
        return {pid: ob.to_dict() for pid, ob in self.order_books.items()}