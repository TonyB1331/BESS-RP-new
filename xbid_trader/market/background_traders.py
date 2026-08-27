"""Background participant models for the market engine.

This module defines several classes that simulate the behaviour of typical
market participants in intraday power markets.  These agents generate orders
probabilistically based on the current state of the order book, time to
delivery and simple price signals.  They provide exogenous order flow
in the simulator, thereby generating realistic liquidity dynamics and price
movements.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional
from weakref import ref

import numpy as np

from ..types import Order, OrderSide, OrderType
from .market_state import MarketState


class BackgroundTrader(ABC):
    """Abstract base class for background trading agents.

    Subclasses must implement :meth:`generate_orders` which is called by the
    market engine at each event step.  The method returns a list of orders
    (possibly empty) to insert into the market.  Agents should respect gate
    closure times via the provided market state.
    """

    def __init__(self, trader_id: str, order_rate_per_minute: float) -> None:
        self.trader_id = trader_id
        self.order_rate_per_minute = order_rate_per_minute
        self.reference_prices: Optional[np.ndarray] = None
    
    def update_reference_prices(self, reference_prices: np.ndarray) -> None:
        self.reference_prices = np.asarray(reference_prices, dtype=float)
        
    @abstractmethod
    def generate_orders(self, market_state: MarketState, current_time: float) -> List[Order]:  # pragma: no cover
        pass

    def _should_generate(self, delta_minutes: float) -> bool:
        """Return True if at least one order should be generated in the given interval."""
        # Poisson process: probability of at least one arrival in interval dt
        lam = self.order_rate_per_minute * delta_minutes
        return random.random() < 1 - np.exp(-lam)
    
    def _get_anchor_price(self, market_state: MarketState, product_id: int) -> float:
        ob = market_state.order_books[product_id]

        if ob.mid_price is not None:
            return float(ob.mid_price)

        if ob.last_trade_price is not None:
            return float(ob.last_trade_price)

        if self.reference_prices is not None:
            return float(self.reference_prices[product_id])

        return 100.0

class NoiseTrader(BackgroundTrader):
    """Places random limit orders at prices drawn from a normal distribution.

    Noise traders contribute to general liquidity without a specific directional
    view.  They submit both buy and sell orders with equal probability and
    place prices around the current mid price plus random noise.  The order
    quantity is sampled from a log‑normal distribution.
    """

    def __init__(
        self,
        trader_id: str = "noise",
        order_rate_per_minute: float = 1.0,
        price_sigma: float = 2.0,
        volume_mean: float = 1.0,
        volume_std: float = 0.3,
    ) -> None:
        super().__init__(trader_id, order_rate_per_minute)
        self.price_sigma = price_sigma
        self.volume_mean = volume_mean
        self.volume_std = volume_std

    def generate_orders(self, market_state: MarketState, current_time: float) -> List[Order]:
        orders: List[Order] = []
        # Determine whether to generate an order in this small interval
        if not self._should_generate(delta_minutes=1 / 60.0):
            return orders
        num_products = len(market_state.order_books)
        # Choose a random active product
        product_id = random.randint(0, num_products - 1)
        if not market_state.is_active(product_id):
            return orders
        ob = market_state.order_books[product_id]
        mid = ob.mid_price
        if mid is None:
            mid = self._get_anchor_price(market_state, product_id)
        # Randomly decide side
        side = OrderSide.BUY if random.random() < 0.5 else OrderSide.SELL
        volume = float(max(np.random.lognormal(mean=np.log(self.volume_mean), sigma=self.volume_std), 0.01))

        # 25% πιθανότητα να στείλει market order
        if random.random() < 0.25:
            order = Order(
                id=-1,
                product_id=product_id,
                side=side,
                order_type=OrderType.MARKET,
                price=None,
                quantity=volume,
                timestamp=current_time,
                trader_id=self.trader_id,
            )
        else:
            price = float(np.random.normal(mid, self.price_sigma))
            price = max(price, 1e-3)
            order = Order(
                id=-1,
                product_id=product_id,
                side=side,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=volume,
                timestamp=current_time,
                trader_id=self.trader_id,
            )

        orders.append(order)
        return orders


class LiquidityProvider(BackgroundTrader):
    """Posts symmetric bid and ask orders around the mid price.

    Liquidity providers supply depth to the book by maintaining small sizes on
    both sides of the spread.  Prices are placed at fixed offsets relative to
    the mid price and quantities are sampled around a mean.  The arrival
    intensity may be lower than that of noise traders.
    """

    def __init__(
        self,
        trader_id: str = "lp",
        order_rate_per_minute: float = 0.5,
        price_offset: float = 1.0,
        volume_mean: float = 2.0,
        volume_std: float = 0.5,
    ) -> None:
        super().__init__(trader_id, order_rate_per_minute)
        self.price_offset = price_offset
        self.volume_mean = volume_mean
        self.volume_std = volume_std

    def generate_orders(self, market_state: MarketState, current_time: float) -> List[Order]:
        orders: List[Order] = []
        if not self._should_generate(delta_minutes=1 / 60.0):
            return orders
        num_products = len(market_state.order_books)
        product_id = random.randint(0, num_products - 1)
        if not market_state.is_active(product_id):
            return orders
        ob = market_state.order_books[product_id]
        anchor = self._get_anchor_price(market_state, product_id)

        qty = float(max(np.random.lognormal(mean=np.log(self.volume_mean), sigma=self.volume_std), 0.01))

        if ob.best_bid is not None and ob.best_ask is not None:
            bid_price = min(ob.best_bid + 0.01, anchor - 0.01)
            ask_price = max(ob.best_ask - 0.01, anchor + 0.01)

        # fallback αν η βελτίωση καταρρεύσει το spread
            if bid_price >= ask_price:
                bid_price = max(anchor - self.price_offset, 1e-3)
                ask_price = anchor + self.price_offset
        else:
            bid_price = max(anchor - self.price_offset, 1e-3)
            ask_price = anchor + self.price_offset
        orders.append(
            Order(
                id=-1,
                product_id=product_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                quantity=qty,
                timestamp=current_time,
                trader_id=self.trader_id,
            )
        )
        
        orders.append(
            Order(
                id=-1,
                product_id=product_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                quantity=qty,
                timestamp=current_time,
                trader_id=self.trader_id,
            )
        )
        return orders


class UrgencyTrader(BackgroundTrader):
    """Trades aggressively as gate closure approaches.

    The urgency trader models participants who enter the market late and need to
    close positions quickly.  Their order arrival rate increases with reduced
    time to delivery.  Orders are placed as market orders to guarantee
    execution, with sizes drawn from a log‑normal distribution.
    """

    def __init__(
        self,
        trader_id: str = "urgency",
        order_rate_per_minute: float = 0.2,
        aggression_factor: float = 2.0,
        volume_mean: float = 1.5,
        volume_std: float = 0.4,
    ) -> None:
        super().__init__(trader_id, order_rate_per_minute)
        self.aggression_factor = aggression_factor
        self.volume_mean = volume_mean
        self.volume_std = volume_std

    def generate_orders(self, market_state: MarketState, current_time: float) -> List[Order]:
        orders: List[Order] = []
        # Choose a product uniformly from those still active
        active_products = [pid for pid in range(len(market_state.order_books)) if market_state.is_active(pid)]
        if not active_products:
            return orders
        product_id = random.choice(active_products)
        # Determine time to delivery in minutes
        ttd_minutes = market_state.gate_closures[product_id] - market_state.current_time
        # Increase arrival rate as time to delivery shrinks
        effective_rate = self.order_rate_per_minute * (1 + self.aggression_factor / max(ttd_minutes, 1e-3))
        # Convert to per‑second probability
        if random.random() > effective_rate / 60.0:
            return orders
        # Decide whether to buy or sell
        side = OrderSide.BUY if random.random() < 0.5 else OrderSide.SELL
        volume = float(max(np.random.lognormal(mean=np.log(self.volume_mean), sigma=self.volume_std), 0.01))
        orders.append(
            Order(
                id=-1,
                product_id=product_id,
                side=side,
                order_type=OrderType.MARKET,
                price=None,
                quantity=volume,
                timestamp=current_time,
                trader_id=self.trader_id,
            )
        )
        return orders


class ForecastInformedTrader(BackgroundTrader):
    """Trades based on expected price signals from a scenario generator.

    This agent obtains a forward price signal (e.g. from the scenario generator
    or a forecast model) and submits buy or sell orders accordingly.  For
    simplicity this implementation treats the difference between the expected
    imbalance price and the current mid price as the signal.  The trader places
    market orders when the signal exceeds a threshold.
    """

    def __init__(
        self,
        trader_id: str = "forecast",
        order_rate_per_minute: float = 0.3,
        signal_sensitivity: float = 0.5,
        volume_mean: float = 1.0,
        volume_std: float = 0.2,
    ) -> None:
        super().__init__(trader_id, order_rate_per_minute)
        self.signal_sensitivity = signal_sensitivity
        self.volume_mean = volume_mean
        self.volume_std = volume_std
        self.expected_imbalance_prices: Optional[np.ndarray] = None

    def update_signals(self, expected_imbalance_prices: np.ndarray) -> None:
        """Update the expected imbalance price signal for each product."""
        self.expected_imbalance_prices = expected_imbalance_prices

    def generate_orders(self, market_state: MarketState, current_time: float) -> List[Order]:
        orders: List[Order] = []
        if not self._should_generate(delta_minutes=1 / 60.0):
            return orders
        if self.expected_imbalance_prices is None:
            return orders
        num_products = len(market_state.order_books)
        product_id = random.randint(0, num_products - 1)
        if not market_state.is_active(product_id):
            return orders
        # Compute signal: expected imbalance price minus mid price
        ob = market_state.order_books[product_id]
        mid = ob.mid_price
        if mid is None:
            mid = self._get_anchor_price(market_state, product_id)
        signal = float(self.expected_imbalance_prices[product_id] - mid)
        if abs(signal) < self.signal_sensitivity:
            return orders
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        volume = float(max(np.random.lognormal(mean=np.log(self.volume_mean), sigma=self.volume_std), 0.01))
        orders.append(
            Order(
                id=-1,
                product_id=product_id,
                side=side,
                order_type=OrderType.MARKET,
                price=None,
                quantity=volume,
                timestamp=current_time,
                trader_id=self.trader_id,
            )
        )
        return orders