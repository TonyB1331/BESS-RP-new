"""Representation of the market state for the hybrid trader.

The :class:`MarketState` class aggregates information across all active
delivery products.  It owns a mapping from product identifiers to
:class:`~xbid_trader.market.order_book.SingleProductOrderBook` instances and
exposes convenience properties such as best bids, best asks and mid prices.
The state also tracks the current simulation time and whether each product
is still tradable (i.e. before gate closure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..types import OrderSide
from .order_book import SingleProductOrderBook


@dataclass
class MarketState:
    """Aggregated state of the multi‑product market.

    Parameters
    ----------
    order_books:
        Dictionary mapping product identifiers to the corresponding order book.
    current_time:
        Current simulation time measured in minutes since the start of the day.
    gate_closures:
        Vector of gate closure times (in minutes) for each product.  Once
        ``current_time`` exceeds the closure time of a product the product is
        considered expired and trading ceases.
    """

    order_books: Dict[int, SingleProductOrderBook]
    current_time: float = 0.0
    gate_closures: List[float] = field(default_factory=list)
    reference_prices: Optional[np.ndarray] = None
    
    def reference_price(self, product_id: int) -> float: #επιστρέφει την τιμή αναφοράς για το συγκεκριμένο προϊόν, αν δεν υπάρχουν τιμές αναφοράς επιστρέφει 100.0
        if self.reference_prices is None:
            return 100.0
        return float(self.reference_prices[product_id])
    
    def is_active(self, product_id: int) -> bool:
        """Return whether the specified product is still tradable."""
        return self.current_time < self.gate_closures[product_id]

    def best_bid(self, product_id: int) -> Optional[float]:
        """Return the best bid for a given product or ``None`` if none exists."""
        ob = self.order_books[product_id]
        return ob.best_bid

    def best_ask(self, product_id: int) -> Optional[float]:
        """Return the best ask for a given product or ``None`` if none exists."""
        ob = self.order_books[product_id]
        return ob.best_ask

    def mid_price(self, product_id: int) -> Optional[float]:
        """Return the mid price for a given product or ``None`` if undefined."""
        ob = self.order_books[product_id]
        return ob.mid_price

    def snapshot_prices(self) -> Dict[str, List[Optional[float]]]:
        """Return lists of best bids, best asks and mid prices for all products."""
        bids = [self.best_bid(pid) for pid in range(len(self.order_books))]
        asks = [self.best_ask(pid) for pid in range(len(self.order_books))]
        mids = [self.mid_price(pid) for pid in range(len(self.order_books))]
        return {"bid": bids, "ask": asks, "mid": mids}

    def time_to_delivery(self) -> np.ndarray:
        """Return an array of minutes to delivery for each product.

        Products that have already expired (``current_time`` ≥ ``gate_closures``)
        return zero.  The vector length equals the number of products.
        """
        gate_array = np.array(self.gate_closures, dtype=float)
        ttd = gate_array - self.current_time
        ttd[ttd < 0] = 0.0
        return ttd