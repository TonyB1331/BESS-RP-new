"""Limit order book implementation with price–time priority.

This module defines :class:`SingleProductOrderBook`, a data structure that
stores resting buy and sell orders for a single delivery product and matches
incoming orders according to the price–time priority rule.  It supports limit
orders, market orders and cancellation requests, and generates trade events
representing executions.  The matching logic follows the convention used in
European intraday power markets: orders are sorted first by price and then by
time.  Partial fills and queue depletion are fully supported.

References
----------
* In continuous order books all orders are grouped by direction and ranked
  according to the price–time priority principle: the best price level is
  matched first and within the same price the oldest order has priority【593430788322325†L56-L60】.  This implementation
  follows that standard.
* Each quarter‑hourly product is traded in its own order book with specific
  gate opening and closing times.
"""

from __future__ import annotations

import bisect #bisect module provides support for maintaining a list in sorted order without having to sort the list after each insertion. It uses a bisection algorithm to find the correct insertion point for new elements, ensuring that the list remains sorted. In this code, bisect is used to insert new price levels into the sorted lists of bid and ask prices while maintaining their order.
from collections import defaultdict, deque #κάθε price level αντιστοιχίζεται αυτόματα σε ουρά orders, deque:Χρειάζεται για FIFO queue σε κάθε price level
from typing import Deque, Dict, List, Optional, Tuple

from ..types import Order, OrderSide, OrderType, Trade


class SingleProductOrderBook:
    """Order book for a single delivery product.

    The order book maintains two sorted lists of price levels for buys and sells
    along with per‑level queues of orders.  When a new order arrives the
    ``match_order`` method executes trades against resting orders until the
    incoming quantity is exhausted or the order cannot be matched.  Any
    remaining limit order quantity is inserted into the appropriate queue.
    """

    def __init__(self, product_id: int) -> None:
        self.product_id = product_id
        # Maps price levels to deques of orders.  Each deque contains orders in
        # ascending arrival time (FIFO) to implement time priority.
        self._bids: Dict[float, Deque[Order]] = defaultdict(deque)#queue από buy orders
        self._asks: Dict[float, Deque[Order]] = defaultdict(deque)#queue από sell orders
        # Sorted lists of price levels.  For bids we maintain descending order,
        # for asks ascending order.  ``bisect`` is used to insert new levels.
        self._bid_prices: List[float] = []#sorted list με τις τιμές των buy orders
        self._ask_prices: List[float] = []#sorted list με τις τιμές των sell orders
        # Last traded price for this product.  None if no trades have occurred.
        self.last_trade_price: Optional[float] = None #τελευταία τιμή που έγινε trade για το συγκεκριμένο προϊόν

    # ------------------------------------------------------------------
    # Public getters for market statistics
    # ------------------------------------------------------------------
    @property
    def best_bid(self) -> Optional[float]:
        """Return the highest bid price or ``None`` if no bids exist."""
        return self._bid_prices[0] if self._bid_prices else None

    @property
    def best_ask(self) -> Optional[float]:
        """Return the lowest ask price or ``None`` if no asks exist."""
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> Optional[float]:
        """Return the mid price between the best bid and best ask.

        If either side of the book is empty the mid price is undefined and
        ``None`` is returned.  A mid price of ``None`` indicates an illiquid
        market state.
        """
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    # ------------------------------------------------------------------
    # Matching and order insertion
    # ------------------------------------------------------------------
    def add_order(self, order: Order) -> List[Trade]:
        """Process an incoming order and return the list of resulting trades.

        Orders are matched immediately against the opposite side of the book
        subject to price–time priority.  For market orders the price limit is
        ignored (buy orders match the lowest asks; sell orders match the highest
        bids).  Limit orders specify a maximum (for buys) or minimum (for sells)
        price; matching stops once the price condition is violated.  Any
        residual quantity from a limit order is inserted into the book.  Cancel
        orders remove the referenced resting order if found.

        Parameters
        ----------
        order:
            The incoming order to process.

        Returns
        -------
        List[Trade]
            A list of generated trades in the order they occurred.
        """
        trades: List[Trade] = []
        # Handle cancellations first
        if order.order_type == OrderType.CANCEL:
            self._cancel_order(order)
            return trades

        remaining_qty = order.quantity
        # Determine match condition depending on order side
        if order.side == OrderSide.BUY:
            # Price limit for matching.  None means a market order with no limit.
            price_limit = order.price
            # Continue matching while there is quantity and asks are available
            while remaining_qty > 0 and self._ask_prices:
                best_ask_price = self._ask_prices[0]
                # Check price condition: for market orders ``price_limit`` is None and therefore
                # always satisfied.  For limit orders the best ask must be <= price_limit.
                if price_limit is not None and best_ask_price > price_limit:
                    break
                # Take the earliest order at the best ask price
                ask_queue = self._asks[best_ask_price]
                resting_order = ask_queue[0]
                trade_qty = min(remaining_qty, resting_order.quantity)
                trade_price = resting_order.price if resting_order.price is not None else best_ask_price
                trades.append(
                    Trade(
                        trade_id=-1,  # The market engine will assign an ID
                        product_id=self.product_id,
                        price=trade_price,
                        quantity=trade_qty,
                        buy_order_id=order.id,
                        sell_order_id=resting_order.id,
                        timestamp=order.timestamp,
                    )
                )
                # Update quantities
                remaining_qty -= trade_qty #μειώνουμε την ποσότητα που απομένει να εκτελεστεί από την εισερχόμενη εντολή
                resting_order.quantity -= trade_qty #μειώνουμε την ποσότητα που απομένει στην εντολή που εκτελέστηκε μερικώς
                self.last_trade_price = trade_price #ενημερώνουμε την τελευταία τιμή που έγινε trade για το συγκεκριμένο προϊόν
                # Remove the resting order if fully filled
                if resting_order.quantity <= 0:#αν η εντολή που εκτελέστηκε έχει μηδενική ποσότητα τότε αφαιρείται από την ουρά
                    ask_queue.popleft()#αφαιρούμε την εντολή από την ουρά των εντολών στο συγκεκριμένο price level
                    if not ask_queue:
                        # Remove price level when empty
                        del self._asks[best_ask_price]
                        self._ask_prices.pop(0)
                # If the incoming order was a market order and there are no more asks then exit
                if price_limit is None and not self._ask_prices:
                    break
            # Insert remaining quantity into book if limit order
            if order.order_type == OrderType.LIMIT and remaining_qty > 0: #
                self._insert_bid(order, remaining_qty)

        else:  # SELL
            price_limit = order.price
            while remaining_qty > 0 and self._bid_prices:
                best_bid_price = self._bid_prices[0]
                if price_limit is not None and best_bid_price < price_limit:#για market order price_limit είναι None και η συνθήκη είναι πάντα αληθής, για limit order το best bid πρέπει να είναι >= price_limit για να συνεχίσει το matching
                    break
                bid_queue = self._bids[best_bid_price]
                resting_order = bid_queue[0]
                trade_qty = min(remaining_qty, resting_order.quantity)
                trade_price = resting_order.price if resting_order.price is not None else best_bid_price
                trades.append(
                    Trade(
                        trade_id=-1,
                        product_id=self.product_id,
                        price=trade_price,
                        quantity=trade_qty,
                        buy_order_id=resting_order.id,
                        sell_order_id=order.id,
                        timestamp=order.timestamp,
                    )
                )
                remaining_qty -= trade_qty #μειώνουμε την ποσότητα που απομένει να εκτελεστεί από την εισερχόμενη εντολή
                resting_order.quantity -= trade_qty #μειώνουμε την ποσότητα που απομένει στην εντολή που εκτελέστηκε μερικώς
                self.last_trade_price = trade_price #ενημερώνουμε την τελευταία τιμή που έγινε trade για το συγκεκριμένο προϊόν
                if resting_order.quantity <= 0:#αν η εντολή που εκτελέστηκε έχει μηδενική ποσότητα τότε αφαιρείται από την ουρά
                    bid_queue.popleft()#αφαιρούμε την εντολή από την ουρά των εντολών στο συγκεκριμένο price level
                    if not bid_queue:
                        del self._bids[best_bid_price]
                        self._bid_prices.pop(0)
                if price_limit is None and not self._bid_prices:
                    break
            if order.order_type == OrderType.LIMIT and remaining_qty > 0:
                self._insert_ask(order, remaining_qty)
        return trades

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    def _insert_bid(self, order: Order, quantity: float) -> None:
        """Insert the remaining portion of a buy limit order into the book."""
        price = order.price
        assert price is not None, "Limit buy order must have a price"
        new_order = Order(
            id=order.id,
            product_id=order.product_id,
            side=order.side,
            order_type=order.order_type,
            price=price,
            quantity=quantity,
            timestamp=order.timestamp,
            trader_id=order.trader_id,
        )
        if price not in self._bids:
            # Insert price into sorted list in descending order
            idx = len(self._bid_prices) - bisect.bisect_left(list(reversed(self._bid_prices)), price)
            self._bid_prices.insert(idx, price)
        self._bids[price].append(new_order)

    def _insert_ask(self, order: Order, quantity: float) -> None:
        """Insert the remaining portion of a sell limit order into the book."""
        price = order.price
        assert price is not None, "Limit sell order must have a price"
        new_order = Order(
            id=order.id,
            product_id=order.product_id,
            side=order.side,
            order_type=order.order_type,
            price=price,
            quantity=quantity,
            timestamp=order.timestamp,
            trader_id=order.trader_id,
        )
        if price not in self._asks:
            # Insert price into sorted list in ascending order
            idx = bisect.bisect_left(self._ask_prices, price)
            self._ask_prices.insert(idx, price)
        self._asks[price].append(new_order)

    def _cancel_order(self, cancel: Order) -> None:
        """Remove a resting order from the book if the identifier matches."""
        target_id = cancel.original_order_id
        if target_id is None:
            return
        # Search through bids
        for price in list(self._bids.keys()):
            queue = self._bids[price]
            for idx, order in enumerate(queue):
                if order.id == target_id:
                    # Remove and adjust price levels if needed
                    del queue[idx]
                    if not queue:
                        del self._bids[price]
                        self._bid_prices.remove(price)
                    return
        # Search through asks
        for price in list(self._asks.keys()):
            queue = self._asks[price]
            for idx, order in enumerate(queue):
                if order.id == target_id:
                    del queue[idx]
                    if not queue:
                        del self._asks[price]
                        self._ask_prices.remove(price)
                    return

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_book_depth(self) -> Tuple[Dict[float, float], Dict[float, float]]:
        """Return the aggregated quantity available at each bid and ask price.

        Returns
        -------
        (bids, asks):
            Two dictionaries mapping price levels to total volume for bids and
            asks respectively.
        """
        bid_depth = {p: sum(o.quantity for o in q) for p, q in self._bids.items()}
        ask_depth = {p: sum(o.quantity for o in q) for p, q in self._asks.items()}
        return bid_depth, ask_depth

    def to_dict(self) -> Dict[str, any]:
        """Serialise a snapshot of the order book for debugging or logging."""
        bid_depth, ask_depth = self.get_book_depth()
        return {
            "product_id": self.product_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "last_trade_price": self.last_trade_price,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
        }