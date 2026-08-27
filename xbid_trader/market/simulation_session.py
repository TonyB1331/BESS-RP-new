"""Session-level orchestrator for the intraday market simulation.

:class:`SimulationSession` wraps :class:`~market_engine.MarketEngine` and
exposes a simple :meth:`step` interface that advances the simulation by one
15-minute slot at a time.  After 96 slots (one full trading day) the session
flags completion and can be reset for a new day.

Typical usage::

    session = SimulationSession()

    while True:
        input("Press Enter to advance 15 minutes...")
        result = session.step()

        if session.is_day_complete:
            session.reset()

The session also prints a formatted summary at the end of each step showing
the order book depth and last trade for every product that is still active.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..types import Order
from .market_engine import MarketEngine
from .market_state import MarketState
from .step_result import StepResult
from .custom_agent_interface import CustomAgent


# Duration of each slot in minutes
_SLOT_MINUTES: float = 15.0


class SimulationSession:
    """Manages a day's worth of 15-minute simulation slots.

    Parameters
    ----------
    engine_config:
        Keyword arguments forwarded verbatim to :class:`MarketEngine`.
        You can override ``num_products``, ``background_agent_configs``,
        ``reference_prices``, etc.
    custom_agent:
        An optional :class:`~custom_agent_interface.CustomAgent` instance.
        If provided its :meth:`~custom_agent_interface.CustomAgent.on_step_start`
        is called at the beginning of every slot and
        :meth:`~custom_agent_interface.CustomAgent.on_step_end` after the
        background simulation finishes.
    print_summary:
        Whether to print the formatted book/trade summary after each step.
        Defaults to ``True``.
    top_n_levels:
        How many price levels to show per side in the printed summary.
        Defaults to 5.
    """

    def __init__(
        self,
        engine_config: Optional[Dict] = None,
        custom_agent: Optional[CustomAgent] = None,
        print_summary: bool = True,
        top_n_levels: int = 5,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._engine_config: Dict = engine_config or {}
        self.custom_agent = custom_agent
        self.print_summary = print_summary
        self.top_n_levels = top_n_levels

        self.current_slot: int = 0
        self.day: int = 0
        self.is_day_complete: bool = False

        self._engine: MarketEngine = self._build_engine()

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _build_engine(self) -> MarketEngine:
        """Instantiate a fresh MarketEngine with the stored config."""
        return MarketEngine(**self._engine_config)

    def reset(self, reference_prices: Optional[np.ndarray] = None) -> None:
        """Reset the session for a new trading day.

        Increments the day counter, resets the slot to 0, and rebuilds the
        MarketEngine from scratch so all order books start empty.

        Parameters
        ----------
        reference_prices:
            Optional new reference price vector for the next day.  If
            ``None`` the prices from ``engine_config`` (or the engine
            defaults) are used.
        """
        self.day += 1
        self.current_slot = 0
        self.is_day_complete = False

        if reference_prices is not None:
            self._engine_config["reference_prices"] = reference_prices
            # Update num_products to match the new day's slot count (92/96/100)
            self._engine_config["num_products"] = len(reference_prices)

        self._engine = self._build_engine()
        self.logger.info(
            "Session reset — starting day %d (%d slots)",
            self.day, self._engine.num_products,
        )

    # ------------------------------------------------------------------
    # Main step interface
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        """Advance the simulation by one 15-minute slot and return a result.

        The sequence within a step is:

        1. (Optional) Ask the custom agent for orders and inject them.
        2. Run background agents until ``(current_slot + 1) * 15`` minutes.
        3. Collect book snapshots and build a :class:`~step_result.StepResult`.
        4. (Optional) Call the custom agent's ``on_step_end`` callback.
        5. Advance the slot counter and check for day completion.
        6. (Optional) Print the formatted summary.

        Returns
        -------
        StepResult
            Fully populated result for this slot.

        Raises
        ------
        RuntimeError
            If called when the day is already complete and
            :meth:`reset` has not been called.
        """
        if self.is_day_complete:
            raise RuntimeError(
                "Day is complete. Call reset() before stepping into a new day."
            )

        time_start = float(self.current_slot * _SLOT_MINUTES)
        time_end = float((self.current_slot + 1) * _SLOT_MINUTES)

        # ── 1. Custom agent orders ────────────────────────────────────
        agent_trades = []
        if self.custom_agent is not None:
            agent_orders: List[Order] = self.custom_agent.on_step_start(
                slot=self.current_slot,
                day=self.day,
                state=self._engine.state,
            )
            for order in agent_orders:
                trades = self._engine.add_external_order(order)
                agent_trades.extend(trades)

        # ── 2. Background simulation ──────────────────────────────────
        background_trades = self._engine.run_until(time_end)

        # ── 3. Build result ───────────────────────────────────────────
        state = self._engine.state
        active_products = [
            pid for pid in range(self._engine.num_products)
            if state.is_active(pid)
        ]

        book_depths: Dict[int, tuple] = {}
        last_trade_prices: Dict[int, Optional[float]] = {}
        for pid in active_products:
            ob = self._engine.order_books[pid]
            book_depths[pid] = ob.get_book_depth()
            last_trade_prices[pid] = ob.last_trade_price

        result = StepResult(
            slot=self.current_slot,
            day=self.day,
            time_start=time_start,
            time_end=time_end,
            agent_trades=agent_trades,
            background_trades=background_trades,
            price_snapshot=state.snapshot_prices(),
            book_depths=book_depths,
            last_trade_prices=last_trade_prices,
            active_products=active_products,
            is_day_complete=(self.current_slot == self._engine.num_products - 1),
        )

        # ── 4. Agent end-of-step callback ─────────────────────────────
        if self.custom_agent is not None:
            self.custom_agent.on_step_end(result)

        # ── 5. Advance slot counter ───────────────────────────────────
        self.current_slot += 1
        if self.current_slot >= self._engine.num_products:
            self.is_day_complete = True

        # ── 6. Print summary ──────────────────────────────────────────
        if self.print_summary:
            self._print_step_summary(result)

        return result

    # ------------------------------------------------------------------
    # Convenience: interactive loop
    # ------------------------------------------------------------------

    def run_interactive(self) -> None:
        """Run the simulation interactively, advancing on Enter.

        The loop runs until the user types ``q`` + Enter.  When a day
        completes it automatically resets and continues with the next day.
        """
        print(
            "\n╔══════════════════════════════════════════════════════╗"
            "\n║   XBID Intraday Market Simulation — Interactive Mode ║"
            "\n║   Press Enter to advance 15 min  |  'q' to quit      ║"
            "\n╚══════════════════════════════════════════════════════╝\n"
        )
        while True:
            try:
                user_input = input(
                    f"[Day {self.day:02d} | Slot {self.current_slot:02d}/95 | "
                    f"T={self.current_slot * 15:04d}min] "
                    "Press Enter to advance (q=quit): "
                )
            except (KeyboardInterrupt, EOFError):
                print("\nSimulation interrupted.")
                break

            if user_input.strip().lower() == "q":
                print("Simulation ended by user.")
                break

            self.step()

            if self.is_day_complete:
                print(
                    f"\n{'═' * 60}"
                    f"\n  Day {self.day} complete — resetting for Day {self.day + 1}"
                    f"\n{'═' * 60}\n"
                )
                self.reset()

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def _print_step_summary(self, result: StepResult) -> None:
        """Print a formatted summary of the step to stdout."""
        # ── Header ────────────────────────────────────────────────────
        slot_start_h = int(result.time_start // 60)
        slot_start_m = int(result.time_start % 60)
        slot_end_h = int(result.time_end // 60)
        slot_end_m = int(result.time_end % 60)

        print(
            f"\n{'═' * 70}\n"
            f"  Day {result.day:02d} | Slot {result.slot:02d} | "
            f"{slot_start_h:02d}:{slot_start_m:02d} → {slot_end_h:02d}:{slot_end_m:02d}"
            f"  ({len(result.all_trades)} trades, "
            f"{result.total_volume:.2f} MW total)\n"
            f"{'═' * 70}"
        )

        # ── Per-product detail ────────────────────────────────────────
        # Show only products that are still active AND deliver in the
        # future (product_id > current slot, i.e. not yet at gate closure)
        future_products = sorted(result.active_products)

        if not future_products:
            print("  No active products remaining.\n")
            return

        for pid in future_products:
            ob = self._engine.order_books[pid]

            # Delivery window label
            delivery_start_m = pid * 15
            delivery_end_m = delivery_start_m + 15
            d_h_s, d_m_s = divmod(delivery_start_m, 60)
            d_h_e, d_m_e = divmod(delivery_end_m, 60)
            label = f"{d_h_s:02d}:{d_m_s:02d}–{d_h_e:02d}:{d_m_e:02d}"

            bid_depth, ask_depth = ob.get_book_depth()
            last_px = ob.last_trade_price

            # Sorted asks ascending, bids descending
            sorted_asks = sorted(ask_depth.items(), key=lambda x: x[0])
            sorted_bids = sorted(bid_depth.items(), key=lambda x: x[0], reverse=True)

            top_asks = sorted_asks[: self.top_n_levels]
            top_bids = sorted_bids[: self.top_n_levels]

            print(f"\n  ┌── Product {pid:02d}  [{label}]")

            # Print asks top-to-bottom (highest ask first so the spread is visible)
            if top_asks:
                print(f"  │  {'ASK':>30}")
                for px, qty in reversed(top_asks):
                    print(f"  │  {'':>18}{px:>8.2f}  {qty:>6.2f} MW")
            else:
                print(f"  │  {'(no asks)':>30}")

            # Spread indicator
            best_bid = ob.best_bid
            best_ask = ob.best_ask
            if best_bid is not None and best_ask is not None:
                spread = best_ask - best_bid
                mid = (best_ask + best_bid) / 2.0
                print(f"  │  {'─── mid':>14} {mid:8.2f}  spread {spread:.2f} €/MWh")
            else:
                print(f"  │  {'─── (no spread)':>30}")

            if top_bids:
                print(f"  │  {'BID':<30}")
                for px, qty in top_bids:
                    print(f"  │  {px:>8.2f}  {qty:>6.2f} MW")
            else:
                print(f"  │  {'(no bids)':>30}")

            # Last trade
            if last_px is not None:
                print(f"  │  Last trade : {last_px:.2f} €/MWh")
            else:
                print(f"  │  Last trade : —")

            print(f"  └{'─' * 50}")

        # ── Step trade summary ────────────────────────────────────────
        if result.agent_trades:
            print(f"\n  Agent trades this slot: {len(result.agent_trades)}")
            for t in result.agent_trades:
                print(
                    f"    Trade {t.trade_id}: product {t.product_id:02d} | "
                    f"{t.quantity:.2f} MW @ {t.price:.2f} €/MWh"
                )
        print()
