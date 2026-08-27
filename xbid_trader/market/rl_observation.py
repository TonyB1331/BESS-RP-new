"""RL observation builder for the XBID intraday BATTERY agent.

Constructs the state matrix ``S_t`` of shape ``(num_products, N_FEATURES)``
fed to the RL policy at the start of every 15-minute decision slot.  This is
the *battery* variant: the supplier-specific residual-imbalance features
(``Rt``, DAM load position, load/RES forecast errors, hedge-gated PnL) are
replaced by battery state features (State of Charge, committed net position,
value-of-energy reference) and a **real cash-flow PnL** accounting.

Key differences vs the supplier builder
---------------------------------------
* ``self._net_position`` — committed net energy per delivery slot (MWh).
  Sign convention (see :mod:`xbid_trader.market.battery`):
  ``+`` = discharge/SELL, ``−`` = charge/BUY.  Initialised from the DAM
  schedule and updated by every intraday fill.
* ``self._realized_pnl`` — **actual** intraday cash flow: a SELL adds
  ``+price·qty`` (money received), a BUY adds ``−price·qty`` (money paid).
  There is *no* hedge-gating; for a merchant battery, capturing the spread
  **is** the objective, constrained only by SoC/power feasibility.
* ``self._value_ref`` — expected value of energy per slot (a noisy forecast,
  time-decaying), used to build the ``γ = value_ref − mid`` arbitrage signal.
  The clean value is kept in ``self._real_value_ref`` for settlement.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from ..types import Trade
from .market_state import MarketState
from .battery import BatteryConfig, soc_after_slots


# ---------------------------------------------------------------------------
# Feature index constants  (N_FEATURES kept at 23 so the shared-slot encoder
# input dimension is unchanged)
# ---------------------------------------------------------------------------
IDX_MID_PRICE        = 0
IDX_LAST_PRICE       = 1
IDX_BEST_BID         = 2
IDX_BEST_ASK         = 3
IDX_SPREAD           = 4
IDX_PRICE_RETURN     = 5
IDX_BID_VOLUME_1     = 6
IDX_ASK_VOLUME_1     = 7
IDX_BID_VOLUME_2     = 8
IDX_ASK_VOLUME_2     = 9
IDX_XBID_VOLUME      = 10
IDX_PRODUCT          = 11
IDX_TIME_TO_DELIVERY = 12
IDX_VALUE_REF        = 13   # expected value of energy for the slot (€/MWh)
IDX_DAM_POSITION     = 14   # day-ahead schedule q_DAM,i (MWh, signed)
IDX_NET_POSITION     = 15   # committed net d_i (MWh, signed)
IDX_SOC              = 16   # current State of Charge (MWh, broadcast)
IDX_SOC_FRAC         = 17   # SoC / energy_capacity ∈ [0, 1]
IDX_REALIZED_PNL     = 18   # cumulative intraday cash flow for the slot (€)
IDX_DELTA_PCHARGE    = 19   # value_ref − ask   (charge attractiveness)
IDX_DELTA_PDISCHARGE = 20   # bid − value_ref   (discharge attractiveness)
IDX_GT_CHARGE        = 21   # feasible-charge shaping signal
IDX_GT_DISCHARGE     = 22   # feasible-discharge shaping signal

N_FEATURES = 23

FEATURE_NAMES = [
    "mid_price",          #  0
    "last_price",         #  1
    "best_bid",           #  2
    "best_ask",           #  3
    "spread",             #  4
    "price_return",       #  5
    "bid_volume_1",       #  6
    "ask_volume_1",       #  7
    "bid_volume_2",       #  8
    "ask_volume_2",       #  9
    "xbid_volume",        # 10
    "product",            # 11
    "time_to_delivery",   # 12
    "value_ref",          # 13  expected value of energy (€/MWh)
    "dam_position",       # 14  DAM schedule (MWh, + discharge / − charge)
    "net_position",       # 15  committed net (MWh)
    "soc",                # 16  State of Charge (MWh)
    "soc_frac",           # 17  SoC fraction [0,1]
    "realized_PnL",       # 18  intraday cash flow (€)
    "delta_pcharge",      # 19
    "delta_pdischarge",   # 20
    "gt_charge",          # 21
    "gt_discharge",       # 22
]


class RLObservationBuilder:
    """Builds the battery RL state matrix from market state + scenario."""

    def __init__(
        self,
        num_products: int = 96,
        fallback_price: float = 100.0,
        value_forecast_sigma: float = 8.0,
        battery: Optional[BatteryConfig] = None,
    ) -> None:
        self.num_products = num_products
        self.fallback_price = fallback_price
        self.value_forecast_sigma = value_forecast_sigma
        self.battery = battery or BatteryConfig()

        # ── Value-of-energy reference (noisy forecast + clean settlement) ──
        self._value_ref:      np.ndarray = np.full(num_products, fallback_price)
        self._real_value_ref: np.ndarray = np.full(num_products, fallback_price)

        # ── Battery position state ────────────────────────────────────────
        # DAM schedule (set once per day) and committed net position (evolves
        # with intraday fills).  Sign: + discharge/SELL, − charge/BUY.
        self._dam_position:  np.ndarray = np.zeros(num_products)
        self._net_position:  np.ndarray = np.zeros(num_products)

        # Cumulative intraday cash flow per slot (€) and total throughput (MWh)
        self._realized_pnl:  np.ndarray = np.zeros(num_products)
        self._throughput_mwh: float = 0.0

        self._xbid_volume:   np.ndarray = np.zeros(num_products)
        self._prev_mid:      np.ndarray = np.full(num_products, fallback_price)

        self._forecast_rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Signal update interface
    # ------------------------------------------------------------------

    def update_scenario(self, scenario, episode_seed: Optional[int] = None) -> None:
        """Ingest a scenario for a new episode.

        The scenario's ``imbalance_prices`` field is re-interpreted as the
        **value-of-energy reference** ``p_ref`` (the expected spot/settlement
        value used both for the arbitrage signal and for pricing any
        undeliverable imbalance).
        """
        ref = np.asarray(scenario.imbalance_prices, dtype=float)
        self._real_value_ref = ref.copy()
        self._value_ref      = ref.copy()

        # ── Deterministic per-episode RNG for the value-forecast noise ────
        if episode_seed is not None:
            seed = int(episode_seed) % (2**31)
        else:
            digest = hashlib.sha256(
                np.ascontiguousarray(self._real_value_ref.astype(np.float64)).tobytes()
            ).hexdigest()
            seed = int(digest[:8], 16)
        self._forecast_rng = np.random.default_rng(seed)

        if getattr(scenario, "xbid_volume", None) is not None:
            self._xbid_volume = np.asarray(scenario.xbid_volume, dtype=float)
        else:
            self._xbid_volume = np.zeros(self.num_products)

    def update_bm_forecast(self, current_slot: int) -> None:
        """Resample the value-of-energy forecast (time-decaying uncertainty).

        Near slots are forecast accurately; far slots carry up to
        ``value_forecast_sigma`` of Gaussian noise, following the same
        exponential decay used for the settlement-price forecast in the
        supplier model::

            noise_std_i = σ × (exp(k·d_i) − 1) / (exp(k) − 1),
            d_i = max(0, i − current_slot) / n,  k = 3
        """
        n = self.num_products
        k = 3.0
        noise = self._forecast_rng.standard_normal(n)
        distance = np.maximum(0, np.arange(n) - current_slot) / max(n, 1)
        noise_std = self.value_forecast_sigma * (np.expm1(k * distance) / np.expm1(k))
        self._value_ref = self._real_value_ref + noise_std * noise

    def update_xbid_volume(self, xbid_volume: np.ndarray) -> None:
        vol = np.asarray(xbid_volume, dtype=float)
        self._xbid_volume = np.nan_to_num(vol, nan=0.0)

    # ------------------------------------------------------------------
    # Position initialisation
    # ------------------------------------------------------------------

    def set_initial_position(
        self,
        dam_schedule: np.ndarray,
        realized_pnl: Optional[np.ndarray] = None,
    ) -> None:
        """Set the DAM schedule at the start of a trading day.

        The committed net position starts equal to the DAM schedule; intraday
        fills then move it away from (or back towards) that baseline.
        """
        self._dam_position = np.asarray(dam_schedule, dtype=float)
        self._net_position = self._dam_position.copy()
        self._realized_pnl = (
            np.asarray(realized_pnl, dtype=float)
            if realized_pnl is not None
            else np.zeros(self.num_products)
        )
        self._throughput_mwh = 0.0

    # ------------------------------------------------------------------
    # Trade recording — real cash flow + net position + throughput
    # ------------------------------------------------------------------

    def record_agent_trades(self, trades: List[Trade], signed_qtys: List[float]) -> None:
        for trade, sq in zip(trades, signed_qtys):
            self._record_trade(trade.product_id, sq, trade.price)

    def record_trade(self, product_id: int, signed_qty: float, price: float) -> None:
        self._record_trade(product_id, signed_qty, price)

    def _record_trade(self, pid: int, signed_qty: float, price: float) -> None:
        """Update cash flow, committed net position and throughput.

        ``signed_qty`` convention: ``+`` = SELL/discharge, ``−`` = BUY/charge.

            SELL (signed_qty > 0):  cash += qty·price   (money received)
            BUY  (signed_qty < 0):  cash += signed·price = −qty·price (paid)

        so a single line ``cash += signed_qty · price`` handles both.  There
        is **no** hedge-gating — every euro of realised spread counts.
        """
        self._realized_pnl[pid] += signed_qty * price
        self._net_position[pid] += signed_qty
        self._throughput_mwh    += abs(signed_qty)

    # ------------------------------------------------------------------
    # Core observation builder
    # ------------------------------------------------------------------

    def build(self, market_state: MarketState, slot: int) -> np.ndarray:
        """Build the full (num_products × N_FEATURES) state matrix."""
        obs = np.zeros((self.num_products, N_FEATURES), dtype=float)

        cap = max(self.battery.energy_mwh, 1e-9)
        # Current SoC = SoC after all slots that have already been delivered.
        soc_now = soc_after_slots(self._net_position, self.battery, int(slot))
        soc_frac = float(np.clip(soc_now / cap, 0.0, 1.0))
        charge_room    = max(self.battery.soc_max_mwh - soc_now, 0.0) / cap
        discharge_room = max(soc_now - self.battery.soc_min_mwh, 0.0) / cap

        ttd = market_state.time_to_delivery()
        ttd_max = float(np.max(ttd)) if len(ttd) and np.max(ttd) > 0 else 1.0

        for pid in range(self.num_products):
            ob = market_state.order_books[pid]

            best_bid = ob.best_bid
            best_ask = ob.best_ask
            last_px  = ob.last_trade_price
            ref      = market_state.reference_price(pid)

            mid = (
                (best_bid + best_ask) / 2.0
                if (best_bid is not None and best_ask is not None)
                else (last_px if last_px is not None else ref)
            )
            last = last_px if last_px is not None else mid

            obs[pid, IDX_MID_PRICE]  = mid
            obs[pid, IDX_LAST_PRICE] = last
            obs[pid, IDX_BEST_BID]   = best_bid if best_bid is not None else 0.0
            obs[pid, IDX_BEST_ASK]   = best_ask if best_ask is not None else 0.0
            obs[pid, IDX_SPREAD]     = (
                best_ask - best_bid
                if (best_bid is not None and best_ask is not None) else 0.0
            )

            prev = self._prev_mid[pid]
            obs[pid, IDX_PRICE_RETURN] = (mid - prev) / prev if prev != 0.0 else 0.0

            bid_depth, ask_depth = ob.get_book_depth()
            sorted_bids = sorted(bid_depth.items(), key=lambda x: x[0], reverse=True)
            sorted_asks = sorted(ask_depth.items(), key=lambda x: x[0])
            obs[pid, IDX_BID_VOLUME_1] = sorted_bids[0][1] if len(sorted_bids) > 0 else 0.0
            obs[pid, IDX_BID_VOLUME_2] = sorted_bids[1][1] if len(sorted_bids) > 1 else 0.0
            obs[pid, IDX_ASK_VOLUME_1] = sorted_asks[0][1] if len(sorted_asks) > 0 else 0.0
            obs[pid, IDX_ASK_VOLUME_2] = sorted_asks[1][1] if len(sorted_asks) > 1 else 0.0

            obs[pid, IDX_XBID_VOLUME] = (
                self._xbid_volume[pid] if pid < len(self._xbid_volume) else 0.0
            )
            obs[pid, IDX_PRODUCT]          = float(pid)
            obs[pid, IDX_TIME_TO_DELIVERY] = float(ttd[pid] / ttd_max) if pid < len(ttd) else 0.0

            value_ref = self._value_ref[pid]
            obs[pid, IDX_VALUE_REF]    = value_ref
            obs[pid, IDX_DAM_POSITION] = self._dam_position[pid]
            obs[pid, IDX_NET_POSITION] = self._net_position[pid]
            obs[pid, IDX_SOC]          = soc_now
            obs[pid, IDX_SOC_FRAC]     = soc_frac
            obs[pid, IDX_REALIZED_PNL] = self._realized_pnl[pid]

            bid_px = best_bid if best_bid is not None else 0.0
            ask_px = best_ask if best_ask is not None else 0.0
            delta_pcharge    = value_ref - ask_px   # buy cheap vs value → charge
            delta_pdischarge = bid_px - value_ref   # sell dear vs value → discharge
            obs[pid, IDX_DELTA_PCHARGE]    = delta_pcharge
            obs[pid, IDX_DELTA_PDISCHARGE] = delta_pdischarge

            # Shaping signals: profitable AND feasible (SoC headroom).
            obs[pid, IDX_GT_CHARGE]    = charge_room    * max(delta_pcharge,    0.0)
            obs[pid, IDX_GT_DISCHARGE] = discharge_room * max(delta_pdischarge, 0.0)

        self._prev_mid = obs[:, IDX_MID_PRICE].copy()
        return obs

    # ------------------------------------------------------------------
    # Day reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset intra-day state for a new trading day."""
        self._realized_pnl   = np.zeros(self.num_products)
        self._throughput_mwh = 0.0
        self._prev_mid       = np.full(self.num_products, self.fallback_price)
        self._dam_position   = np.zeros(self.num_products)
        self._net_position   = np.zeros(self.num_products)
        self._xbid_volume    = np.zeros(self.num_products)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def feature_vector(self, obs: np.ndarray, product_id: int) -> dict:
        return {name: float(obs[product_id, i]) for i, name in enumerate(FEATURE_NAMES)}

    @property
    def observation_shape(self) -> tuple:
        return (self.num_products, N_FEATURES)

    @property
    def flat_size(self) -> int:
        return self.num_products * N_FEATURES
