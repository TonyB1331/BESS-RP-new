"""RL observation builder for the BESS (battery) XBID agent.

Reuses the 23-feature-per-slot layout of the supplier observation so that the
``SharedSlotEncoder`` policy and the meta-layer keep working unchanged, but
**redefines** the asset-specific features for a storage unit.

Feature layout (per slot, ``N_FEATURES = 23``)
----------------------------------------------
Indices 0–9 are identical asset-agnostic market microstructure features.
Indices 10–22 carry the battery state (replacing the supplier's Rt / DAM-load
features):

    0  mid_price          10  value_ref            20  gt_sell (discharge shaping)
    1  last_price         11  soc_frac             21  gt_buy  (charge shaping)
    2  best_bid           12  product              22  xbid_volume
    3  best_ask           13  dam_position (q_DAM, MWh, +discharge)
    4  spread             14  net_position (d = q_DAM + fills, MWh)
    5  price_return       15  realized_pnl (ID cash-flow, €)
    6  bid_volume_1       16  value_ref_forecast (noisy)
    7  ask_volume_1       17  imbalance_gap (committed − deliverable, MWh)
    8  bid_volume_2       18  delta_psell (bid − p_ref)
    9  ask_volume_2       19  delta_pbuy  (p_ref − ask)

The ``value_ref`` (a.k.a. water value) is the arbitrage anchor: the agent
discharges when the market rises θ above it and charges when it falls θ below.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from ..types import Trade
from ..market.market_state import MarketState
from .battery_model import BatterySpec, project_feasible, value_reference


# ── Feature indices ─────────────────────────────────────────────────────────
IDX_MID_PRICE          = 0
IDX_LAST_PRICE         = 1
IDX_BEST_BID           = 2
IDX_BEST_ASK           = 3
IDX_SPREAD             = 4
IDX_PRICE_RETURN       = 5
IDX_BID_VOLUME_1       = 6
IDX_ASK_VOLUME_1       = 7
IDX_BID_VOLUME_2       = 8
IDX_ASK_VOLUME_2       = 9
IDX_VALUE_REF          = 10
IDX_SOC_FRAC           = 11
IDX_PRODUCT            = 12
IDX_DAM_POSITION       = 13
IDX_NET_POSITION       = 14
IDX_REALIZED_PNL       = 15
IDX_VALUE_REF_FC       = 16
IDX_IMBALANCE_GAP      = 17
IDX_DELTA_PSELL        = 18
IDX_DELTA_PBUY         = 19
IDX_GT_SELL            = 20   # discharge shaping
IDX_GT_BUY             = 21   # charge shaping
IDX_XBID_VOLUME        = 22

N_FEATURES = 23

FEATURE_NAMES = [
    "mid_price", "last_price", "best_bid", "best_ask", "spread",
    "price_return", "bid_volume_1", "ask_volume_1", "bid_volume_2",
    "ask_volume_2", "value_ref", "soc_frac", "product", "dam_position",
    "net_position", "realized_pnl", "value_ref_forecast", "imbalance_gap",
    "delta_psell", "delta_pbuy", "gt_sell", "gt_buy", "xbid_volume",
]


class BatteryObservationBuilder:
    """Builds the RL state vector for the battery agent and tracks SoC/PnL."""

    def __init__(
        self,
        spec: BatterySpec,
        num_products: int = 96,
        fallback_price: float = 100.0,
        value_forecast_sigma: float = 8.0,
    ) -> None:
        self.spec = spec
        self.num_products = num_products
        self.fallback_price = fallback_price
        self.value_forecast_sigma = value_forecast_sigma

        # ── Prices / references ───────────────────────────────────────
        self._value_ref: np.ndarray          = np.full(num_products, fallback_price)
        self._value_ref_fc: np.ndarray       = np.full(num_products, fallback_price)
        self._imbalance_prices: np.ndarray   = np.full(num_products, fallback_price)
        self._real_imbalance_prices: np.ndarray = np.full(num_products, fallback_price)
        self._xbid_volume: np.ndarray        = np.zeros(num_products)
        self._prev_mid: np.ndarray           = np.full(num_products, fallback_price)

        # ── Battery positions (MWh, + = discharge/sell) ───────────────
        self._dam_position: np.ndarray       = np.zeros(num_products)
        self._fills: np.ndarray              = np.zeros(num_products)  # cumulative ID net

        # ── ID cash-flow PnL (€, cumulative per slot) ─────────────────
        self._realized_pnl: np.ndarray       = np.zeros(num_products)

        self._soc0: float = spec.soc_init

    # ------------------------------------------------------------------
    # Scenario ingestion
    # ------------------------------------------------------------------

    def update_scenario(
        self,
        scenario,
        dam_position: np.ndarray,
        soc0: Optional[float] = None,
        episode_seed: Optional[int] = None,
    ) -> None:
        """Ingest a day's price scenario and the (given) DAM schedule.

        ``dam_position`` is the battery's committed day-ahead net position
        (MWh, + discharge).  ``scenario.imbalance_prices`` supplies the
        settlement price used to cost any undeliverable position.
        """
        n = self.num_products
        prices = np.asarray(getattr(scenario, "prices", None), dtype=float)

        self._real_imbalance_prices = _fit(
            getattr(scenario, "imbalance_prices", prices), n, self.fallback_price
        )
        self._imbalance_prices = self._real_imbalance_prices.copy()

        # Arbitrage anchor (water value)
        self._value_ref = value_reference(prices) if prices.size else \
            np.full(n, self.fallback_price)
        self._value_ref = _fit(self._value_ref, n, self.fallback_price)
        self._value_ref_fc = self._value_ref.copy()

        self._dam_position = _fit(dam_position, n, 0.0)
        self._fills = np.zeros(n)
        self._realized_pnl = np.zeros(n)
        self._soc0 = self.spec.soc_init if soc0 is None else float(soc0)

        self._xbid_volume = _fit(
            getattr(scenario, "xbid_volume", np.zeros(n)), n, 0.0
        )
        self._xbid_volume = np.nan_to_num(self._xbid_volume, nan=0.0)

        # Deterministic per-episode RNG for the value-forecast noise.
        if episode_seed is not None:
            seed = int(episode_seed) % (2**31)
        else:
            digest = hashlib.sha256(
                np.ascontiguousarray(prices.astype(np.float64)).tobytes()
            ).hexdigest()
            seed = int(digest[:8], 16)
        self._forecast_rng = np.random.default_rng(seed)

    def update_value_forecast(self, current_slot: int) -> None:
        """Resample the value-reference forecast (exponential time-decay noise:
        near slots accurate, far slots uncertain)."""
        n = self.num_products
        k = 3.0
        noise = self._forecast_rng.standard_normal(n)
        distance = np.maximum(0, np.arange(n) - current_slot) / n
        noise_std = self.value_forecast_sigma * (np.expm1(k * distance) / np.expm1(k))
        self._value_ref_fc = self._value_ref + noise_std * noise

    def update_xbid_volume(self, xbid_volume: np.ndarray) -> None:
        self._xbid_volume = np.nan_to_num(
            _fit(xbid_volume, self.num_products, 0.0), nan=0.0
        )

    # ------------------------------------------------------------------
    # Trade recording — cash-flow PnL (NO hedge-gating)
    # ------------------------------------------------------------------

    def record_agent_trades(self, trades: List[Trade], signed_qtys: List[float]) -> None:
        for trade, sq in zip(trades, signed_qtys):
            self._record_trade(trade.product_id, sq, trade.price)

    def _record_trade(self, pid: int, signed_qty: float, price: float) -> None:
        """Update fills and realised cash-flow for one fill.

        ``signed_qty`` is in position terms: + = SELL/discharge, − = BUY/charge.
        Cash-flow = signed_qty × price (sell receives, buy pays).  Every euro
        of the fill counts — arbitrage is the objective, so there is no
        hedge-gating.
        """
        self._realized_pnl[pid] += signed_qty * price
        self._fills[pid] += signed_qty

    # ------------------------------------------------------------------
    # Derived battery state
    # ------------------------------------------------------------------

    @property
    def committed_position(self) -> np.ndarray:
        """Committed net position per slot (DAM + intraday fills)."""
        return self._dam_position + self._fills

    def deliverable_and_imbalance(self):
        """Return (deliverable, soc_path, imbalance) for the committed schedule."""
        return project_feasible(self.committed_position, self.spec, self._soc0)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def build(self, market_state: MarketState, slot: int) -> np.ndarray:
        obs = np.zeros((self.num_products, N_FEATURES), dtype=float)

        committed = self.committed_position
        deliverable, soc, imbalance = project_feasible(
            committed, self.spec, self._soc0
        )
        e_max = max(self.spec.energy_mwh, 1e-9)

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
            sb = sorted(bid_depth.items(), key=lambda x: x[0], reverse=True)
            sa = sorted(ask_depth.items(), key=lambda x: x[0])
            obs[pid, IDX_BID_VOLUME_1] = sb[0][1] if len(sb) > 0 else 0.0
            obs[pid, IDX_BID_VOLUME_2] = sb[1][1] if len(sb) > 1 else 0.0
            obs[pid, IDX_ASK_VOLUME_1] = sa[0][1] if len(sa) > 0 else 0.0
            obs[pid, IDX_ASK_VOLUME_2] = sa[1][1] if len(sa) > 1 else 0.0

            p_ref    = self._value_ref[pid]
            p_ref_fc = self._value_ref_fc[pid]

            obs[pid, IDX_VALUE_REF]    = p_ref
            obs[pid, IDX_VALUE_REF_FC] = p_ref_fc
            obs[pid, IDX_SOC_FRAC]     = soc[pid] / e_max
            obs[pid, IDX_PRODUCT]      = float(pid)
            obs[pid, IDX_DAM_POSITION] = self._dam_position[pid]
            obs[pid, IDX_NET_POSITION] = committed[pid]
            obs[pid, IDX_REALIZED_PNL] = self._realized_pnl[pid]
            obs[pid, IDX_IMBALANCE_GAP] = imbalance[pid]

            bid_px = best_bid if best_bid is not None else 0.0
            ask_px = best_ask if best_ask is not None else 0.0
            delta_psell = bid_px - p_ref     # margin from discharging (selling) now
            delta_pbuy  = p_ref - ask_px     # margin from charging (buying) now
            obs[pid, IDX_DELTA_PSELL] = delta_psell
            obs[pid, IDX_DELTA_PBUY]  = delta_pbuy

            # Shaping: price-margin × available room, with room expressed as a
            # FRACTION of usable capacity so the term stays bounded (≈ margin in
            # magnitude) and decays to zero as the battery saturates — no reward
            # for "charging" when full or "discharging" when empty.
            discharge_room = max(soc[pid] - self.spec.soc_min, 0.0) / e_max
            charge_room    = max(self.spec.soc_max - soc[pid], 0.0) / e_max
            obs[pid, IDX_GT_SELL] = discharge_room * delta_psell
            obs[pid, IDX_GT_BUY]  = charge_room * delta_pbuy

            obs[pid, IDX_XBID_VOLUME] = self._xbid_volume[pid]

        self._prev_mid = obs[:, IDX_MID_PRICE].copy()
        return obs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def feature_vector(self, obs: np.ndarray, product_id: int) -> dict:
        return {n: float(obs[product_id, i]) for i, n in enumerate(FEATURE_NAMES)}

    @property
    def observation_shape(self) -> tuple:
        return (self.num_products, N_FEATURES)

    @property
    def flat_size(self) -> int:
        return self.num_products * N_FEATURES


def _fit(arr, n: int, fill: float) -> np.ndarray:
    """Pad/truncate ``arr`` to length ``n`` filling short arrays with ``fill``."""
    a = np.asarray(arr, dtype=float).ravel()
    if len(a) >= n:
        return a[:n].copy()
    out = np.full(n, fill, dtype=float)
    out[:len(a)] = a
    return out
