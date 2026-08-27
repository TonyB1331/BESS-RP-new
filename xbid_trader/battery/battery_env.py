"""Gymnasium environment for the BESS (battery) XBID intraday agent.

One episode = one trading day (up to 100 quarter-hour slots).  The battery
arrives with a *given* day-ahead schedule ``q_DAM`` and re-optimises intraday
on the XBID limit-order book, subject to State-of-Charge coupling across slots.

Action semantics (per product i) — (θ, δ), re-interpreted for arbitrage
----------------------------------------------------------------------
    γt,i = mid_t,i − p_ref_t,i         (market price minus value of energy)
    ct,i = +1  if γ >  θ   → SELL / discharge 1 MWh   (price is dear)
    ct,i = −1  if γ < −θ   → BUY  / charge    1 MWh   (price is cheap)
    ct,i =  0  if |γ| ≤ θ  → HOLD

    SELL price: best_ask − δ·spread     BUY price: best_bid + δ·spread

Feasibility
-----------
* Per-slot power cap ``|pos_i| ≤ P_max·Δt`` is enforced **hard** (a single
  product cannot carry more than the battery's power).
* SoC coupling is enforced **softly**: the agent may commit a schedule the SoC
  cannot fully deliver; the undeliverable part is settled at imbalance prices
  as a penalty (merchant convention — ~0 at the optimum).

Reward
------
    per step   : Δ(ID cash-flow) + λ_shaping·max(gt_sell+gt_buy, 0)
                 − λ_deg·throughput_step
    at terminal: − Σ_i |imbalance_i|·imbalance_price_i          (undeliverable)
                 − λ_term·|SoC_final − SoC_target|·price_scale  (cyclic target)

ID cash-flow of a fill = signed_pos·price  (SELL receives, BUY pays); every euro
counts — arbitrage is the objective, so there is no hedge-gating.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..types import Order, OrderSide, OrderType
from ..market.simulation_session import SimulationSession
from ..scenario.scenario_generator import ScenarioGenerator
from .battery_model import BatterySpec, dam_arbitrage_schedule
from .battery_observation import (
    BatteryObservationBuilder,
    IDX_MID_PRICE, IDX_BEST_BID, IDX_BEST_ASK, IDX_SPREAD,
    IDX_GT_SELL, IDX_GT_BUY, N_FEATURES,
)


FIXED_VOLUME_MWH: float = 1.0
ACTION_THETA_MAX: float = 10.0
ACTION_DELTA_MAX: float = 2.0
MAX_SLOTS: int = 100


class BatteryXBIDEnv(gym.Env):
    """Gymnasium environment wrapping the XBID market for a battery unit."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dam_prices_by_day: Dict,
        battery_spec: Optional[BatterySpec] = None,
        engine_config: Optional[Dict] = None,
        scenario_provider=None,
        dam_schedule_by_day: Optional[Dict] = None,
        lambda_shaping: float = 1.0,
        lambda_deg: float = 1.0,          # scale on degradation €/MWh throughput
        lambda_residual: float = 0.01,    # scale on imbalance settlement cost
        lambda_terminal: float = 1.0,     # scale on terminal-SoC deviation
        soc_target_by_day: Optional[Dict] = None,
        scenario_seed: Optional[int] = None,
        anchor_to_xbid: bool = True,
        forecast_seed: Optional[int] = 42,
        **legacy_kwargs,
    ) -> None:
        super().__init__()

        self.dam_prices_by_day   = dam_prices_by_day
        self.dam_days            = sorted(dam_prices_by_day.keys())
        self.spec                = battery_spec or BatterySpec()
        self._base_engine_config = engine_config or {}
        self._scenario_provider  = scenario_provider
        self._dam_schedule_by_day = dam_schedule_by_day or {}
        self._soc_target_by_day  = soc_target_by_day or {}
        self.lambda_shaping      = lambda_shaping
        self.lambda_deg          = lambda_deg
        self.lambda_residual     = lambda_residual
        self.lambda_terminal     = lambda_terminal
        self.scenario_seed       = scenario_seed
        self._anchor_to_xbid     = anchor_to_xbid
        self._forecast_seed      = forecast_seed

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MAX_SLOTS * N_FEATURES,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(MAX_SLOTS * 2, dtype=np.float32),
            high=np.array([ACTION_THETA_MAX, ACTION_DELTA_MAX] * MAX_SLOTS,
                          dtype=np.float32),
            dtype=np.float32,
        )

        self._session: Optional[SimulationSession]        = None
        self._obs_builder: Optional[BatteryObservationBuilder] = None
        self._current_obs: Optional[np.ndarray]           = None
        self._n_slots: int = 96
        self._prev_realized_pnl: Optional[np.ndarray]     = None
        self._episode_return: float = 0.0
        self._day_index: int = 0
        self._soc_target: float = self.spec.soc_target

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if self._day_index >= len(self.dam_days):
            self._day_index = 0

        day        = self.dam_days[self._day_index]
        dam_prices = np.asarray(self.dam_prices_by_day[day], dtype=float)
        n_slots    = len(dam_prices)
        self._n_slots = n_slots
        episode_idx = self._day_index
        self._day_index += 1

        if self._scenario_provider is not None:
            try:
                scenario = self._scenario_provider.get_scenario(day)
            except KeyError:
                scenario = self._make_synthetic_scenario(n_slots, dam_prices)
        else:
            scenario = self._make_synthetic_scenario(n_slots, dam_prices)

        ref_prices = self._select_reference_prices(scenario, dam_prices)

        engine_cfg = {
            **self._base_engine_config,
            "reference_prices": ref_prices,
            "num_products":     n_slots,
        }
        self._session = SimulationSession(
            engine_config=engine_cfg, custom_agent=None, print_summary=False,
        )

        # ── DAM schedule (given, else greedy arbitrage stand-in) ─────
        if day in self._dam_schedule_by_day:
            dam_pos = np.asarray(self._dam_schedule_by_day[day], dtype=float)
        elif getattr(scenario, "dam_schedule", None) is not None:
            dam_pos = np.asarray(scenario.dam_schedule, dtype=float)
        else:
            dam_pos = dam_arbitrage_schedule(dam_prices, self.spec)
        if len(dam_pos) != n_slots:
            tmp = np.zeros(n_slots); tmp[:min(n_slots, len(dam_pos))] = \
                dam_pos[:min(n_slots, len(dam_pos))]
            dam_pos = tmp

        self._soc_target = float(self._soc_target_by_day.get(day, self.spec.soc_target))

        self._obs_builder = BatteryObservationBuilder(
            spec=self.spec, num_products=n_slots,
            fallback_price=float(np.mean(ref_prices)),
        )
        ep_seed = (None if self._forecast_seed is None
                   else int(self._forecast_seed) * 100003 + int(episode_idx))
        self._obs_builder.update_scenario(
            scenario, dam_position=dam_pos, episode_seed=ep_seed,
        )
        if getattr(scenario, "xbid_volume", None) is not None:
            self._obs_builder.update_xbid_volume(scenario.xbid_volume)

        self._obs_builder.update_value_forecast(current_slot=0)
        self._current_obs = self._obs_builder.build(self._session._engine.state, slot=0)
        self._prev_realized_pnl = np.zeros(n_slots)
        self._episode_return = 0.0

        return self._flatten_obs(self._current_obs), {"day": str(day), "n_slots": n_slots}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._session is not None, "Call reset() before step()"

        slot = self._session.current_slot
        n    = self._n_slots

        result = self._session.step()
        state  = self._session._engine.state
        self._obs_builder.update_value_forecast(current_slot=slot)

        post_obs = self._obs_builder.build(state, slot=slot + 1)
        self._current_obs = post_obs

        mid    = post_obs[:, IDX_MID_PRICE]
        bid    = post_obs[:, IDX_BEST_BID]
        ask    = post_obs[:, IDX_BEST_ASK]
        spread = post_obs[:, IDX_SPREAD]

        action = np.asarray(action, dtype=np.float32).reshape(MAX_SLOTS, 2)
        theta  = action[:n, 0]
        delta  = action[:n, 1]

        # ── Arbitrage signal: γ = mid − p_ref ────────────────────────
        p_ref_fc = self._obs_builder._value_ref_fc[:n]
        gamma = mid[:n] - p_ref_fc

        ct = np.zeros(n, dtype=np.int8)
        ct[gamma >  theta] =  1   # dear → SELL / discharge
        ct[gamma < -theta] = -1   # cheap → BUY / charge

        # ── Hard per-slot power cap ──────────────────────────────────
        # Block trades that would push |pos_i| past the battery's power on a
        # single product (a physical limit, independent of SoC).
        cap = self.spec.per_slot_energy_cap
        committed = self._obs_builder.committed_position[:n]
        bad_sell = (ct == +1) & (committed + FIXED_VOLUME_MWH > cap + 1e-9)
        bad_buy  = (ct == -1) & (-committed + FIXED_VOLUME_MWH > cap + 1e-9)
        ct[bad_sell | bad_buy] = 0

        orders: List[Order] = []
        order_signs: List[float] = []
        current_time = state.current_time

        for pid in range(n):
            if not state.is_active(pid) or ct[pid] == 0:
                continue
            sp = max(float(spread[pid]), 1e-3)
            if ct[pid] == 1:                                    # SELL / discharge
                price = max(float(ask[pid]) - float(delta[pid]) * sp, 1e-3)
                side  = OrderSide.SELL
                signed_pos = +FIXED_VOLUME_MWH
            else:                                               # BUY / charge
                price = max(float(bid[pid]) + float(delta[pid]) * sp, 1e-3)
                side  = OrderSide.BUY
                signed_pos = -FIXED_VOLUME_MWH
            orders.append(Order(
                id=-1, product_id=pid, side=side,
                order_type=OrderType.LIMIT, price=price,
                quantity=FIXED_VOLUME_MWH, timestamp=current_time,
                trader_id="rl_agent",
            ))
            order_signs.append(signed_pos)

        agent_trades: List = []
        executed_signs: List[float] = []
        for order, sq in zip(orders, order_signs):
            trades = self._session._engine.add_external_order(order)
            for trade in trades:
                agent_trades.append(trade)
                executed_signs.append(sq)

        throughput_step = float(sum(abs(s) for s in executed_signs))

        if agent_trades:
            self._obs_builder.record_agent_trades(agent_trades, executed_signs)
            post_obs = self._obs_builder.build(state, slot=slot + 1)
            self._current_obs = post_obs

        terminated = result.is_day_complete
        reward = self._compute_reward(post_obs, throughput_step, terminated)
        self._episode_return += reward

        info = {
            "slot": slot,
            "n_agent_trades": len(agent_trades),
            "agent_trades": agent_trades,
            "agent_signs": list(executed_signs),
            "episode_return": self._episode_return if terminated else None,
        }

        if terminated and self._obs_builder is not None:
            ob = self._obs_builder
            deliverable, soc, imbalance = ob.deliverable_and_imbalance()
            imb_cost = float(np.sum(np.abs(imbalance) * np.abs(ob._real_imbalance_prices[:n])))
            cash = float(np.sum(ob._realized_pnl[:n]))
            deg  = self.spec.degradation_eur_per_mwh * float(np.sum(np.abs(deliverable)))
            soc_dev = abs(float(soc[-1]) - self._soc_target)
            info["ep_cash_flow"]     = cash
            info["ep_imbalance_mwh"] = float(np.sum(np.abs(imbalance)))
            info["ep_imbalance_cost"] = imb_cost
            info["ep_throughput_mwh"] = float(np.sum(np.abs(deliverable)))
            info["ep_degradation"]   = deg
            info["ep_soc_final"]     = float(soc[-1])
            info["ep_soc_dev"]       = soc_dev
            info["ep_cycles"]        = float(np.sum(np.abs(deliverable))) / (2.0 * max(self.spec.energy_mwh, 1e-9))
            info["ep_economic_pnl"]  = cash - imb_cost - deg
            info["ep_n_trades"]      = int(np.sum(np.abs(ob._fills[:n])))

        return self._flatten_obs(post_obs), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: np.ndarray, throughput_step: float,
                        terminated: bool) -> float:
        n = self._n_slots
        ob = self._obs_builder

        # Δ cash-flow this step
        realized = ob._realized_pnl[:n].copy()
        r_pnl = float(np.sum(realized - self._prev_realized_pnl))
        self._prev_realized_pnl = realized

        # Directional arbitrage shaping (clamped ≥ 0)
        gt_sell = obs[:n, IDX_GT_SELL]
        gt_buy  = obs[:n, IDX_GT_BUY]
        r_shape = float(np.sum(np.maximum(gt_sell + gt_buy, 0.0)))

        # Degradation on throughput this step
        r_deg = self.lambda_deg * self.spec.degradation_eur_per_mwh * throughput_step

        reward = r_pnl + self.lambda_shaping * r_shape - r_deg

        if terminated:
            deliverable, soc, imbalance = ob.deliverable_and_imbalance()
            imb_cost = float(np.sum(np.abs(imbalance) * np.abs(ob._real_imbalance_prices[:n])))
            price_scale = float(np.mean(np.abs(ob._real_imbalance_prices[:n]))) or 1.0
            soc_pen = abs(float(soc[-1]) - self._soc_target) * price_scale
            reward += -self.lambda_residual * imb_cost - self.lambda_terminal * soc_pen

        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_reference_prices(self, scenario, dam_prices: np.ndarray) -> np.ndarray:
        if not self._anchor_to_xbid:
            return dam_prices
        ip = getattr(scenario, "imbalance_prices", None)
        if ip is None:
            return dam_prices
        ref = np.asarray(ip, dtype=float).copy()
        if ref.shape != dam_prices.shape:
            return dam_prices
        bad = ~np.isfinite(ref) | (ref <= 0)
        if bad.any():
            ref[bad] = dam_prices[bad]
        return ref

    def _make_synthetic_scenario(self, n_slots: int, ref_prices: np.ndarray):
        warnings.warn(
            "BatteryXBIDEnv is falling back to a SYNTHETIC scenario (no "
            "historical provider or missing day). Synthetic days carry proxy "
            "prices only; for report-aligned results pass a "
            "HistoricalScenarioProvider.",
            RuntimeWarning, stacklevel=2,
        )
        gen = ScenarioGenerator(n_products=n_slots)
        return gen.generate_day_scenario(
            seed=self.scenario_seed, dam_reference_prices=ref_prices,
        )

    def _flatten_obs(self, obs: np.ndarray) -> np.ndarray:
        full = np.zeros((MAX_SLOTS, N_FEATURES), dtype=np.float32)
        full[:self._n_slots] = obs.astype(np.float32)
        return full.flatten()

    def render(self) -> None:
        pass
