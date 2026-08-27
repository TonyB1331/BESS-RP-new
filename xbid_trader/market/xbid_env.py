"""Gymnasium environment for the XBID intraday BATTERY (BESS) RL trader.

One episode = one full trading day (up to 100 quarter-hour products).

This is the *battery merchant* variant.  A supplier hedged an exogenous
residual imbalance ``Rt`` to zero; a battery instead performs **energy
arbitrage** over a State-of-Charge–coupled day, refining a day-ahead
schedule on XBID and paying imbalance only on the part of its committed net
position it cannot physically deliver.

Observation : Box(shape=(MAX_SLOTS × N_FEATURES,)) — flattened state matrix
Action      : Box(low=0, shape=(MAX_SLOTS × 2,)) — (θt,i, δt,i) per product

Action semantics (per product i) — re-interpreted for arbitrage
---------------------------------------------------------------
    γt,i  = value_ref_t,i  −  mid_t,i
    ct,i  = +1  if  γt,i  >  θt,i     → CHARGE  (BUY)  1 MWh   (energy is cheap)
    ct,i  = −1  if  γt,i  < −θt,i     → DISCHARGE (SELL) 1 MWh (energy is dear)
    ct,i  =  0  if  |γt,i| ≤  θt,i    → HOLD

    CHARGE   (BUY)  price: best_bid + δt,i × spread
    DISCHARGE(SELL) price: best_ask − δt,i × spread

θ is the learned *arbitrage band*; δ is price aggressiveness.

Feasibility
-----------
The only hard constraint enforced at order time is the **per-slot power
limit**: the committed net position of a slot may not exceed ``P_max·Δt`` in
magnitude.  The (temporally coupled) **energy / SoC limits** are enforced
economically: any committed net that cannot be delivered given the SoC path
is settled at imbalance / BM prices (see the reward), which the agent learns
to drive to zero.

Reward (per step)
-----------------
    r_t = ΔCashFlow_t                          (real intraday € received/paid)
        + λ_shaping × Σ max(gt_charge+gt_discharge, 0)   (arbitrage shaping)
        − degradation_€_per_MWh × Δthroughput_t          (cycling cost)

    at episode end additionally:
        − λ_residual × imbalance_settlement_cost(€)      (undeliverable net)
        − λ_terminal × |SoC_final − SoC_target| × price  (terminal SoC)
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

from ..types import Order, OrderSide, OrderType
from .simulation_session import SimulationSession
from .battery import (
    BatteryConfig, soc_dispatch, make_dam_arbitrage_schedule, cycle_usage,
)
from .rl_observation import (
    RLObservationBuilder,
    IDX_MID_PRICE,
    IDX_BEST_BID,
    IDX_BEST_ASK,
    IDX_SPREAD,
    IDX_GT_CHARGE,
    IDX_GT_DISCHARGE,
    N_FEATURES,
)
from ..scenario.scenario_generator import ScenarioGenerator


# ── Constants ──────────────────────────────────────────────────────────────
MIN_TRADE_MWH: float = 0.05    # ignore target changes smaller than this
ACTION_AGGR_MAX: float = 1.0   # how far to walk the book beyond the touch
MAX_SLOTS: int = 100   # DST-end days have 100 slots


class XBIDEnv(gym.Env):
    """Gymnasium environment wrapping the XBID market simulation for a battery."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dam_prices_by_day: Dict,
        engine_config: Optional[Dict] = None,
        scenario_provider=None,
        lambda_shaping: float = 0.0,   # 0 = pure economic reward (recommended for battery)
        lambda_pos: float = 0.5,          # fallback imbalance weight when no BM data
        lambda_residual: float = 0.01,    # scale on imbalance settlement cost
        scenario_seed: Optional[int] = None,
        dam_position_mwh: float = 10.0,   # kept for signature back-compat (unused)
        anchor_to_xbid: bool = True,
        forecast_seed: Optional[int] = 42,
        id_revision_sigma_frac: float = 0.0,   # ID price revision std as frac of |price|
        id_revision_rho: float = 0.8,          # AR(1) persistence of the revision
        decision_interval: int = 1,            # re-plan targets every K steps (K>1 kills churn)
        reveal_window: int = 24,               # quarters before gate over which a price reveals
        spike_prob: float = 0.0,               # per-product prob. of a RES up-spike
        neg_prob: float = 0.0,                 # per-product prob. of a negative-price event
        battery_config=None,              # dict | BatteryConfig | None
        lambda_degradation: Optional[float] = None,
        lambda_terminal: Optional[float] = None,
        dam_cycles: float = 1.0,          # size of the default DAM arbitrage schedule
        **legacy_kwargs,
    ) -> None:
        super().__init__()

        legacy_kwargs.pop("lambda_imb", None)

        self.dam_prices_by_day   = dam_prices_by_day
        self.dam_days            = sorted(dam_prices_by_day.keys())
        self._base_engine_config = engine_config or {}
        self._scenario_provider  = scenario_provider
        self.lambda_shaping      = lambda_shaping
        self.lambda_pos          = lambda_pos
        self.lambda_residual     = lambda_residual
        self.scenario_seed       = scenario_seed
        self._anchor_to_xbid     = anchor_to_xbid
        self._forecast_seed      = forecast_seed
        self._id_revision_sigma_frac = float(id_revision_sigma_frac)
        self._id_revision_rho        = float(id_revision_rho)
        self._decision_interval      = max(1, int(decision_interval))
        self._held_action            = None
        self._reveal_window          = max(1, int(reveal_window))
        self._spike_prob             = float(spike_prob)
        self._neg_prob               = float(neg_prob)
        self._product_side           = None    # per-product traded side (0/±1)
        self._dam_cycles         = dam_cycles

        # ── Battery configuration ─────────────────────────────────────────
        if isinstance(battery_config, BatteryConfig):
            self.battery = battery_config
        else:
            self.battery = BatteryConfig.from_mapping(battery_config)
        if lambda_degradation is not None:
            self.battery.lambda_degradation = float(lambda_degradation)
        if lambda_terminal is not None:
            self.battery.lambda_terminal = float(lambda_terminal)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MAX_SLOTS * N_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0] * MAX_SLOTS, dtype=np.float32),
            high=np.array([1.0, ACTION_AGGR_MAX] * MAX_SLOTS, dtype=np.float32),
            dtype=np.float32,
        )

        self._session:     Optional[SimulationSession]    = None
        self._obs_builder: Optional[RLObservationBuilder] = None
        self._current_obs: Optional[np.ndarray]           = None
        self._n_slots:     int   = 96
        self._prev_realized_pnl: Optional[np.ndarray]     = None
        self._prev_throughput:   float = 0.0
        self._episode_return: float = 0.0
        self._episode_economic: float = 0.0
        self._day_index:  int   = 0

        # Imbalance / BM dual prices for the current episode (or None).
        self._bm_up_price:   Optional[np.ndarray] = None
        self._bm_down_price: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if self._day_index >= len(self.dam_days):
            self._day_index = 0

        day        = self.dam_days[self._day_index]
        dam_prices = np.asarray(self.dam_prices_by_day[day], dtype=float)
        n_slots    = len(dam_prices)
        self._n_slots   = n_slots
        episode_idx     = self._day_index
        self._day_index += 1

        if self._scenario_provider is not None:
            try:
                scenario = self._scenario_provider.get_scenario(day)
            except KeyError:
                scenario = self._make_synthetic_scenario(n_slots, dam_prices)
        else:
            scenario = self._make_synthetic_scenario(n_slots, dam_prices)

        dam_ref = self._select_reference_prices(scenario, dam_prices)
        self._dam_ref = np.asarray(dam_ref, dtype=float).copy()

        # ── Intraday price revision, revealed GRADUALLY (like real XBID) ──
        # A final deviation ``rev_final`` (AR(1), std = frac·|price|) is drawn
        # per day but is UNKNOWN in advance. Each product's observable ID price
        # only reveals its deviation over a window of ``reveal_window`` quarters
        # before that product's gate: far products trade ≈ DAM, near-gate
        # products reveal their full deviation. The order book is re-anchored
        # every step (see step()), so the agent must react to deviations as
        # they appear and manage SoC under genuine uncertainty. The battery's
        # value reference stays the DAM price (its committed opportunity cost).
        if self._id_revision_sigma_frac > 0.0:
            rev_seed = (0 if self._forecast_seed is None
                        else int(self._forecast_seed) * 7919 + int(episode_idx))
            rev_rng = np.random.default_rng(rev_seed)
            self._rev_final  = self._apply_id_revision(dam_ref, rev_rng) - self._dam_ref
            self._reveal_rng = np.random.default_rng(rev_seed + 12345)
            # Realistic XBID microstructure: RES-driven UP spikes + negative
            # (or near-zero) price events, revealed near each product's gate.
            ev_rng = np.random.default_rng(rev_seed + 777)
            if self._spike_prob > 0.0:
                sp = ev_rng.random(len(dam_ref)) < self._spike_prob
                self._rev_final[sp] += ev_rng.uniform(0.5, 2.0, int(sp.sum())) * np.abs(self._dam_ref[sp])
            if self._neg_prob > 0.0:
                ng = ev_rng.random(len(dam_ref)) < self._neg_prob
                final = self._dam_ref + self._rev_final
                final[ng] = ev_rng.uniform(-30.0, 3.0, int(ng.sum()))
                self._rev_final = final - self._dam_ref
        else:
            self._rev_final  = np.zeros_like(self._dam_ref)
            self._reveal_rng = np.random.default_rng(0)
        # Final clearing curve (what each product settles at by its gate) — used
        # by the perfect-foresight oracle / imitation targets.
        self._id_reference = self._dam_ref + self._rev_final

        # The engine starts anchored to DAM: nothing is revealed at t=0.
        engine_ref = self._dam_ref.copy()

        engine_cfg = {
            "seed_books_on_init": False,   # so the book tracks the revealed ref
            **self._base_engine_config,
            "reference_prices": engine_ref,
            "num_products":     n_slots,
        }
        self._session = SimulationSession(
            engine_config=engine_cfg,
            custom_agent=None,
            print_summary=False,
        )

        self._obs_builder = RLObservationBuilder(
            num_products=n_slots,
            fallback_price=float(np.mean(self._dam_ref)),
            battery=self.battery,
        )
        ep_seed = (
            None if self._forecast_seed is None
            else int(self._forecast_seed) * 100003 + int(episode_idx)
        )
        self._obs_builder.update_scenario(scenario, episode_seed=ep_seed)

        if getattr(scenario, "xbid_volume", None) is not None:
            self._obs_builder.update_xbid_volume(scenario.xbid_volume)

        # ── Day-ahead schedule (merchant arbitrage baseline) ──────────────
        # For a battery the "given" DAM position is its own DAM-arbitrage
        # schedule.  A real schedule can be injected via
        # ``scenario.battery_dam_schedule``; otherwise a price-responsive one
        # is generated from the day's DAM prices.
        provided = getattr(scenario, "battery_dam_schedule", None)
        if provided is not None and len(provided) == n_slots:
            dam_schedule = np.asarray(provided, dtype=float)
        else:
            dam_schedule = make_dam_arbitrage_schedule(
                dam_prices, self.battery, cycles=self._dam_cycles,
            )
        self._obs_builder.set_initial_position(dam_schedule)

        # ── Imbalance / BM dual prices for undeliverable-net settlement ────
        self._bm_up_price = (
            np.asarray(scenario.bm_up_price, dtype=float)
            if getattr(scenario, "bm_up_price", None) is not None else None
        )
        self._bm_down_price = (
            np.asarray(scenario.bm_down_price, dtype=float)
            if getattr(scenario, "bm_down_price", None) is not None else None
        )

        self._obs_builder.update_bm_forecast(current_slot=0)
        self._current_obs = self._obs_builder.build(self._session._engine.state, slot=0)
        self._prev_realized_pnl = np.zeros(n_slots)
        self._prev_throughput   = 0.0
        self._episode_return    = 0.0
        self._episode_economic  = 0.0
        self._held_action       = None
        self._product_side      = np.zeros(n_slots, dtype=np.int8)

        return self._flatten_obs(self._current_obs), {"day": str(day), "n_slots": n_slots}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._session is not None, "Call reset() before step()"

        slot = self._session.current_slot
        n    = self._n_slots

        # ── Gradual price revelation ──────────────────────────────────────
        # Re-anchor the order book to the partially-revealed ID curve for this
        # step, BEFORE the background traders re-quote. Far-from-gate products
        # sit near DAM; each product reveals its deviation over the last
        # ``reveal_window`` quarters before its own gate.
        if self._id_revision_sigma_frac > 0.0:
            self._session._engine.set_reference_prices(self._revealed_reference(slot))

        result = self._session.step()
        state  = self._session._engine.state

        self._obs_builder.update_bm_forecast(current_slot=slot)

        post_obs = self._obs_builder.build(state, slot=slot + 1)
        self._current_obs = post_obs

        mid    = post_obs[:, IDX_MID_PRICE]
        bid    = post_obs[:, IDX_BEST_BID]
        ask    = post_obs[:, IDX_BEST_ASK]
        spread = post_obs[:, IDX_SPREAD]

        action = np.asarray(action, dtype=np.float32).reshape(MAX_SLOTS, 2)

        # ── Decision-frequency hold ───────────────────────────────────────
        # Re-plan the target only every ``decision_interval`` steps; between
        # re-plans the previous target is held. Because the dispatch is
        # idempotent, holding a stable target means the agent trades to reach
        # it and then STOPS — this eliminates the per-step target churn that
        # otherwise wash-trades away the spread.
        if self._decision_interval > 1:
            if self._held_action is None or (slot % self._decision_interval == 0):
                self._held_action = action
            else:
                action = self._held_action

        target_frac = np.clip(action[:n, 0], -1.0, 1.0)   # desired net as frac of power
        aggr        = np.clip(action[:n, 1], 0.0, ACTION_AGGR_MAX)

        # ── Direct-dispatch decode ────────────────────────────────────────
        # The policy names a TARGET net position per slot; the env trades to
        # move the current committed net toward it. This is idempotent (once
        # at target, no more orders → no over-trading) and can express any
        # feasible re-dispatch of the day (unlike a reactive threshold).
        net    = self._obs_builder._net_position[:n]
        p_slot = self.battery.power_per_slot_mwh
        target_net = target_frac * p_slot
        target_net[:slot] = net[:slot]          # gated slots are delivered, cannot retrade

        # ── Exact feasibility projection ──────────────────────────────────
        # Project the desired target onto the physically deliverable set
        # (SoC/power) with the SAME routine used at settlement, so the
        # committed position after this step is ALWAYS deliverable — the agent
        # can never create imbalance itself (imbalance = 0 by construction).
        # Then hard-cap the daily cycle budget: never let the physical schedule
        # exceed Cs cycles.
        _, feas_target, _ = soc_dispatch(target_net, self.battery)
        feas_target = self._cap_cycles(feas_target, net, slot)
        delta_net = feas_target - net                      # + → sell/discharge more

        orders:      List[Order]  = []
        order_signs: List[float]  = []
        current_time = state.current_time

        for pid in range(n):
            if not state.is_active(pid):
                continue
            d = float(delta_net[pid])
            if abs(d) < MIN_TRADE_MWH:
                continue

            # ── One-side-per-product rule ─────────────────────────────────
            # A product may be traded on only ONE side over the day: once the
            # agent has bought (charged) a product it may not sell it, and vice
            # versa. This forbids placing both bid and ask offers on the same
            # 15-min product and rules out wash trading.
            intended_side = 1 if d > 0 else -1     # +1 = SELL/discharge, −1 = BUY/charge
            if self._product_side[pid] != 0 and self._product_side[pid] != intended_side:
                continue

            sp = max(float(spread[pid]), 1e-3)

            if d > 0:                                          # SELL / discharge
                qty = d
                price      = max(float(bid[pid]) - aggr[pid] * sp, 1e-3)
                side       = OrderSide.SELL
                signed_qty = +qty
            else:                                              # BUY / charge
                qty = -d
                price      = max(float(ask[pid]) + aggr[pid] * sp, 1e-3)
                side       = OrderSide.BUY
                signed_qty = -qty

            orders.append(Order(
                id=-1, product_id=pid, side=side,
                order_type=OrderType.LIMIT, price=price,
                quantity=qty, timestamp=current_time,
                trader_id="rl_agent",
            ))
            order_signs.append(signed_qty)

        agent_trades:   List = []
        executed_signs: List[float] = []
        for order, sq in zip(orders, order_signs):
            trades = self._session._engine.add_external_order(order)
            for trade in trades:
                agent_trades.append(trade)
                executed_signs.append(sq)
                # Lock this product's side on its first executed intraday trade.
                self._product_side[int(trade.product_id)] = 1 if sq > 0 else -1

        if agent_trades:
            self._obs_builder.record_agent_trades(agent_trades, executed_signs)
            post_obs = self._obs_builder.build(state, slot=slot + 1)
            self._current_obs = post_obs

        # ── Action-aligned arbitrage edge from THIS step's fills ──────────
        # Reward each executed fill by how far it beat the true value of
        # energy (real DAM value): buying below value or selling above value
        # is good. Summed over a round-trip this ≈ 2× the arbitrage profit and,
        # unlike an observation-only bonus, it is zero when the agent does not
        # trade — so it actually incentivises profitable trading.
        edge_reward = 0.0
        for trade, sq in zip(agent_trades, executed_signs):
            vr = float(self._obs_builder._real_value_ref[int(trade.product_id)])
            edge = (trade.price - vr) if sq > 0 else (vr - trade.price)  # SELL vs BUY
            edge_reward += edge * float(trade.quantity)

        terminated = result.is_day_complete
        reward, economic = self._compute_reward(terminated, edge_reward)
        self._episode_return   += reward
        self._episode_economic += economic

        info = {
            "slot":           slot,
            "n_agent_trades": len(agent_trades),
            "agent_trades":   agent_trades,
            "agent_signs":    list(executed_signs),
            "episode_return": self._episode_return if terminated else None,
        }

        if terminated and self._obs_builder is not None:
            info.update(self._episode_metrics())

        return self._flatten_obs(post_obs), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, terminated: bool, edge_reward: float) -> Tuple[float, float]:
        """Return ``(reward, economic_pnl)`` for the current step.

        ``reward`` adds an action-aligned arbitrage-edge shaping term (for
        learning); ``economic_pnl`` is the pure € result (cash flow −
        degradation − imbalance − terminal − cycle) used for CVaR shaping and
        reporting.
        """
        n = self._n_slots
        ob = self._obs_builder

        # ── Real cash flow this step (no hedge-gating) ────────────────────
        realized = ob._realized_pnl[:n].copy()
        r_cash   = float(np.sum(realized - self._prev_realized_pnl))
        self._prev_realized_pnl = realized

        # ── Degradation on this step's throughput ─────────────────────────
        throughput = ob._throughput_mwh
        d_through  = max(throughput - self._prev_throughput, 0.0)
        self._prev_throughput = throughput
        r_deg = (self.battery.degradation_eur_per_mwh
                 * self.battery.lambda_degradation * d_through)

        # ── Terminal: imbalance settlement + terminal-SoC + cycle penalty ─
        r_imbalance = 0.0
        r_terminal  = 0.0
        r_cycle     = 0.0
        if terminated:
            imb_cost, _ = self._imbalance_cost()
            r_imbalance = self.lambda_residual * imb_cost
            r_terminal  = self._terminal_soc_penalty()
            r_cycle     = self._cycle_penalty()

        economic = r_cash - r_deg - r_imbalance - r_terminal - r_cycle
        reward   = economic + self.lambda_shaping * float(edge_reward)
        return reward, economic

    def _imbalance_cost(self) -> Tuple[float, float]:
        """€ cost and MWh of the undeliverable committed net position."""
        n = self._n_slots
        ob = self._obs_builder
        net = ob._net_position[:n]
        _, _, imbalance = soc_dispatch(net, self.battery)

        short = np.where(imbalance > 0, imbalance, 0.0)    # undeliverable discharge
        long_ = np.where(imbalance < 0, -imbalance, 0.0)   # unabsorbable charge

        if self._bm_up_price is not None and self._bm_down_price is not None:
            cost = float(np.sum(short * self._bm_up_price[:n])
                         + np.sum(long_ * self._bm_down_price[:n]))
        else:
            # No BM dual prices: imbalance is priced at a multiple of the
            # energy value (imbalance is worse than spot).
            price = self.battery.imbalance_penalty_mult * np.abs(ob._real_value_ref[:n])
            cost = self.lambda_pos * float(np.sum((short + long_) * price))
        return cost, float(np.sum(np.abs(imbalance)))

    def _cycle_penalty(self) -> float:
        """Penalise physical dispatch beyond the daily cycle limit Cs."""
        n = self._n_slots
        ob = self._obs_builder
        _, delivered, _ = soc_dispatch(ob._net_position[:n], self.battery)
        ch_cyc, dh_cyc = cycle_usage(delivered, self.battery)
        excess_cycles = (max(ch_cyc - self.battery.max_cycles_per_day, 0.0)
                         + max(dh_cyc - self.battery.max_cycles_per_day, 0.0))
        excess_mwh = excess_cycles * self.battery.energy_mwh
        price_scale = float(np.mean(np.abs(ob._real_value_ref[:n]))) or 1.0
        return self.battery.lambda_cycle * excess_mwh * price_scale

    def _terminal_soc_penalty(self) -> float:
        n = self._n_slots
        ob = self._obs_builder
        soc_path, _, _ = soc_dispatch(ob._net_position[:n], self.battery)
        soc_final = float(soc_path[-1]) if len(soc_path) else float(self.battery.soc_initial_mwh)
        deviation = abs(soc_final - self.battery.soc_target_mwh)
        price_scale = float(np.mean(np.abs(ob._real_value_ref[:n]))) or 1.0
        return self.battery.lambda_terminal * deviation * price_scale

    def _episode_metrics(self) -> Dict:
        n = self._n_slots
        ob = self._obs_builder
        soc_path, delivered, imbalance = soc_dispatch(ob._net_position[:n], self.battery)
        imb_cost, imb_mwh = self._imbalance_cost()
        soc_final = float(soc_path[-1]) if len(soc_path) else float(self.battery.soc_initial_mwh)
        ch_cyc, dh_cyc = cycle_usage(delivered, self.battery)
        return {
            "ep_realized_pnl":   float(np.sum(ob._realized_pnl[:n])),
            "ep_economic_pnl":   float(self._episode_economic),
            "ep_throughput_mwh": float(ob._throughput_mwh),
            "ep_imbalance_mwh":  float(imb_mwh),
            "ep_imbalance_cost": float(imb_cost),
            "ep_soc_final":      soc_final,
            "ep_soc_target":     float(self.battery.soc_target_mwh),
            "ep_terminal_penalty": float(self._terminal_soc_penalty()),
            "ep_charge_cycles":    ch_cyc,
            "ep_discharge_cycles": dh_cyc,
            "ep_cycle_penalty":    float(self._cycle_penalty()),
            # For CVaR downside shaping: the per-episode loss (−profit).
            "ep_loss":           float(-self._episode_economic),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cap_cycles(self, target: np.ndarray, net: np.ndarray, slot: int) -> np.ndarray:
        """Scale down tradeable-slot charge/discharge so daily cycles ≤ Cs.

        Gated slots (already delivered) are left untouched; only future slots
        are scaled, then the whole vector is re-projected to SoC feasibility.
        """
        Cs = self.battery.max_cycles_per_day
        t = np.array(target, dtype=float)
        idx = np.arange(len(t))
        for _ in range(4):
            ch, dh = cycle_usage(t, self.battery)
            if ch <= Cs + 1e-6 and dh <= Cs + 1e-6:
                break
            if ch > Cs + 1e-6:
                m = (idx >= slot) & (t < 0)
                if m.any():
                    t[m] *= max(Cs / ch, 0.0)
            if dh > Cs + 1e-6:
                m = (idx >= slot) & (t > 0)
                if m.any():
                    t[m] *= max(Cs / dh, 0.0)
            _, t, _ = soc_dispatch(t, self.battery)
        return t

    def _revealed_reference(self, t: int) -> np.ndarray:
        """Partially-revealed ID reference at decision step ``t``.

        Product ``i`` reveals its final deviation ``rev_final[i]`` over the last
        ``reveal_window`` quarters before its gate at slot ``i``::

            progress_i = clip((t - (i - W)) / W, 0, 1)
            ref_i(t)   = DAM_i + progress_i · rev_final[i] + bridge_noise_i(t)

        The bridge noise vanishes at progress 0 and 1, so far products sit at
        DAM and each product settles exactly at ``DAM_i + rev_final[i]`` by its
        gate. Between, the price is a noisy partial reveal — the agent cannot
        know in advance how it will move.
        """
        n = self._n_slots
        idx = np.arange(n)
        W = float(self._reveal_window)
        progress = np.clip((t - (idx - W)) / W, 0.0, 1.0)
        z = self._reveal_rng.standard_normal(n)
        noise = (self._id_revision_sigma_frac * np.abs(self._dam_ref[:n])
                 * np.sqrt(progress * (1.0 - progress)) * z)
        ref = self._dam_ref[:n] + progress * self._rev_final[:n] + noise
        return np.maximum(ref, 1e-3)

    def _apply_id_revision(self, dam_ref: np.ndarray, rng) -> np.ndarray:
        """Return an intraday reference = DAM reference + AR(1) revision.

        The revision has per-slot std ``id_revision_sigma_frac × |dam_ref|`` and
        AR(1) persistence ``id_revision_rho`` (so revisions are smooth across
        neighbouring quarters rather than white noise).
        """
        dam_ref = np.asarray(dam_ref, dtype=float)
        n = len(dam_ref)
        rho = self._id_revision_rho
        z = rng.standard_normal(n)
        rev = np.zeros(n)
        scale = np.sqrt(max(1.0 - rho * rho, 1e-6))
        for i in range(1, n):
            rev[i] = rho * rev[i - 1] + scale * z[i]
        rev = rev * self._id_revision_sigma_frac * np.abs(dam_ref)
        id_ref = dam_ref + rev
        return np.maximum(id_ref, 1e-3)

    def _select_reference_prices(self, scenario, dam_prices: np.ndarray) -> np.ndarray:
        if not self._anchor_to_xbid:
            return dam_prices
        if getattr(scenario, "imbalance_prices", None) is None:
            return dam_prices
        ref = np.asarray(scenario.imbalance_prices, dtype=float).copy()
        if ref.shape != dam_prices.shape:
            return dam_prices
        bad = ~np.isfinite(ref) | (ref <= 0)
        if bad.any():
            ref[bad] = dam_prices[bad]
        return ref

    def _make_synthetic_scenario(self, n_slots: int, ref_prices: np.ndarray):
        warnings.warn(
            "XBIDEnv is falling back to a SYNTHETIC scenario (no historical "
            "scenario provider, or the requested day is missing). Synthetic "
            "scenarios do NOT carry real value / BM dual prices, so the "
            "imbalance term degrades to the lambda_pos formulation. For "
            "report-aligned battery results, pass a HistoricalScenarioProvider "
            "with imbalance + BM data.",
            RuntimeWarning,
            stacklevel=2,
        )
        gen = ScenarioGenerator(n_products=n_slots)
        return gen.generate_day_scenario(
            seed=self.scenario_seed,
            dam_reference_prices=ref_prices,
        )

    def _flatten_obs(self, obs: np.ndarray) -> np.ndarray:
        full = np.zeros((MAX_SLOTS, N_FEATURES), dtype=np.float32)
        full[:self._n_slots] = obs.astype(np.float32)
        return full.flatten()

    def render(self) -> None:
        pass
