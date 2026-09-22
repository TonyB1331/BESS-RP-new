""" Battery (BESS) physical model for the XBID intraday RL trader.

This module replaces the supplier-specific *residual-imbalance* asset model
with a **merchant battery** model.  Where a supplier had an exogenous,
per-slot-independent hedging need ``Rt`` driven by demand-forecast revisions,
a battery is a *dispatchable* asset whose slots are **temporally coupled**
through the State of Charge (SoC): energy charged in one slot constrains the
energy that can be discharged in any later slot.

Sign convention (used everywhere in the battery code path)
----------------------------------------------------------
For a delivery slot ``i`` the *net physical position* ``d_i`` (MWh) is::

    d_i > 0   →  DISCHARGE / SELL   (battery injects energy into the grid)
    d_i < 0   →  CHARGE    / BUY    (battery withdraws energy from the grid)

The committed net position is ``d_i = q_DAM_i + Σ(intraday fills)_i`` where
``q_DAM_i`` is the day-ahead schedule and intraday fills are signed with the
same convention (a discharge/SELL fill is ``+qty``, a charge/BUY fill is
``-qty``).

SoC dynamics
------------
SoC is measured in MWh of *stored* energy.  With charge / discharge
efficiencies ``eta_c`` / ``eta_d``::

    discharge d_i > 0 :  storage draw   = d_i / eta_d
    charge    d_i < 0 :  storage gain   = |d_i| * eta_c

so that::

    SoC_i = SoC_{i-1} − (d_i)+ / eta_d + (−d_i)+ · eta_c

Feasibility (the coupling constraint)::

    SoC_min ≤ SoC_i ≤ SoC_max     ∀ i        (energy limits)
    |d_i|   ≤ P_max · Δt                       (power limit per slot)

Imbalance for a merchant battery
--------------------------------
The battery is a Balance Responsible Party: whatever committed net position
``d_i`` it *cannot physically deliver* (because doing so would violate the SoC
or power limits) is settled at Balancing-Market / imbalance prices.  A
well-behaved agent keeps this undeliverable part at ~0 — it never promises
beyond what the SoC allows — while still capturing the intraday arbitrage
spread.  :func:`soc_dispatch` computes exactly this split into *delivered* and
*undeliverable (imbalance)* energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


SLOT_DURATION_HOURS: float = 0.25   # 15-minute delivery products


@dataclass
class BatteryConfig:
    """Technical and economic parameters of the battery unit.

    Defaults describe the unit used in the DAM dataset (``1_5cycle...``):
    a **50 MW / 100 MWh** unit with a 20 % lower SoE floor, 95 %/95 %
    charge/discharge efficiency, a **1.5 cycles/day** limit and a cyclic
    daily SoE (start == end == 50 MWh).  Efficiency convention matches the
    day-ahead LP (``basic_constraints.py``)::

        e[h] = e[h-1] + (eta_charge * p_ch[h] - (1/eta_discharge) * p_dh[h]) * dt

    i.e. charging *stores* ``eta_charge * energy`` and discharging *draws*
    ``energy / eta_discharge`` from the store.
    """

    power_mw: float = 50.0           # Pc_max = Pd_max (MW)
    energy_mwh: float = 100.0        # Emax — usable energy capacity (MWh)
    eta_charge: float = 0.95         # N_c
    eta_discharge: float = 0.95      # N_d   (round-trip ≈ 0.9025)
    soc_min_mwh: float = 10.0        # Emin (MWh)
    soc_max_mwh: float = 100.0       # Emax (MWh)
    soc_initial_mwh: float = 50.0    # E_initial_running (MWh)
    soc_target_mwh: float = 50.0     # E_final_target (MWh)
    max_cycles_per_day: float = 1.5  # Cs — daily cycle limit
    degradation_eur_per_mwh: float = 2.0   # marginal degradation cost per MWh throughput
    slot_duration_hours: float = SLOT_DURATION_HOURS
    # Multiplier applied to the value-of-energy reference to price imbalance
    # when no explicit Balancing-Market dual prices are available.
    imbalance_penalty_mult: float = 1.5

    # ── Reward weights specific to the battery reward ────────────────────
    lambda_degradation: float = 1.0   # scale on degradation cost (€)
    lambda_terminal: float = 1.0      # scale on terminal-SoC deviation penalty
    lambda_cycle: float = 1.0         # scale on cycle-limit violation penalty

    @property
    def power_per_slot_mwh(self) -> float:
        """Maximum energy (MWh) that can flow in a single delivery slot."""
        return self.power_mw * self.slot_duration_hours

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_charge * self.eta_discharge

    @classmethod
    def from_mapping(cls, cfg: dict | None) -> "BatteryConfig":
        """Build a :class:`BatteryConfig` from a (possibly partial) mapping.

        Unknown keys are ignored; missing keys fall back to the defaults.
        A ``battery`` sub-mapping is unwrapped automatically so both
        ``{"battery": {...}}`` and a flat ``{...}`` are accepted.
        """
        if not cfg:
            return cls()
        if "battery" in cfg and isinstance(cfg["battery"], dict):
            cfg = cfg["battery"]
        valid = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        kwargs = {k: v for k, v in cfg.items() if k in valid}
        obj = cls(**kwargs)
        # If only energy_mwh was given, keep soc_max consistent unless the
        # caller explicitly overrode it.
        if "soc_max_mwh" not in kwargs and "energy_mwh" in kwargs:
            obj.soc_max_mwh = obj.energy_mwh
        return obj


def soc_dispatch(
    net_position: np.ndarray,
    cfg: BatteryConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a committed net-position vector onto SoC/power feasibility.

    Parameters
    ----------
    net_position:
        Committed net energy per delivery slot (MWh, sign convention above),
        in *delivery order* (index 0 delivers first).
    cfg:
        Battery configuration.

    Returns
    -------
    soc_path:
        SoC (MWh) *after* each slot's delivery — length ``len(net_position)``.
    delivered:
        Physically feasible net delivery per slot (MWh).  ``delivered[i]`` has
        the same sign as ``net_position[i]`` and ``|delivered[i]| ≤
        |net_position[i]|``.
    imbalance:
        Undeliverable part per slot ``= net_position − delivered`` (MWh).  This
        is what is settled at imbalance / BM prices.
    """
    n = len(net_position)
    soc = float(cfg.soc_initial_mwh)
    p_slot = cfg.power_per_slot_mwh
    lo, hi = float(cfg.soc_min_mwh), float(cfg.soc_max_mwh)
    eta_c, eta_d = float(cfg.eta_charge), float(cfg.eta_discharge)

    delivered = np.zeros(n, dtype=float)
    soc_path = np.zeros(n, dtype=float)

    for i in range(n):
        di = float(net_position[i])
        # Hard power limit — excess beyond P_max·Δt is physically impossible.
        di = float(np.clip(di, -p_slot, p_slot))

        if di >= 0.0:                       # DISCHARGE — draw di/eta_d from store
            draw = di / eta_d
            if soc - draw < lo:             # not enough stored energy
                draw = max(soc - lo, 0.0)
                di = draw * eta_d           # max feasible discharge
            soc -= draw
        else:                               # CHARGE — add |di|*eta_c to store
            add = (-di) * eta_c
            if soc + add > hi:              # not enough headroom
                add = max(hi - soc, 0.0)
                di = -(add / eta_c)         # max feasible charge (negative)
            soc += add

        delivered[i] = di
        soc_path[i] = soc

    imbalance = np.asarray(net_position, dtype=float) - delivered
    return soc_path, delivered, imbalance


def cycle_usage(delivered: np.ndarray, cfg: BatteryConfig) -> Tuple[float, float]:
    """Daily cycle usage implied by a *delivered* net-position vector.

    Mirrors the day-ahead LP cycle constraints::

        charge cycles    = Σ_{d<0}  eta_charge · |d|      / Emax   ≤ Cs
        discharge cycles = Σ_{d>0}  d / eta_discharge      / Emax   ≤ Cs

    Returns ``(charge_cycles, discharge_cycles)``.
    """
    d = np.asarray(delivered, dtype=float)
    cap = max(cfg.energy_mwh, 1e-9)
    charge_stored = cfg.eta_charge * np.sum(np.where(d < 0, -d, 0.0))
    discharge_drawn = np.sum(np.where(d > 0, d, 0.0)) / cfg.eta_discharge
    return float(charge_stored / cap), float(discharge_drawn / cap)


def soc_after_slots(net_position: np.ndarray, cfg: BatteryConfig, upto: int) -> float:
    """SoC (MWh) after delivering slots ``0 .. upto-1`` of ``net_position``.

    Used by the observation builder to report the *current* SoC at decision
    time ``upto`` (slots with index < ``upto`` have already been delivered).
    """
    if upto <= 0:
        return float(cfg.soc_initial_mwh)
    soc_path, _, _ = soc_dispatch(np.asarray(net_position, dtype=float)[:upto], cfg)
    return float(soc_path[-1]) if len(soc_path) else float(cfg.soc_initial_mwh)


def lp_optimal_dispatch(
    prices: np.ndarray,
    cfg: BatteryConfig,
    e_initial: Optional[float] = None,
    e_final: Optional[float] = None,
) -> np.ndarray:
    """Perfect-foresight optimal net dispatch (MWh/slot) for a price vector.

    Solves the same LP as the day-ahead formulation (``basic_constraints.py``)
    but on the given (intraday) price curve, and returns the optimal net
    position per slot with the module sign convention (+ discharge / − charge).
    Used to generate imitation-learning targets. Requires ``pulp``.
    """
    import pulp

    p = np.asarray(prices, dtype=float)
    H = len(p)
    dt = cfg.slot_duration_hours
    Emin, Emax = cfg.soc_min_mwh, cfg.soc_max_mwh
    Pc = Pd = cfg.power_mw
    Nc, Nd = cfg.eta_charge, cfg.eta_discharge
    Cs = cfg.max_cycles_per_day
    E0 = cfg.soc_initial_mwh if e_initial is None else e_initial
    Ef = cfg.soc_target_mwh if e_final is None else e_final

    m = pulp.LpProblem("battery_id", pulp.LpMaximize)
    e  = {h: pulp.LpVariable(f"e{h}",  Emin, Emax) for h in range(H)}
    pc = {h: pulp.LpVariable(f"pc{h}", 0, Pc)      for h in range(H)}
    pd = {h: pulp.LpVariable(f"pd{h}", 0, Pd)      for h in range(H)}
    m += pulp.lpSum(p[h] * (pd[h] - pc[h]) * dt for h in range(H))
    m += e[0] == E0 + (Nc * pc[0] - (1 / Nd) * pd[0]) * dt
    for h in range(1, H):
        m += e[h] == e[h - 1] + (Nc * pc[h] - (1 / Nd) * pd[h]) * dt
    m += pulp.lpSum(Nc * pc[h] * dt for h in range(H)) <= Cs * Emax
    m += pulp.lpSum((1 / Nd) * pd[h] * dt for h in range(H)) <= Cs * Emax
    m += e[H - 1] == Ef
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    return np.array([(pd[h].value() - pc[h].value()) * dt for h in range(H)])


def make_dam_arbitrage_schedule(
    prices: np.ndarray,
    cfg: BatteryConfig,
    cycles: float = 1.0,
) -> np.ndarray:
    """Build a sensible, SoC-feasible day-ahead schedule from DAM prices.

    This is a *default* merchant schedule used when no externally-provided
    DAM schedule is available: the battery charges in the cheapest slots and
    discharges in the most expensive ones, capped by power and total energy
    (≈ ``cycles`` full cycles per day), then made SoC-feasible.

    Positive entries are discharge/SELL commitments, negative are charge/BUY,
    consistent with the module sign convention.  Replace with a real given
    schedule by passing ``scenario.dam_position`` from your DAM optimiser.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    p_slot = cfg.power_per_slot_mwh
    # Energy budget for one direction (MWh) over the day.
    budget = float(cycles) * cfg.energy_mwh

    order = np.argsort(prices)                 # ascending price
    n_slots_dir = int(min(n // 2, max(1, round(budget / max(p_slot, 1e-9)))))
    charge_slots = order[:n_slots_dir]          # cheapest → charge
    discharge_slots = order[-n_slots_dir:]      # priciest → discharge

    sched = np.zeros(n, dtype=float)
    sched[charge_slots] = -p_slot               # charge (BUY)
    sched[discharge_slots] = +p_slot            # discharge (SELL)

    # Enforce SoC feasibility: clip against the SoC path (any infeasible part
    # is dropped from the schedule so the committed baseline is deliverable).
    _, delivered, _ = soc_dispatch(sched, cfg)
    return delivered
