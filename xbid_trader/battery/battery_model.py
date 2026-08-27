"""Physical model of a grid-scale battery energy storage system (BESS).

This module replaces the *supplier* asset model (DAM load position + exogenous
``Rt`` residual-imbalance from ISP forecast revisions) with the correct
physics of a **merchant storage unit** that participates in DAM and re-optimises
intraday on XBID.

Key differences from the supplier model
---------------------------------------
* The battery is **dispatchable**: its position is a *choice*, not an
  involuntary demand-forecast error.  There is therefore **no exogenous
  ``Rt``** and no ISP1/2/3 machinery.
* Delivery slots are **temporally coupled through State of Charge (SoC)**:
  energy charged in one slot is only available to discharge in a later slot.
  (The supplier model treated slots as independent — correct for load, wrong
  for storage.)
* "Imbalance" for a battery is the part of its committed (DAM + intraday) net
  position that is **not physically deliverable** given the SoC trajectory.
  A well-behaved agent drives it to ~0; it is settled at imbalance/BM prices
  as a penalty (the merchant convention the pipeline is configured for).

Sign convention (used everywhere in the battery layer)
------------------------------------------------------
For every quarter-hour delivery slot ``i`` the *position* ``pos_i`` is the net
energy the battery has **sold / discharged** to the grid, in MWh:

    pos_i > 0   → net DISCHARGE  (battery injects energy, drains storage)
    pos_i < 0   → net CHARGE     (battery withdraws energy, fills storage)

A market **SELL** increases ``pos`` (discharge); a market **BUY** decreases it
(charge).  This matches ``OrderSide`` at the exchange:
    SELL 1 MWh → pos += 1 ;  BUY 1 MWh → pos -= 1.

SoC dynamics (energy stored, MWh)
---------------------------------
    discharging pos_i > 0 MWh to the grid draws  pos_i / eta_d  from storage
    charging   |pos_i| MWh from the grid stores |pos_i| * eta_c  into storage

so, over slots in delivery-time order:

    SoC_i = SoC_{i-1} - (pos_i)_+ / eta_d + (-pos_i)_+ * eta_c
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BatterySpec:
    """Technical parameters of the battery unit.

    Defaults describe a 10 MW / 20 MWh (2-hour) grid battery with a 90 %
    round-trip efficiency, split symmetrically into one-way efficiencies
    ``eta_c = eta_d = sqrt(0.90) ≈ 0.9487``.

    Parameters
    ----------
    power_mw:
        Maximum charge/discharge power (MW).  Caps the energy tradable on a
        single 15-minute product at ``power_mw * slot_hours`` MWh.
    energy_mwh:
        Usable energy capacity (MWh) between ``soc_min_frac`` and
        ``soc_max_frac``.
    eta_charge, eta_discharge:
        One-way charge / discharge efficiencies in (0, 1].  Round-trip
        efficiency is ``eta_charge * eta_discharge``.
    soc_min_frac, soc_max_frac:
        Fractional SoC operating band (of ``energy_mwh``).  Defaults span the
        full 0–100 % usable window.
    soc_init_frac:
        SoC at the start of the trading day (fraction of ``energy_mwh``).
    soc_target_frac:
        Desired SoC at the end of the day (fraction).  Deviations are penalised
        in the reward; can be overridden per-day (e.g. from a 3-day look-ahead).
    degradation_eur_per_mwh:
        Marginal degradation cost per MWh of energy throughput (charge or
        discharge), €/MWh.
    slot_hours:
        Delivery-product duration in hours (0.25 for quarter-hour products).
    """

    power_mw: float = 10.0
    energy_mwh: float = 20.0
    eta_charge: float = 0.9487      # sqrt(0.90)
    eta_discharge: float = 0.9487   # sqrt(0.90)
    soc_min_frac: float = 0.0
    soc_max_frac: float = 1.0
    soc_init_frac: float = 0.5
    soc_target_frac: float = 0.5
    degradation_eur_per_mwh: float = 2.0
    slot_hours: float = 0.25

    # ── Derived quantities (MWh) ─────────────────────────────────────
    @property
    def soc_min(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_init(self) -> float:
        return self.soc_init_frac * self.energy_mwh

    @property
    def soc_target(self) -> float:
        return self.soc_target_frac * self.energy_mwh

    @property
    def per_slot_energy_cap(self) -> float:
        """Max energy (MWh) tradable on one 15-minute product."""
        return self.power_mw * self.slot_hours

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_charge * self.eta_discharge

    @classmethod
    def from_config(cls, cfg: dict) -> "BatterySpec":
        """Build a spec from a plain dict (e.g. a parsed YAML ``battery`` block).
        Unknown keys are ignored so configs can carry extra metadata."""
        fields = {
            "power_mw", "energy_mwh", "eta_charge", "eta_discharge",
            "soc_min_frac", "soc_max_frac", "soc_init_frac", "soc_target_frac",
            "degradation_eur_per_mwh", "slot_hours",
        }
        return cls(**{k: cfg[k] for k in fields if k in cfg})


# ---------------------------------------------------------------------------
# SoC trajectory helpers
# ---------------------------------------------------------------------------

def soc_delta(pos_i: float, spec: BatterySpec) -> float:
    """Change in stored energy (MWh) caused by a net slot position ``pos_i``.

    Discharge (pos>0) removes ``pos/eta_d``; charge (pos<0) adds ``|pos|*eta_c``.
    """
    if pos_i >= 0.0:
        return -pos_i / spec.eta_discharge
    return (-pos_i) * spec.eta_charge


def soc_path(pos: np.ndarray, spec: BatterySpec, soc0: float) -> np.ndarray:
    """Un-clipped SoC after each slot given a committed position vector.

    May leave the ``[soc_min, soc_max]`` band — that excursion is exactly what
    :func:`project_feasible` detects and settles as imbalance.
    """
    pos = np.asarray(pos, dtype=float)
    soc = np.empty(len(pos), dtype=float)
    s = float(soc0)
    for i, p in enumerate(pos):
        s = s + soc_delta(float(p), spec)
        soc[i] = s
    return soc


def project_feasible(
    pos: np.ndarray,
    spec: BatterySpec,
    soc0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causally clip a committed position vector to what the battery can deliver.

    Walking forward in delivery-time order, each slot's discharge is limited by
    the energy currently in storage (down to ``soc_min``) and each slot's charge
    by the remaining headroom (up to ``soc_max``).  Whatever cannot be delivered
    is returned as ``imbalance`` (same sign convention as ``pos``).

    Returns
    -------
    deliverable : np.ndarray
        The physically realisable position per slot (MWh, signed).
    soc         : np.ndarray
        SoC after each slot along the *deliverable* trajectory (MWh).
    imbalance   : np.ndarray
        ``pos - deliverable`` — the undeliverable MWh per slot (the merchant
        "imbalance", settled at imbalance prices).  ~0 for a feasible schedule.
    """
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    deliverable = np.zeros(n, dtype=float)
    soc = np.empty(n, dtype=float)
    s = float(soc0)

    for i in range(n):
        p = float(pos[i])
        if p >= 0.0:  # discharge: limited by available energy above soc_min
            max_out_storage = max(s - spec.soc_min, 0.0)          # MWh from storage
            max_out_grid = max_out_storage * spec.eta_discharge   # MWh to grid
            d = min(p, max_out_grid)
            s -= d / spec.eta_discharge
        else:         # charge: limited by headroom below soc_max
            want = -p
            max_in_storage = max(spec.soc_max - s, 0.0)           # MWh into storage
            max_in_grid = max_in_storage / spec.eta_charge        # MWh from grid
            c = min(want, max_in_grid)
            d = -c
            s += c * spec.eta_charge
        deliverable[i] = d
        soc[i] = s

    imbalance = pos - deliverable
    return deliverable, soc, imbalance


def is_feasible(pos: np.ndarray, spec: BatterySpec, soc0: float, tol: float = 1e-6) -> bool:
    """True iff the committed schedule needs no imbalance (fully deliverable)
    and respects the per-slot power cap."""
    if np.any(np.abs(np.asarray(pos, dtype=float)) > spec.per_slot_energy_cap + tol):
        return False
    _, _, imb = project_feasible(pos, spec, soc0)
    return bool(np.all(np.abs(imb) <= tol))


# ---------------------------------------------------------------------------
# Value reference ("water value") for the arbitrage signal
# ---------------------------------------------------------------------------

def value_reference(prices: np.ndarray) -> np.ndarray:
    """Marginal value of stored energy per slot (€/MWh) — the arbitrage anchor.

    Used as ``p_ref`` in the (θ, δ) action: the agent discharges when the
    market price rises θ above ``p_ref`` and charges when it falls θ below.
    A simple, stable "water value" is the day's median price, broadcast to all
    slots; the stochastic forecast layer adds slot-dependent noise on top.
    """
    prices = np.asarray(prices, dtype=float)
    ref = float(np.nanmedian(prices)) if prices.size else 100.0
    return np.full(len(prices), ref, dtype=float)


# ---------------------------------------------------------------------------
# Day-ahead arbitrage scheduler  (the "given" DAM position)
# ---------------------------------------------------------------------------

def dam_arbitrage_schedule(
    prices: np.ndarray,
    spec: BatterySpec,
    soc0: float | None = None,
    max_cycles: float | None = None,
) -> np.ndarray:
    """Construct a feasible day-ahead arbitrage schedule from DAM prices.

    This stands in for an external DAM optimiser: it is the battery's *own*
    day-ahead plan (charge cheap slots, discharge expensive slots) that the RL
    agent then re-optimises intraday.  Replace with a real schedule by passing
    ``dam_schedule_by_day`` to the environment.

    Method — a causal, always-feasible greedy: allocate one 1-MWh (grid-side)
    charge to the cheapest slot and one discharge to the most expensive slot
    whose combined spread covers the round-trip loss, then project onto SoC
    feasibility.  Repeats until no profitable, feasible unit remains or the
    energy budget is exhausted.

    Returns a position vector (MWh, signed: + discharge, − charge).
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    soc0 = spec.soc_init if soc0 is None else float(soc0)
    cap = spec.per_slot_energy_cap
    usable = spec.soc_max - spec.soc_min
    if max_cycles is None:
        max_cycles = 1.0  # at most one full charge/discharge cycle in DAM
    energy_budget = usable * max_cycles

    pos = np.zeros(n, dtype=float)
    charged_grid = 0.0     # cumulative grid-side charge energy committed

    # Order slots for charging (cheap first) and discharging (dear first).
    charge_order = list(np.argsort(prices))
    discharge_order = list(np.argsort(-prices))

    step = min(1.0, cap)  # 1-MWh grid-side increments (or the power cap)
    while charged_grid < energy_budget:
        # Cheapest slot with charge headroom (grid-side power cap).
        ci = next((i for i in charge_order if -pos[i] + step <= cap + 1e-9), None)
        # Dearest slot with discharge headroom.
        dj = next((j for j in discharge_order if pos[j] + step <= cap + 1e-9), None)
        if ci is None or dj is None:
            break
        # Profitability after round-trip loss: sell η_rt·step at price[dj],
        # buy step at price[ci].
        gross = prices[dj] * spec.round_trip_efficiency * step - prices[ci] * step
        if gross <= spec.degradation_eur_per_mwh * step * 2.0:
            break  # no more profitable cycles beyond degradation cost
        trial = pos.copy()
        trial[ci] -= step   # charge
        trial[dj] += step    # discharge
        # Keep only if the whole schedule stays SoC-feasible.
        if is_feasible(trial, spec, soc0):
            pos = trial
            charged_grid += step
        else:
            # Remove the exhausted endpoints from consideration to avoid loops.
            if pos[dj] + step > cap:
                discharge_order.remove(dj)
            else:
                # infeasible ordering (discharge before enough charge) — drop dj
                discharge_order.remove(dj)
        if not charge_order or not discharge_order:
            break

    return pos
