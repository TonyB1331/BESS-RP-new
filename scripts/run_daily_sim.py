#!/usr/bin/env python3
"""Daily D-1 → D simulation runner.

Feed it ONE day-ahead schedule file (the `BESS_Optimized_Schedule_*.xlsx` format:
committed DAM positions, physical dispatch, SoE and reserve awards for tomorrow),
and it plays day D forward against a Monte-Carlo XBID market calibrated to real
Greek XBID, running the intraday controller over many scenarios. Output: the
expected daily uplift, its risk band (P10–P90), and a representative order
blotter — the live shape, one day at a time.

    python run_daily_sim.py --schedule BESS_Optimized_Schedule_05_08.xlsx \
        --power-mw 1.5 --energy-mwh 2.0 --scenarios 200

When the real XBID prints for that day later become available, the same day can
be re-run in backtest mode to check the real result lands inside this band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "xbid_trader" / "market"))

from xbid_trader.data.real_market_loader import RealDay, SLOT_HOURS      # noqa: E402
from xbid_trader.market.battery import BatteryConfig                     # noqa: E402
import xbid_simulator as sim                                             # noqa: E402
import operational_backtest_real as obr                                  # noqa: E402


# ==========================================================================
# Loader for the daily D-1 schedule format
# ==========================================================================
def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(float)


def load_daily_schedule(path, activation_rate: float = 0.20,
                        eta_charge: float = 0.95, eta_discharge: float = 0.95):
    """Parse one `BESS_Optimized_Schedule_*.xlsx` day into a RealDay (no XBID yet)."""
    df = pd.read_excel(path, sheet_name="BESS_Schedule")
    dt = pd.to_datetime(df["Delivery_Interval"], errors="coerce")
    df = df[dt.notna()].copy()
    day = str(pd.to_datetime(df["Delivery_Interval"].iloc[0]).date())

    dam_price = _num(df["DAM_Price_EUR_MWh"])
    physical = (_num(df["Physical_Discharge_MW"]) - _num(df["Physical_Charge_MW"])) * SLOT_HOURS
    settled = (_num(df["DAM_Discharge_MW"]) - _num(df["DAM_Charge_MW"])) * SLOT_HOURS

    # reserves: prefer the pre-summed totals, else sum the products
    if "Total_Capacity_Up_MW" in df.columns:
        up = _num(df["Total_Capacity_Up_MW"])
        dn = _num(df["Total_Capacity_Dn_MW"])
    else:
        up = sum(_num(df[c]) for c in ["FCR_Up_Capacity_MW", "aFRR_Up_Capacity_MW",
                                       "mFRR_Up_Capacity_MW"] if c in df.columns)
        dn = sum(_num(df[c]) for c in ["FCR_Down_Capacity_MW", "aFRR_Down_Capacity_MW",
                                       "mFRR_Down_Capacity_MW"] if c in df.columns)

    activation = activation_rate * (up - dn) * SLOT_HOURS

    plan_soc = _num(df["SOE_Capacity_MWh"]) if "SOE_Capacity_MWh" in df.columns else None
    soc_start = None
    if plan_soc is not None:
        q0 = float(physical[0])
        soc_start = float(plan_soc[0]) - (eta_charge * max(-q0, 0.0)
                                          - max(q0, 0.0) / eta_discharge)

    # committed capacity revenue for the day (from the summary sheet, if present)
    reserve_revenue = 0.0
    try:
        fs = pd.read_excel(path, sheet_name="Daily_Financial_Summary")
        cols = [c for c in fs.columns if "Profit" in c and "DAM" not in c and "Total" not in c]
        reserve_revenue = float(sum(pd.to_numeric(fs[c], errors="coerce").fillna(0).iloc[0]
                                    for c in cols))
    except Exception:
        pass

    rd = RealDay(
        dam_price=dam_price,
        xbid_price=np.full(len(dam_price), np.nan),   # filled per scenario
        xbid_spread=np.zeros(len(dam_price)),
        imb_price=dam_price.copy(),                   # proxy until real prints exist
        dam_net=physical,                             # tradeable = physical operation
        tradeable=np.zeros(len(dam_price), dtype=bool),
        up_mw=up, dn_mw=dn,
        reserve_activation=activation,
        reserve_revenue=reserve_revenue,
        plan_soc=plan_soc, soc_start=soc_start,
        dam_settled=settled,
    )
    return day, rd


# ==========================================================================
# Run one scenario: draw an XBID market, run the controller
# ==========================================================================
def _run_scenario(rd, cfg, rng, sim_stats=None, **kw):
    m = sim.simulate_xbid_day(rd.dam_price, rng, stats=(sim_stats or {}))
    scen = RealDay(
        dam_price=rd.dam_price, xbid_price=m.final_price,
        xbid_spread=m.half_spread * 2.0, imb_price=rd.imb_price,
        dam_net=rd.dam_net, tradeable=m.tradeable,
        up_mw=rd.up_mw, dn_mw=rd.dn_mw, reserve_activation=rd.reserve_activation,
        reserve_revenue=rd.reserve_revenue, plan_soc=rd.plan_soc,
        soc_start=rd.soc_start, dam_settled=rd.dam_settled,
    )
    return obr.run_day(scen, cfg, **kw)


def run_daily_sim(schedule_path, *, power_mw=1.5, energy_mwh=2.0, capacity_floor=0.1,
                  max_cycles=2.0, wear=2.0, max_trade_mwh=1.5, activation_rate=0.20,
                  min_edge_eur=10.0, reoptimize_dam=False, reserve_duration_h=0.25,
                  n_scenarios=200, seed=7, sim_stats=None, conservative=False):
    # conservative preset: calmer, less-liquid, wider-spread XBID than the
    # baseline calibration -> deliberately understates the opportunity.
    if conservative and sim_stats is None:
        sim_stats = {"dev_scale": 0.85, "liquidity_scale": 0.85, "spread_scale": 1.3}
    day, rd = load_daily_schedule(schedule_path, activation_rate=activation_rate)

    cfg = BatteryConfig()
    cfg.power_mw, cfg.energy_mwh, cfg.soc_max_mwh = power_mw, energy_mwh, energy_mwh
    cfg.soc_min_mwh = min(capacity_floor, energy_mwh)
    cfg.max_cycles_per_day = max_cycles
    cfg.degradation_eur_per_mwh = wear

    kw = dict(capacity_floor=capacity_floor, reserve_duration_h=reserve_duration_h,
              reveal_window=48, action_window=16, reopt_every=2, fee=0.0,
              max_trade_mwh=max_trade_mwh, min_edge_eur=min_edge_eur,
              reoptimize_dam=reoptimize_dam)

    rng = np.random.default_rng(seed)
    pnls, uplifts, cycles, imbs, socs, norders = [], [], [], [], [], []
    dam_rev = float(np.sum(rd.dam_price * rd.dam_net))
    dam_settled_rev = float(np.sum(rd.dam_price * rd.dam_settled))
    denom = abs(dam_settled_rev) if abs(dam_settled_rev) > 1e-6 else max(abs(dam_rev), 1.0)

    best_scen = None
    reopt_gains = []
    for k in range(n_scenarios):
        r = _run_scenario(rd, cfg, rng, sim_stats=sim_stats, **kw)
        pnls.append(r["economic"]); cycles.append(r["cycles"])
        imbs.append(r["imbalance_mwh"]); socs.append(r["soc_min"]); norders.append(r["n_orders"])
        reopt_gains.append(r.get("dam_reopt_gain", 0.0))
        # total system value = extra DAM revenue from reoptimisation + XBID PnL
        uplifts.append(100.0 * (r["economic"] + r.get("dam_reopt_gain", 0.0)) / denom)
        if best_scen is None or abs(r["economic"] - np.median(pnls)) < best_scen[0]:
            best_scen = (abs(r["economic"] - np.median(pnls)), r)

    pnls = np.array(pnls); uplifts = np.array(uplifts); reopt_gains = np.array(reopt_gains)
    total_value = pnls + reopt_gains
    return {
        "day": day, "n_scenarios": n_scenarios,
        "dam_energy_revenue": dam_rev, "dam_settled_revenue": dam_settled_rev,
        "reserve_revenue": rd.reserve_revenue,
        "reopt_gain_median": float(np.median(reopt_gains)),
        "xbid_pnl_median": float(np.median(pnls)),
        "total_value_median": float(np.median(total_value)),
        "total_value_p10": float(np.percentile(total_value, 10)),
        "total_value_p90": float(np.percentile(total_value, 90)),
        "pnl_mean": float(pnls.mean()), "pnl_median": float(np.median(pnls)),
        "pnl_p10": float(np.percentile(pnls, 10)), "pnl_p90": float(np.percentile(pnls, 90)),
        "uplift_mean": float(uplifts.mean()), "uplift_median": float(np.median(uplifts)),
        "uplift_p10": float(np.percentile(uplifts, 10)), "uplift_p90": float(np.percentile(uplifts, 90)),
        "max_cycles": float(np.max(cycles)), "max_imbalance": float(np.max(imbs)),
        "min_soe": float(np.min(socs)), "mean_orders": float(np.mean(norders)),
        "representative": best_scen[1],
    }


def main():
    ap = argparse.ArgumentParser(description="Daily D-1 -> D XBID simulation.")
    ap.add_argument("--schedule", required=True, help="BESS_Optimized_Schedule_*.xlsx")
    ap.add_argument("--power-mw", type=float, default=1.5)
    ap.add_argument("--energy-mwh", type=float, default=2.0)
    ap.add_argument("--capacity-floor", type=float, default=0.1)
    ap.add_argument("--max-cycles", type=float, default=2.0)
    ap.add_argument("--wear-eur-mwh", type=float, default=2.0)
    ap.add_argument("--max-trade-mwh", type=float, default=1.5)
    ap.add_argument("--activation-rate", type=float, default=0.20)
    ap.add_argument("--min-edge-eur", type=float, default=10.0)
    ap.add_argument("--reoptimize-dam", action="store_true")
    ap.add_argument("--conservative", action="store_true",
                    help="calmer/less-liquid/wider-spread XBID (understates opportunity)")
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    out = run_daily_sim(a.schedule, power_mw=a.power_mw, energy_mwh=a.energy_mwh,
                        capacity_floor=a.capacity_floor, max_cycles=a.max_cycles,
                        wear=a.wear_eur_mwh, max_trade_mwh=a.max_trade_mwh,
                        activation_rate=a.activation_rate, min_edge_eur=a.min_edge_eur,
                        reoptimize_dam=a.reoptimize_dam, n_scenarios=a.scenarios, seed=a.seed,
                        conservative=a.conservative)

    bar = "=" * 78
    print(bar)
    print(f"  DAILY XBID SIMULATION — {out['day']}   ({out['n_scenarios']} scenarios)")
    print(f"  battery {a.power_mw} MW / {a.energy_mwh} MWh   activation {a.activation_rate*100:.0f}%   "
          f"min_edge {a.min_edge_eur} EUR   reoptimize_dam {a.reoptimize_dam}")
    print(bar)
    print(f"  Committed DAM energy revenue .. {out['dam_energy_revenue']:>10,.1f} EUR")
    print(f"  Committed capacity revenue .... {out['reserve_revenue']:>10,.1f} EUR")
    print(bar)
    print("  VALUE PRODUCED BY THE SYSTEM (over the committed baseline):")
    print(f"    from DAM re-optimisation .... {out['reopt_gain_median']:>+10,.1f} EUR"
          f"   ({'ON' if a.reoptimize_dam else 'off'})")
    print(f"    from XBID intraday trading .. {out['xbid_pnl_median']:>+10,.1f} EUR")
    print(f"    TOTAL system value ......... {out['total_value_median']:>+10,.1f} EUR  "
          f"(P10 {out['total_value_p10']:,.0f} … P90 {out['total_value_p90']:,.0f})")
    print(bar)
    print(f"  Total uplift over DAM ......... {out['uplift_median']:+.2f}%  "
          f"(band {out['uplift_p10']:+.2f}% … {out['uplift_p90']:+.2f}%)")
    print(bar)
    print(f"  CONSTRAINTS (all scenarios): cycles ≤ {out['max_cycles']:.2f} | "
          f"imbalance {out['max_imbalance']:.2e} MWh | min SoE {out['min_soe']:.2f} MWh")
    print(bar)

    r = out["representative"]
    orders = sorted(r["orders"], key=lambda o: o["product_id"])
    print(f"  Representative scenario: {len(orders)} XBID orders, PnL {r['economic']:,.1f} EUR")
    if orders:
        print(f"  {'slot':>4} {'side':>4} {'qty':>6} {'fill':>7} {'strategy':>10}")
        for o in orders[:20]:
            print(f"  {o['product_id']:>4} {o['side']:>4} {o['qty_mwh']:>6.3f} "
                  f"{o['price']:>7.1f} {o['strategy']:>10}")
        if len(orders) > 20:
            print(f"  ... and {len(orders)-20} more")
    print(bar)


if __name__ == "__main__":
    main()
