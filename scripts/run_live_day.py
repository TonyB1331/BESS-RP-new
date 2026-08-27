#!/usr/bin/env python3
"""Live single-day runner — the XBID decisions for ONE delivery day.

This is the shape the system runs in production: one day at a time. Give it a
date and the three workbooks; it loads only that day, runs the controller, and
returns/prints the XBID orders it decided, quarter by quarter, plus a summary.

    from run_live_day import run_live_day
    out = run_live_day("2026-03-14", schedule_xlsx, energy_xlsx, imbalance_xlsx,
                       scale=50, activation_rate=0.20, max_cycles=2.0,
                       capacity_floor=10, reoptimize_dam=False)
    out["orders"]     # list of decisions (quarter, time, side, qty, price, ...)
    out["summary"]    # dam_revenue, xbid_pnl, cycles, imbalance, soc_min, ...

CLI:
    python run_live_day.py --date 2026-03-14 \
        --schedule-xlsx CAP.xlsx --energy-xlsx EM.xlsx --imbalance-xlsx IMB.xlsx \
        --scale 50 --activation-rate 0.20 --max-cycles 2.0 --capacity-floor 10 \
        --csv orders_2026-03-14.csv
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

from xbid_trader.data.real_market_loader import load_real_market       # noqa: E402
from xbid_trader.market.battery import BatteryConfig                   # noqa: E402
import operational_backtest_real as obr                               # noqa: E402


def _slot_time(day: str, q: int) -> str:
    """15-min product index -> HH:MM of the delivery day."""
    h, m = divmod(q * 15, 60)
    return f"{h:02d}:{m:02d}"


def run_live_day(date, schedule_xlsx, energy_xlsx, imbalance_xlsx, *,
                 scale=50.0, activation_rate=0.20, max_cycles=2.0,
                 capacity_floor=10.0, wear=2.0, max_trade_mwh=5.0,
                 reserve_duration_h=0.25, reveal_window=48, action_window=16,
                 reopt_every=2, fee=0.0, reoptimize_dam=False, min_edge_eur=10.0,
                 roundtrip_margin_eur=None,
                 power_mw=50.0, energy_mwh=100.0, preloaded=None):
    """Run the controller for a SINGLE day and return its XBID decisions.

    ``preloaded`` (optional): a dict {date -> RealDay} from a previous
    ``load_real_market`` call, to avoid reloading the workbook for every day
    when auditing many days in a loop.
    """
    by_day = preloaded
    if by_day is None or date not in by_day:
        month = str(date)[:7]
        by_day = load_real_market(schedule_xlsx, energy_xlsx, imbalance_xlsx, [month],
                                  scale_mw=scale, activation_rate=activation_rate)
    if date not in by_day:
        raise KeyError(f"{date} not found (available: {sorted(by_day)[:3]}... "
                       f"{len(by_day)} days)")
    rd = by_day[date]

    cfg = BatteryConfig()
    cfg.power_mw, cfg.energy_mwh, cfg.soc_max_mwh = power_mw, energy_mwh, energy_mwh
    cfg.soc_min_mwh = min(capacity_floor, energy_mwh)
    cfg.max_cycles_per_day = max_cycles
    cfg.degradation_eur_per_mwh = wear

    r = obr.run_day(rd, cfg, capacity_floor=capacity_floor,
                    reserve_duration_h=reserve_duration_h, reoptimize_dam=reoptimize_dam,
                    reveal_window=reveal_window, action_window=action_window,
                    reopt_every=reopt_every, fee=fee, max_trade_mwh=max_trade_mwh,
                    min_edge_eur=min_edge_eur, roundtrip_margin_eur=roundtrip_margin_eur)

    dam_p = r["dam_price_arr"]
    xbid_p = r["xbid_price_arr"]
    dam_net = r["dam_net_arr"]
    up = r["reserve_up_mw"]
    dn = r["reserve_dn_mw"]

    # enrich each decision with time, prices and context
    orders = []
    for o in sorted(r["orders"], key=lambda x: x["product_id"]):
        i = o["product_id"]
        orders.append({
            "date": date,
            "quarter": i,
            "time": _slot_time(date, i),
            "side": o["side"],
            "qty_mwh": o["qty_mwh"],
            "fill_price": o["price"],
            "dam_price": round(float(dam_p[i]), 2),
            "xbid_price": round(float(xbid_p[i]), 2) if np.isfinite(xbid_p[i]) else None,
            "dam_net_mwh": round(float(dam_net[i]), 3),
            "reserve_up_mw": round(float(up[i]), 2),
            "reserve_dn_mw": round(float(dn[i]), 2),
            "cash_eur": o["cash"],
            "strategy": o["strategy"],
        })

    summary = {
        "date": date,
        "dam_revenue": r["dam_revenue"],
        "reserve_revenue": r["reserve_revenue"],
        "xbid_pnl": r["economic"],
        "xbid_cash": r["cash"],
        "terminal_value": r["terminal_value"],
        "soc_drift": r["soc_drift"],
        "wear": r["wear"],
        "n_orders": r["n_orders"],
        "n_tradeable": r["n_tradeable"],
        "cycles": r["cycles"],
        "base_cycles": r["base_cycles"],
        "cycle_cap": r["cycle_cap"],
        "imbalance_mwh": r["imbalance_mwh"],
        "base_imbalance_mwh": r["base_imbalance_mwh"],
        "soc_min": r["soc_min"],
        "reoptimize_dam": reoptimize_dam,
        "min_edge_eur": min_edge_eur,
        "strategy_mwh": {k: r[k] for k in ("buyback", "resell", "spike_dis", "cheap_chg")},
    }
    return {"orders": orders, "summary": summary, "raw": r}


def _print_report(out):
    s = out["summary"]
    bar = "=" * 84
    print(bar)
    print(f"  LIVE DAY {s['date']}   reoptimize_dam={s['reoptimize_dam']}")
    print(bar)
    print(f"  DAM revenue {s['dam_revenue']:>10,.0f} EUR   capacity {s['reserve_revenue']:>10,.0f} EUR   "
          f"XBID PnL {s['xbid_pnl']:>8,.0f} EUR")
    print(f"  cycles {s['cycles']:.2f} / cap {s['cycle_cap']:.2f}  (baseline {s['base_cycles']:.2f})   "
          f"imbalance {s['imbalance_mwh']:.2e} MWh (baseline {s['base_imbalance_mwh']:.2e})   "
          f"min SoE {s['soc_min']:.1f} MWh")
    mix = s["strategy_mwh"]
    print(f"  strategy MWh: buy-back {mix['buyback']:.1f} | resell {mix['resell']:.1f} | "
          f"spike {mix['spike_dis']:.1f} | cheap/neg {mix['cheap_chg']:.1f}")
    print(bar)

    orders = out["orders"]
    if not orders:
        print("  No XBID orders — no exploitable deviation this day.")
        print(bar)
        return
    print(f"  {'time':>5} {'side':>4} {'qty':>6} {'fill':>8} {'DAM':>7} {'XBID':>7} "
          f"{'DAMnet':>7} {'up':>4} {'dn':>4} {'cash':>8}  strategy")
    print("  " + "-" * 80)
    for o in orders:
        xb = f"{o['xbid_price']:>7.1f}" if o["xbid_price"] is not None else f"{'-':>7}"
        print(f"  {o['time']:>5} {o['side']:>4} {o['qty_mwh']:>6.2f} {o['fill_price']:>8.1f} "
              f"{o['dam_price']:>7.1f} {xb} {o['dam_net_mwh']:>7.2f} "
              f"{o['reserve_up_mw']:>4.0f} {o['reserve_dn_mw']:>4.0f} {o['cash_eur']:>8.1f}  {o['strategy']}")
    print("  " + "-" * 80)
    print(f"  {len(orders)} orders   net cash {sum(o['cash_eur'] for o in orders):,.1f} EUR")
    print(bar)


def main():
    ap = argparse.ArgumentParser(description="Run the XBID controller for one day.")
    ap.add_argument("--date", required=True, help="delivery day YYYY-MM-DD")
    ap.add_argument("--schedule-xlsx", required=True)
    ap.add_argument("--energy-xlsx", required=True)
    ap.add_argument("--imbalance-xlsx", required=True)
    ap.add_argument("--scale", type=float, default=50.0)
    ap.add_argument("--activation-rate", type=float, default=0.20)
    ap.add_argument("--max-cycles", type=float, default=2.0)
    ap.add_argument("--capacity-floor", type=float, default=10.0)
    ap.add_argument("--wear-eur-mwh", type=float, default=2.0)
    ap.add_argument("--max-trade-mwh", type=float, default=5.0)
    ap.add_argument("--reserve-duration-h", type=float, default=0.25)
    ap.add_argument("--reveal-window", type=int, default=48)
    ap.add_argument("--action-window", type=int, default=16)
    ap.add_argument("--reopt-every", type=int, default=2)
    ap.add_argument("--fee-per-mwh", type=float, default=0.0)
    ap.add_argument("--reoptimize-dam", action="store_true")
    ap.add_argument("--min-edge-eur", type=float, default=10.0,
                    help="conservative trigger for cancelling committed positions (0=off)")
    ap.add_argument("--power-mw", type=float, default=50.0)
    ap.add_argument("--energy-mwh", type=float, default=100.0)
    ap.add_argument("--csv", help="write the order blotter to this CSV")
    a = ap.parse_args()

    out = run_live_day(
        a.date, a.schedule_xlsx, a.energy_xlsx, a.imbalance_xlsx,
        scale=a.scale, activation_rate=a.activation_rate, max_cycles=a.max_cycles,
        capacity_floor=a.capacity_floor, wear=a.wear_eur_mwh, max_trade_mwh=a.max_trade_mwh,
        reserve_duration_h=a.reserve_duration_h, reveal_window=a.reveal_window,
        action_window=a.action_window, reopt_every=a.reopt_every, fee=a.fee_per_mwh,
        reoptimize_dam=a.reoptimize_dam, min_edge_eur=a.min_edge_eur,
        power_mw=a.power_mw, energy_mwh=a.energy_mwh)

    _print_report(out)
    if a.csv and out["orders"]:
        pd.DataFrame(out["orders"]).to_csv(a.csv, index=False)
        print(f"  orders -> {a.csv}")


if __name__ == "__main__":
    main()
