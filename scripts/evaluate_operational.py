#!/usr/bin/env python3
"""One-stop EVALUATION of the operational XBID battery controller.

Runs the real-data backtest and prints:
  * data coverage,
  * a per-day results table (PnL, strategy volumes, SoC floor, cycles, imbalance),
  * constraint-adherence and risk summary,
  * a robustness sweep (imperfect intraday price prediction),
and writes five figures to ``--out-dir``:
  fig_dispatch_profile.png   DAM vs real XBID physical dispatch (+prices), one day
  fig_soc_trajectory.png     SoC path DAM vs after intraday + Capacity floor
  fig_cumulative_pnl.png     cumulative intraday PnL
  fig_daily_pnl_hist.png     daily PnL distribution
  fig_strategy_mix.png       MWh traded per strategy

Usage:
  python scripts/evaluate_operational.py --schedule-xlsx S.xlsx \
      --energy-xlsx E.xlsx --imbalance-xlsx I.xlsx --out-dir eval_out
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from xbid_trader.data.real_market_loader import load_real_market, coverage_report
from xbid_trader.market.battery import BatteryConfig, lp_optimal_dispatch
import operational_backtest_real as obr

C_DAM, C_ID, C_POS, C_NEG, C_GREY = "#5B6C8F", "#E08A3C", "#3C8C5A", "#B4453C", "#999999"


def main():
    ap = argparse.ArgumentParser(description="Evaluate the operational controller.")
    ap.add_argument("--schedule-xlsx", required=True)
    ap.add_argument("--energy-xlsx", required=True)
    ap.add_argument("--imbalance-xlsx", required=True)
    ap.add_argument("--months", nargs="*", default=["2026-01", "2026-02", "2026-03"])
    ap.add_argument("--max-cycles", type=float, default=2.0)
    ap.add_argument("--capacity-floor", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale the plan workbook (50 if normalised to 1 MW)")
    ap.add_argument("--activation-rate", type=float, default=0.40,
                    help="fraction of awarded reserve capacity assumed activated")
    ap.add_argument("--wear-eur-mwh", type=float, default=2.0)
    ap.add_argument("--max-trade-mwh", type=float, default=5.0)
    ap.add_argument("--reveal-window", type=int, default=48)
    ap.add_argument("--action-window", type=int, default=16)
    ap.add_argument("--reopt-every", type=int, default=2)
    ap.add_argument("--reoptimize-dam", action="store_true")
    ap.add_argument("--robustness", action="store_true", help="run the price-noise stress test (slower)")
    ap.add_argument("--out-dir", default="eval_out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    by_day = load_real_market(args.schedule_xlsx, args.energy_xlsx, args.imbalance_xlsx,
                              args.months, scale_mw=args.scale,
                              activation_rate=args.activation_rate)
    cfg = BatteryConfig(); cfg.max_cycles_per_day = args.max_cycles
    cfg.degradation_eur_per_mwh = args.wear_eur_mwh; cfg.soc_min_mwh = args.capacity_floor
    if args.reoptimize_dam:
        for _, rd in by_day.items():
            rd.dam_net = lp_optimal_dispatch(rd.dam_price, cfg)

    def run(seed, noise=0.0):
        rng = np.random.default_rng(seed)
        return {d: obr.run_day(by_day[d], cfg, capacity_floor=args.capacity_floor,
                               wear=args.wear_eur_mwh, reveal_window=args.reveal_window,
                               action_window=args.action_window, reopt_every=args.reopt_every,
                               fee=0.0, max_trade_mwh=args.max_trade_mwh, rng=rng,
                               reveal_noise=noise) for d in sorted(by_day)}

    bar = "═" * 96
    print(bar)
    print(f"  EVALUATION — Operational XBID battery controller")
    print(f"  {coverage_report(by_day)}")
    print(f"  P={cfg.power_mw:.0f}MW E={cfg.energy_mwh:.0f}MWh  Cs={cfg.max_cycles_per_day}  "
          f"floor={args.capacity_floor:.0f}MWh  wear={args.wear_eur_mwh}€/MWh  "
          f"max_trade={args.max_trade_mwh}MWh  reoptimize_dam={args.reoptimize_dam}")
    print(bar)

    rows = run(args.seed)
    days = sorted(rows)

    # ── per-day table ─────────────────────────────────────────────────────
    print(f"  {'Day':<12}{'ID PnL €':>9}{'buyback':>9}{'resell':>8}{'spike':>7}"
          f"{'chgNeg':>8}{'#XBID':>7}{'SOEmin':>8}{'cyc':>6}{'imb':>6}")
    print("  " + "─" * 94)
    for d in days:
        r = rows[d]
        print(f"  {d:<12}{r['economic']:>9.0f}{r['buyback']:>9.1f}{r['resell']:>8.1f}"
              f"{r['spike_dis']:>7.1f}{r['cheap_chg']:>8.1f}{r['n_tradeable']:>7}"
              f"{r['soc_min']:>8.1f}{max(r['charge_cycles'],r['discharge_cycles']):>6.2f}"
              f"{r['imbalance_mwh']:>6.1f}")

    econ = np.array([rows[d]["economic"] for d in days])
    dam_rev = sum(rows[d]["dam_revenue"] for d in days)
    strat = {k: sum(rows[d][k] for d in days) for k in ("buyback", "resell", "spike_dis", "cheap_chg")}
    cvar = float(np.mean(np.sort(econ)[:max(1, len(econ)//20)]))
    print("  " + "─" * 94)
    print(bar)
    print(f"  SUMMARY")
    print(f"    Intraday uplift .......... {econ.sum():>12,.0f} €  (+{100*econ.sum()/abs(dam_rev):.2f}% of DAM)")
    print(f"    Positive days ............ {int((econ>0).sum())}/{len(days)}")
    print(f"    Daily PnL ................ mean {econ.mean():.0f} €   std {econ.std():.0f} €   CVaR5% {cvar:.0f} €")
    print(f"    CONSTRAINTS (all days) ... imbalance max {max(rows[d]['imbalance_mwh'] for d in days):.3f} MWh"
          f" | SoC min {min(rows[d]['soc_min'] for d in days):.1f} (floor {args.capacity_floor:.0f})"
          f" | cycles max {max(max(rows[d]['charge_cycles'],rows[d]['discharge_cycles']) for d in days):.2f} (cap {cfg.max_cycles_per_day})")
    print(f"    Strategy MWh ............. buy-back {strat['buyback']:.0f} | resell {strat['resell']:.0f}"
          f" | spike {strat['spike_dis']:.0f} | cheap/neg {strat['cheap_chg']:.0f}")
    print(bar)

    if args.robustness:
        print("  ROBUSTNESS — imperfect price prediction (uplift %):")
        for nz in (0.0, 0.10, 0.20):
            vals = []
            for s in (7, 42):
                rr = run(s, noise=nz)
                e = sum(x["economic"] for x in rr.values()); dd = sum(x["dam_revenue"] for x in rr.values())
                vals.append(100 * e / abs(dd))
            print(f"    noise {int(nz*100):>3d}% : {np.mean(vals):+.2f}%   (min {min(vals):+.2f}%)"
                  + ("   [deterministic]" if nz == 0 else ""))
        print(bar)

    # ── figures ───────────────────────────────────────────────────────────
    rep = max(days, key=lambda d: rows[d]["economic"])
    r = rows[rep]; n = len(r["dam_net_arr"]); x = np.arange(n)
    print(f"  Representative day (max PnL): {rep}  (+{r['economic']:.0f} €)")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - 0.2, r["dam_net_arr"], width=0.4, color=C_DAM, label="DAM committed net")
    ax.bar(x + 0.2, r["committed_net"], width=0.4, color=C_ID, label="After XBID (physical)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Net dispatch  MWh/15min  (+discharge / −charge)"); ax.set_xlabel("15-min product")
    ax.set_title(f"DAM vs real XBID dispatch profile — {rep}")
    ax2 = ax.twinx()
    ax2.plot(x, r["dam_price_arr"], color=C_DAM, lw=1.2, ls="--", alpha=.7, label="DAM price")
    xb = r["xbid_price_arr"]
    ax2.plot(x, np.where(np.isfinite(xb), xb, np.nan), color=C_ID, lw=1.4, label="XBID price")
    ax2.set_ylabel("€/MWh")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig_dispatch_profile.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(r["soc_dam_arr"], color=C_DAM, lw=1.8, label="SoC (DAM only)")
    ax.plot(r["soc_final_arr"], color=C_ID, lw=1.8, label="SoC (after XBID)")
    ax.axhline(args.capacity_floor, color=C_NEG, ls=":", lw=1.5, label=f"Capacity floor ({args.capacity_floor:.0f})")
    ax.axhline(cfg.soc_max_mwh, color=C_GREY, ls=":", lw=1)
    ax.set_ylabel("SoC  MWh"); ax.set_xlabel("15-min product")
    ax.set_title(f"State of charge — {rep}"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "fig_soc_trajectory.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(np.cumsum(econ), color=C_POS, lw=2)
    ax.fill_between(range(len(econ)), np.cumsum(econ), color=C_POS, alpha=.12)
    ax.set_ylabel("Cumulative intraday PnL  €"); ax.set_xlabel("day")
    ax.set_title(f"Cumulative XBID uplift — {len(days)} days (+{100*econ.sum()/abs(dam_rev):.1f}% of DAM)")
    fig.tight_layout(); fig.savefig(out / "fig_cumulative_pnl.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(econ, bins=30, color=C_ID, alpha=.85)
    ax.axvline(0, color="k", lw=1); ax.axvline(econ.mean(), color=C_POS, lw=1.5, ls="--", label=f"mean {econ.mean():.0f} €")
    ax.set_xlabel("daily PnL  €"); ax.set_ylabel("days"); ax.set_title("Daily PnL distribution"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_daily_pnl_hist.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = ["financial\nbuy-back", "resell", "RES-spike\ndischarge", "cheap/neg\ncharge"]
    vals = [strat["buyback"], strat["resell"], strat["spike_dis"], strat["cheap_chg"]]
    ax.bar(names, vals, color=[C_DAM, C_DAM, C_ID, C_POS])
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MWh traded (all days)"); ax.set_title("Strategy contribution")
    fig.tight_layout(); fig.savefig(out / "fig_strategy_mix.png", dpi=120); plt.close(fig)

    print(f"  Figures written to: {out.resolve()}")
    print(bar)


if __name__ == "__main__":
    main()
