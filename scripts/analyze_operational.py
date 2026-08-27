#!/usr/bin/env python3
"""Deep evaluation of the operational controller: robustness stats + figures.

Produces:
  * fig_dispatch_profile.png  — DAM vs post-XBID physical dispatch + prices (a day)
  * fig_soc_trajectory.png    — SoC path DAM vs after intraday + Capacity floor
  * fig_cumulative_pnl.png    — cumulative intraday PnL over the backtest
  * fig_daily_pnl_hist.png    — daily PnL distribution
  * fig_strategy_mix.png      — MWh traded per strategy (aggregate)
and prints a robustness / constraint-adherence report (seed stability, floor,
cycles, imbalance).
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
from xbid_trader.market.battery import BatteryConfig
import operational_backtest_real as obr

OUT = Path("/mnt/user-data/outputs")
C_DAM, C_ID, C_POS, C_NEG, C_FLOOR = "#5B6C8F", "#E08A3C", "#3C8C5A", "#B4453C", "#999999"


def run_all(by_day, cfg, floor, wear, seed, **kw):
    rng = np.random.default_rng(seed)
    rows = {}
    for day in sorted(by_day):
        rows[day] = obr.run_day(by_day[day], cfg, capacity_floor=floor, wear=wear,
                                reveal_window=kw.get("rw", 48), action_window=kw.get("aw", 16),
                                reopt_every=kw.get("re", 2), fee=0.0,
                                max_trade_mwh=kw.get("mt", 5.0), rng=rng)
    return rows


def main():
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args()

    by_day = load_real_market(args.schedule_xlsx, args.energy_xlsx, args.imbalance_xlsx,
                              args.months, scale_mw=args.scale,
                              activation_rate=args.activation_rate)
    cfg = BatteryConfig(); cfg.max_cycles_per_day = args.max_cycles
    cfg.degradation_eur_per_mwh = args.wear_eur_mwh; cfg.soc_min_mwh = args.capacity_floor
    print("Data:", coverage_report(by_day))

    # ── robustness: sensitivity to IMPERFECT intraday price prediction ────
    # The real revelation is deterministic (linear toward the true VWAP), so
    # seed alone is trivially stable. The meaningful stress test is: how much
    # does uplift degrade when the price the LP acts on is a NOISY forecast of
    # the eventual XBID clearing price? (fills still happen at the real price.)
    print("\nROBUSTNESS — imperfect price prediction (uplift %, 2 seeds each):")
    for nz in (0.0, 0.10, 0.20):
        vals = []
        for s in (7, 42):
            rng = np.random.default_rng(s)
            e = d = 0.0
            for day in sorted(by_day):
                r = obr.run_day(by_day[day], cfg, capacity_floor=args.capacity_floor,
                                wear=args.wear_eur_mwh, reveal_window=48, action_window=16,
                                reopt_every=4, fee=0.0, max_trade_mwh=5.0, rng=rng, reveal_noise=nz)
                e += r["economic"]; d += r["dam_revenue"]
            vals.append(100 * e / abs(d))
        tag = "(deterministic)" if nz == 0 else ""
        print(f"  noise={nz:.2f}: mean={np.mean(vals):+.2f}  min={min(vals):+.2f} {tag}")

    # ── main run (seed 7) for figures + constraint checks ─────────────────
    rows = run_all(by_day, cfg, args.capacity_floor, args.wear_eur_mwh, 7)
    days = sorted(rows)
    econ = np.array([rows[d]["economic"] for d in days])
    dam_rev = sum(rows[d]["dam_revenue"] for d in days)
    max_imb = max(rows[d]["imbalance_mwh"] for d in days)
    min_soc = min(rows[d]["soc_min"] for d in days)
    max_cyc = max(max(rows[d]["charge_cycles"], rows[d]["discharge_cycles"]) for d in days)
    print(f"\nCONSTRAINT ADHERENCE (all {len(days)} days):")
    print(f"  max imbalance     = {max_imb:.3f} MWh   (target 0)")
    print(f"  min SoC           = {min_soc:.1f} MWh    (floor {args.capacity_floor:.0f})")
    print(f"  max cycles        = {max_cyc:.2f}        (cap {args.max_cycles})")
    print(f"  uplift            = +{100*econ.sum()/abs(dam_rev):.2f}%  ({econ.sum():,.0f} €)")
    print(f"  positive days     = {int((econ>0).sum())}/{len(days)}")
    print(f"  daily PnL: mean={econ.mean():.0f} std={econ.std():.0f} "
          f"CVaR5%={np.mean(np.sort(econ)[:max(1,len(econ)//20)]):.0f}")

    strat = {k: sum(rows[d][k] for d in days) for k in ("buyback", "resell", "spike_dis", "cheap_chg")}
    print(f"  strategy MWh: {strat}")

    # representative day = biggest-PnL day with a real event mix
    rep = max(days, key=lambda d: rows[d]["economic"])
    r = rows[rep]; n = len(r["dam_net_arr"]); x = np.arange(n)
    print(f"\nRepresentative day (max PnL): {rep}  (+{r['economic']:.0f} €)")

    # ── Fig 1: DAM vs post-XBID dispatch profile + prices ─────────────────
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - 0.2, r["dam_net_arr"], width=0.4, color=C_DAM, label="DAM committed net")
    ax.bar(x + 0.2, r["committed_net"], width=0.4, color=C_ID, label="After XBID (physical)")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Net dispatch  MWh/15min  (+ discharge / − charge)")
    ax.set_xlabel("15-min product"); ax.set_title(f"DAM vs real XBID dispatch profile — {rep}")
    ax2 = ax.twinx()
    ax2.plot(x, r["dam_price_arr"], color=C_DAM, lw=1.2, ls="--", alpha=.7, label="DAM price")
    xb = r["xbid_price_arr"].copy()
    ax2.plot(x, np.where(np.isfinite(xb), xb, np.nan), color=C_ID, lw=1.4, label="XBID price")
    ax2.set_ylabel("€/MWh")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_dispatch_profile.png", dpi=120); plt.close(fig)

    # ── Fig 2: SoC trajectory ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(r["soc_dam_arr"], color=C_DAM, lw=1.8, label="SoC (DAM only)")
    ax.plot(r["soc_final_arr"], color=C_ID, lw=1.8, label="SoC (after XBID)")
    ax.axhline(args.capacity_floor, color=C_NEG, ls=":", lw=1.5, label=f"Capacity floor ({args.capacity_floor:.0f})")
    ax.axhline(cfg.soc_max_mwh, color=C_FLOOR, ls=":", lw=1)
    ax.set_ylabel("SoC  MWh"); ax.set_xlabel("15-min product")
    ax.set_title(f"State of charge — {rep}"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig_soc_trajectory.png", dpi=120); plt.close(fig)

    # ── Fig 3: cumulative PnL ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(np.cumsum(econ), color=C_POS, lw=2)
    ax.fill_between(range(len(econ)), np.cumsum(econ), color=C_POS, alpha=.12)
    ax.set_ylabel("Cumulative intraday PnL  €"); ax.set_xlabel("day")
    ax.set_title(f"Cumulative XBID uplift — {len(days)} days  (+{100*econ.sum()/abs(dam_rev):.1f}% of DAM)")
    fig.tight_layout(); fig.savefig(OUT / "fig_cumulative_pnl.png", dpi=120); plt.close(fig)

    # ── Fig 4: daily PnL histogram ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(econ, bins=30, color=C_ID, alpha=.85)
    ax.axvline(0, color="k", lw=1); ax.axvline(econ.mean(), color=C_POS, lw=1.5, ls="--",
                                               label=f"mean {econ.mean():.0f} €")
    ax.set_xlabel("daily PnL  €"); ax.set_ylabel("days"); ax.set_title("Daily PnL distribution")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_daily_pnl_hist.png", dpi=120); plt.close(fig)

    # ── Fig 5: strategy mix ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = ["financial\nbuy-back", "resell", "RES-spike\ndischarge", "cheap/neg\ncharge"]
    vals = [strat["buyback"], strat["resell"], strat["spike_dis"], strat["cheap_chg"]]
    ax.bar(names, vals, color=[C_DAM, C_DAM, C_ID, C_POS])
    ax.set_ylabel("MWh traded (all days)"); ax.set_title("Strategy contribution")
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_strategy_mix.png", dpi=120); plt.close(fig)

    print("\nFigures written to", OUT)


if __name__ == "__main__":
    main()
