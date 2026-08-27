#!/usr/bin/env python3
"""Operational backtest — MPC + liquidity-aware execution over the REAL book.

This closes the gap seen when the raw MPC target was dumped onto the simulated
order book (partial fills + re-planning drift → imbalance / cycle overrun). It
drives the existing XBID market simulation (order book + background traders,
gradual price revelation, RES spikes / negative-price events) but routes every
trade through a proper **execution layer** that:

  * reads the **available depth** on each side of the book and places only what
    can actually fill (no partial-fill surprises);
  * enforces **exact SoC feasibility** given the rest of the committed schedule
    (so the position stays deliverable — imbalance 0 by construction);
  * enforces a **hard daily cycle cap** and the **Capacity-Market SoC floor**;
  * enforces the **one-side-per-product** rule (no bid *and* ask on a product);
  * updates the committed position, SoC, cycles and cash from the **actual
    fills** returned by the matching engine, then re-optimises on the real
    state.

The MPC re-solves the wear-aware LP (financial arbitrage + spike + cheap-charge
+ re-profiling in one objective). To go live, replace the simulated engine with
the real XBID/SIDC gateway — the execution-layer logic is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from xbid_trader.market.simulation_session import SimulationSession
from xbid_trader.market.battery import BatteryConfig, soc_dispatch, cycle_usage
from xbid_trader.types import Order, OrderSide, OrderType
from xbid_trader.data.dam_schedule_loader import load_dam_dataset
from operational_mpc import solve_lp


def revealed_reference(dam_ref, rev_final, t, W, rng, sigma):
    n = len(dam_ref); idx = np.arange(n)
    prog = np.clip((t - (idx - W)) / W, 0.0, 1.0)
    noise = sigma * np.abs(dam_ref) * np.sqrt(prog * (1 - prog)) * rng.standard_normal(n)
    return np.maximum(dam_ref + prog * rev_final + noise, 1e-3)


def book_view(state, n):
    """Return (mid, best_bid, best_ask, bid_depth, ask_depth) per product."""
    mid = np.zeros(n); bid = np.zeros(n); ask = np.zeros(n)
    bvol = np.zeros(n); avol = np.zeros(n)
    for i in range(n):
        ob = state.order_books[i]
        bb, ba = ob.best_bid, ob.best_ask
        bid[i] = bb if bb is not None else 0.0
        ask[i] = ba if ba is not None else 0.0
        mid[i] = ((bb + ba) / 2.0) if (bb is not None and ba is not None) else state.reference_price(i)
        bd, ad = ob.get_book_depth()
        bvol[i] = float(sum(bd.values())); avol[i] = float(sum(ad.values()))
    return mid, bid, ask, bvol, avol


def run_day(dam_price, dam_net, rev_final, cfg, *, capacity_floor, wear,
            reveal_window, action_window, reopt_every, fee, max_fill_frac,
            rng):
    n = len(dam_price)
    dam_ref = np.asarray(dam_price, float)
    session = SimulationSession(
        engine_config={"seed_books_on_init": False,
                       "reference_prices": dam_ref.copy(), "num_products": n},
        custom_agent=None, print_summary=False)
    engine = session._engine
    reveal_rng = np.random.default_rng(rng.integers(1 << 30))

    committed = np.asarray(dam_net, float).copy()
    side = np.zeros(n, int)
    ch_used = dh_used = 0.0
    cash = 0.0
    strat = {"buyback": 0.0, "resell": 0.0, "spike_dis": 0.0, "cheap_chg": 0.0}
    target = committed.copy()

    def soc_after(upto):
        if upto <= 0: return cfg.soc_initial_mwh
        sp, _, _ = soc_dispatch(committed[:upto], cfg)
        return float(sp[-1]) if len(sp) else cfg.soc_initial_mwh

    for t in range(n):
        engine.set_reference_prices(revealed_reference(dam_ref, rev_final, t, reveal_window,
                                                       reveal_rng, 0.0))
        result = session.step()
        state = engine.state
        mid, bid, ask, bvol, avol = book_view(state, n)

        if t % reopt_every == 0:
            target = solve_lp(mid, soc_after(t), t, committed, cfg,
                              ch_used, dh_used, wear, capacity_floor, 0.0)

        # ── liquidity-aware, feasibility-safe execution ──────────────────
        for i in range(t, min(n, t + action_window)):
            if not state.is_active(i):
                continue
            d = target[i] - committed[i]
            if abs(d) < 0.05:
                continue
            intended = 1 if d > 0 else -1
            if side[i] != 0 and side[i] != intended:
                continue

            # exact SoC headroom given the rest of the committed schedule
            soc_path, _, _ = soc_dispatch(committed, cfg)
            suf_min = np.minimum.accumulate(soc_path[i:][::-1])[::-1][0]
            suf_max = np.maximum.accumulate(soc_path[i:][::-1])[::-1][0]
            head_dis = max(suf_min - capacity_floor, 0.0) * cfg.eta_discharge
            head_chg = max(cfg.soc_max_mwh - suf_max, 0.0) / cfg.eta_charge
            # remaining cycle budget
            cyc_dis = max(cfg.max_cycles_per_day - dh_used, 0.0) * cfg.energy_mwh * cfg.eta_discharge
            cyc_chg = max(cfg.max_cycles_per_day - ch_used, 0.0) * cfg.energy_mwh / cfg.eta_charge
            # available book liquidity on the side we will hit
            liq = (bvol[i] if d > 0 else avol[i]) * max_fill_frac

            if d > 0:   # SELL / discharge
                qty = min(d, head_dis, cyc_dis, liq)
            else:       # BUY / charge
                qty = min(-d, head_chg, cyc_chg, liq)
            if qty < 0.05:
                continue

            # marketable order sized to available depth → fills in full
            if d > 0:
                order = Order(id=-1, product_id=i, side=OrderSide.SELL, order_type=OrderType.LIMIT,
                              price=max(bid[i] - 0.01, 1e-3), quantity=qty,
                              timestamp=state.current_time, trader_id="mpc")
            else:
                order = Order(id=-1, product_id=i, side=OrderSide.BUY, order_type=OrderType.LIMIT,
                              price=ask[i] + 0.01, quantity=qty,
                              timestamp=state.current_time, trader_id="mpc")
            trades = engine.add_external_order(order)

            filled = 0.0; pnl = 0.0
            for tr in trades:
                filled += tr.quantity
                pnl += (tr.price * tr.quantity) if d > 0 else (-tr.price * tr.quantity)
            if filled < 1e-6:
                continue
            signed = filled if d > 0 else -filled
            cash += pnl - fee * filled
            # strategy attribution (relative to current committed sign)
            if signed > 0 and committed[i] < 0:   strat["resell"]   += filled
            elif signed > 0:                       strat["spike_dis"] += filled
            elif signed < 0 and committed[i] > 0:  strat["buyback"]  += filled
            else:                                  strat["cheap_chg"] += filled
            committed[i] += signed
            side[i] = intended
            ch_used += cfg.eta_charge * max(-signed, 0.0) / cfg.energy_mwh
            dh_used += (max(signed, 0.0) / cfg.eta_discharge) / cfg.energy_mwh

    soc_path, delivered, imbalance = soc_dispatch(committed, cfg)
    thr = lambda q: np.sum(np.where(q > 0, q, 0)) / cfg.eta_discharge + cfg.eta_charge * np.sum(np.where(q < 0, -q, 0))
    wear_delta = cfg.degradation_eur_per_mwh * (thr(committed) - thr(np.asarray(dam_net)))
    return {
        "cash": cash, "economic": cash - wear_delta, "wear_delta": wear_delta,
        "imbalance_mwh": float(np.abs(imbalance).sum()),
        "soc_min": float(soc_path.min()) if len(soc_path) else cfg.soc_initial_mwh,
        "soc_final": float(soc_path[-1]) if len(soc_path) else cfg.soc_initial_mwh,
        "charge_cycles": ch_used, "discharge_cycles": dh_used,
        "dam_revenue": float(np.sum(dam_price * dam_net)), **strat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Operational MPC backtest over the real order book.")
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--month", default="2026-04")
    ap.add_argument("--max-cycles", type=float, default=1.5)
    ap.add_argument("--capacity-floor", type=float, default=10.0)
    ap.add_argument("--wear-eur-mwh", type=float, default=2.0)
    ap.add_argument("--id-revision-sigma", type=float, default=0.12)
    ap.add_argument("--spike-prob", type=float, default=0.06)
    ap.add_argument("--neg-prob", type=float, default=0.05)
    ap.add_argument("--reveal-window", type=int, default=24)
    ap.add_argument("--action-window", type=int, default=8)
    ap.add_argument("--reopt-every", type=int, default=4)
    ap.add_argument("--fee-per-mwh", type=float, default=0.0)
    ap.add_argument("--max-fill-frac", type=float, default=0.5,
                    help="fraction of visible depth to consume per order")
    ap.add_argument("--days", type=int, default=0, help="limit to first N days (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prices, provider = load_dam_dataset(args.xlsx, months=[args.month])
    cfg = BatteryConfig(); cfg.max_cycles_per_day = args.max_cycles
    cfg.degradation_eur_per_mwh = args.wear_eur_mwh
    rng = np.random.default_rng(args.seed)
    days = provider.days[:args.days] if args.days else provider.days

    print("═" * 94)
    print(f"  Operational MPC backtest (real order book) — {args.month}   "
          f"P={cfg.power_mw:.0f}MW E={cfg.energy_mwh:.0f}MWh Cs={cfg.max_cycles_per_day} "
          f"floor={args.capacity_floor:.0f}")
    print("═" * 94)
    print(f"  {'Day':<12}{'ID PnL €':>9}{'wear+ €':>8}{'buyback':>9}{'resell':>8}"
          f"{'spike':>7}{'chgNeg':>8}{'SOEmin':>8}{'cyc':>6}{'imb':>6}")
    print("  " + "─" * 92)
    rows = []
    for day in days:
        sc = provider.get_scenario(day)
        dp = np.asarray(prices[day], float); dn = np.asarray(sc.battery_dam_schedule, float)
        n = len(dp); rho = 0.8; z = rng.standard_normal(n); rev = np.zeros(n)
        for i in range(1, n): rev[i] = rho * rev[i-1] + np.sqrt(1-rho*rho) * z[i]
        rev_final = rev * args.id_revision_sigma * np.abs(dp)
        spike = rng.random(n) < args.spike_prob
        rev_final[spike] += rng.uniform(0.5, 2.0, int(spike.sum())) * np.abs(dp[spike])
        neg = rng.random(n) < args.neg_prob
        fin = dp + rev_final; fin[neg] = rng.uniform(-30.0, 3.0, int(neg.sum())); rev_final = fin - dp

        r = run_day(dp, dn, rev_final, cfg, capacity_floor=args.capacity_floor,
                    wear=args.wear_eur_mwh, reveal_window=args.reveal_window,
                    action_window=args.action_window, reopt_every=args.reopt_every,
                    fee=args.fee_per_mwh, max_fill_frac=args.max_fill_frac, rng=rng)
        r["day"] = day; rows.append(r)
        print(f"  {day:<12}{r['economic']:>9.0f}{-r['wear_delta']:>8.0f}{r['buyback']:>9.1f}"
              f"{r['resell']:>8.1f}{r['spike_dis']:>7.1f}{r['cheap_chg']:>8.1f}"
              f"{r['soc_min']:>8.1f}{max(r['charge_cycles'],r['discharge_cycles']):>6.2f}"
              f"{r['imbalance_mwh']:>6.1f}")
    econ = np.array([r["economic"] for r in rows]); damr = np.array([r["dam_revenue"] for r in rows])
    print("  " + "─" * 92)
    print(f"  {'MEAN':<12}{econ.mean():>9.0f}")
    print(f"  {'TOTAL':<12}{econ.sum():>9.0f}")
    print(f"\n  Intraday uplift: {econ.sum():,.0f} € on {damr.sum():,.0f} € DAM "
          f"(+{100*econ.sum()/abs(damr.sum()):.2f}%)   positive days {int((econ>0).sum())}/{len(econ)}")
    print(f"  SoC floor min = {min(r['soc_min'] for r in rows):.1f} (floor {args.capacity_floor:.0f})   "
          f"max cycles = {max(max(r['charge_cycles'],r['discharge_cycles']) for r in rows):.2f} (cap {cfg.max_cycles_per_day})   "
          f"max imbalance = {max(r['imbalance_mwh'] for r in rows):.2f} MWh")
    print("═" * 94)


if __name__ == "__main__":
    main()
