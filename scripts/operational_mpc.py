#!/usr/bin/env python3
"""Operational XBID/SIDC battery controller — wear-aware rolling MPC.

Production-oriented controller for a battery that already holds committed
day-ahead (DAM) positions and Capacity-Market obligations, and wants EXTRA
profit on the continuous intraday market without violating any constraint.

It re-solves a rolling LP as XBID prices reveal, maximising

    Σ price_xbid[i] · q[i]   −   c_wear · physical_throughput(q)

over the committed net position ``q`` (starts at the DAM schedule), subject to
  * SoC ∈ [capacity-market floor, Emax] at all times,
  * per-slot power limit,
  * daily cycle cap (hard safety limit),
  * terminal SoE target,
and executes the incremental XBID trades ``x = q − q_dam`` at the quoted price
± half-spread. Because wear is charged on the *physical* dispatch (not on the
number of trades), the LP naturally performs all four operational strategies:

  1. Financial arbitrage (wear-free): buy back a DAM-sold discharge when XBID
     drops (raises q toward 0 → cash in + LESS wear), resell a DAM-bought
     charge when XBID spikes.
  2. RES-spike exploitation: discharge extra where XBID spikes, only while SoC
     stays above the capacity-market floor.
  3. Dynamic SoC / cheap charging: charge where XBID ≤ 0 (paid to charge),
     securing capacity-market SoC at minimal cost.
  4. Re-profiling: redistribute the hourly DAM energy across 15-min products to
     smooth the C-rate (add --crate-limit to cap per-slot power for thermal
     comfort).

Imbalance is zero by construction (only deliverable positions are committed).
To go live, replace ``observed_price`` / the fill model with the real XBID API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xbid_trader.data.dam_schedule_loader import load_dam_dataset
from xbid_trader.market.battery import BatteryConfig, soc_dispatch, cycle_usage


def solve_lp(price, soc_now, slot, committed, cfg, ch_used, dh_used,
             c_wear, soc_floor, crate_mwh):
    """Optimal remaining net dispatch maximising value − wear, from ``slot``."""
    import pulp
    H = len(price); dt = cfg.slot_duration_hours
    Emax = cfg.soc_max_mwh; Nc, Nd = cfg.eta_charge, cfg.eta_discharge
    P = min(cfg.power_mw, crate_mwh / dt) if crate_mwh else cfg.power_mw
    Cs = cfg.max_cycles_per_day

    m = pulp.LpProblem("mpc", pulp.LpMaximize)
    e  = {h: pulp.LpVariable(f"e{h}", soc_floor, Emax) for h in range(slot, H)}
    pc = {h: pulp.LpVariable(f"pc{h}", 0, P) for h in range(slot, H)}   # charge power
    pd = {h: pulp.LpVariable(f"pd{h}", 0, P) for h in range(slot, H)}   # discharge power

    # value of the schedule at XBID prices − wear on physical storage throughput
    value = pulp.lpSum(price[h] * (pd[h] - pc[h]) * dt for h in range(slot, H))
    wear  = pulp.lpSum(c_wear * ((1.0 / Nd) * pd[h] + Nc * pc[h]) * dt
                       for h in range(slot, H))
    m += value - wear

    prev = soc_now
    for h in range(slot, H):
        m += e[h] == prev + (Nc * pc[h] - (1.0 / Nd) * pd[h]) * dt
        prev = e[h]
    m += pulp.lpSum(Nc * pc[h] * dt for h in range(slot, H)) <= (Cs - ch_used) * Emax
    m += pulp.lpSum((1.0 / Nd) * pd[h] * dt for h in range(slot, H)) <= (Cs - dh_used) * Emax
    if H - 1 >= slot:
        m += e[H - 1] == cfg.soc_target_mwh
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    tgt = np.array(committed, dtype=float)
    if pulp.LpStatus[m.status] == "Optimal":
        for h in range(slot, H):
            tgt[h] = (pd[h].value() - pc[h].value()) * dt
    return tgt


def run_day(dam_net, dam_price, rev_final, cfg, *, c_wear, soc_floor,
            reveal_window, action_window, half_spread, fee, reopt_every,
            crate_mwh, reveal_noise, rng):
    H = len(dam_price); dt = cfg.slot_duration_hours
    committed = np.array(dam_net, dtype=float)
    side = np.zeros(H, dtype=int)
    cash = 0.0; ch_used = dh_used = 0.0
    strat = {"buyback": 0.0, "resell": 0.0, "spike_dis": 0.0, "cheap_chg": 0.0}

    def observed_price(t):
        idx = np.arange(H)
        prog = np.clip((t - (idx - reveal_window)) / reveal_window, 0.0, 1.0)
        noise = reveal_noise * np.abs(dam_price) * np.sqrt(prog * (1 - prog)) * rng.standard_normal(H)
        return dam_price + prog * rev_final + noise    # XBID prices can be negative

    def soc_after(upto):
        if upto <= 0: return cfg.soc_initial_mwh
        sp, _, _ = soc_dispatch(committed[:upto], cfg)
        return float(sp[-1]) if len(sp) else cfg.soc_initial_mwh

    target = committed.copy()
    for t in range(H):
        px = observed_price(t)
        if t % reopt_every == 0:
            target = solve_lp(px, soc_after(t), t, committed, cfg,
                              ch_used, dh_used, c_wear, soc_floor, crate_mwh)
        for i in range(t, min(H, t + action_window)):
            d = target[i] - committed[i]
            if abs(d) < 0.05:
                continue
            intended = 1 if d > 0 else -1
            if side[i] != 0 and side[i] != intended:
                continue

            # Exact feasibility given the rest of the committed schedule: the
            # most we can change slot i without pushing any later SoC out of
            # [floor, Emax] (keeps the whole position deliverable → imbalance 0).
            soc_path, _, _ = soc_dispatch(committed, cfg)
            suf_min = np.minimum.accumulate(soc_path[i:][::-1])[::-1]
            suf_max = np.maximum.accumulate(soc_path[i:][::-1])[::-1]
            head_dis = max(suf_min[0] - soc_floor, 0.0) * cfg.eta_discharge
            head_chg = max(cfg.soc_max_mwh - suf_max[0], 0.0) / cfg.eta_charge
            if d > 0:
                d = min(d, head_dis)
            else:
                d = -min(-d, head_chg)
            if abs(d) < 0.05:
                continue
            new_i = committed[i] + d

            add_ch = cfg.eta_charge * max(-d, 0.0) / cfg.energy_mwh
            add_dh = (max(d, 0.0) / cfg.eta_discharge) / cfg.energy_mwh
            if ch_used + add_ch > cfg.max_cycles_per_day + 1e-9: continue
            if dh_used + add_dh > cfg.max_cycles_per_day + 1e-9: continue

            fill = (px[i] - half_spread) if d > 0 else (px[i] + half_spread)
            cash += fill * d - fee * abs(d)
            if d > 0 and committed[i] < 0:   strat["resell"]   += d
            elif d > 0:                      strat["spike_dis"] += d
            elif d < 0 and committed[i] > 0: strat["buyback"]  += -d
            else:                            strat["cheap_chg"] += -d
            committed[i] = new_i; side[i] = intended
            ch_used += add_ch; dh_used += add_dh

    soc_path, delivered, imbalance = soc_dispatch(committed, cfg)
    # marginal wear vs the DAM baseline (financial arbitrage can make this < 0)
    thr = lambda q: np.sum(np.where(q > 0, q, 0)) / cfg.eta_discharge + cfg.eta_charge * np.sum(np.where(q < 0, -q, 0))
    wear = cfg.degradation_eur_per_mwh * (thr(committed) - thr(np.asarray(dam_net)))
    return {
        "cash": cash, "economic": cash - wear, "wear_delta": wear,
        "imbalance_mwh": float(np.abs(imbalance).sum()),
        "soc_final": float(soc_path[-1]) if len(soc_path) else cfg.soc_initial_mwh,
        "soc_min": float(soc_path.min()) if len(soc_path) else cfg.soc_initial_mwh,
        "charge_cycles": ch_used, "discharge_cycles": dh_used,
        "dam_revenue": float(np.sum(dam_price * dam_net)), **strat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Operational XBID battery MPC.")
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--month", default="2026-04")
    ap.add_argument("--max-cycles", type=float, default=1.5)
    ap.add_argument("--capacity-floor", type=float, default=None,
                    help="Capacity-Market SoC floor (MWh); default = technical Emin")
    ap.add_argument("--wear-eur-mwh", type=float, default=2.0, help="battery wear cost €/MWh throughput")
    ap.add_argument("--id-revision-sigma", type=float, default=0.12)
    ap.add_argument("--reveal-window", type=int, default=24)
    ap.add_argument("--action-window", type=int, default=8)
    ap.add_argument("--half-spread", type=float, default=0.5)
    ap.add_argument("--fee-per-mwh", type=float, default=0.0)
    ap.add_argument("--reopt-every", type=int, default=4)
    ap.add_argument("--crate-mwh", type=float, default=0.0,
                    help="per-slot power cap for thermal re-profiling (MWh; 0 = off)")
    ap.add_argument("--spike-prob", type=float, default=0.06, help="prob. of a RES up-spike per product")
    ap.add_argument("--neg-prob", type=float, default=0.05, help="prob. of a negative-price event per product")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prices, provider = load_dam_dataset(args.xlsx, months=[args.month])
    cfg = BatteryConfig(); cfg.max_cycles_per_day = args.max_cycles
    cfg.degradation_eur_per_mwh = args.wear_eur_mwh
    floor = args.capacity_floor if args.capacity_floor is not None else cfg.soc_min_mwh
    rng = np.random.default_rng(args.seed)

    print("═" * 90)
    print(f"  Operational XBID MPC — {args.month}   P={cfg.power_mw:.0f}MW E={cfg.energy_mwh:.0f}MWh  "
          f"Cs={cfg.max_cycles_per_day}  floor={floor:.0f}MWh  wear={args.wear_eur_mwh}€/MWh")
    print("═" * 90)
    print(f"  {'Day':<12}{'ID PnL €':>9}{'wearΔ €':>8}{'buyback':>9}{'resell':>8}"
          f"{'spike':>7}{'chgNeg':>8}{'SOEmin':>8}{'cyc':>6}{'imb':>6}")
    print("  " + "─" * 88)
    rows = []
    for day in provider.days:
        sc = provider.get_scenario(day)
        dp = np.asarray(prices[day], float); dn = np.asarray(sc.battery_dam_schedule, float)
        n = len(dp); rho = 0.8; z = rng.standard_normal(n); rev = np.zeros(n)
        for i in range(1, n): rev[i] = rho * rev[i-1] + np.sqrt(1-rho*rho) * z[i]
        rev_final = rev * args.id_revision_sigma * np.abs(dp)
        # Realistic XBID microstructure: RES-driven UP spikes + negative-price events
        spike = rng.random(n) < args.spike_prob
        rev_final[spike] += rng.uniform(0.5, 2.0, spike.sum()) * np.abs(dp[spike])
        neg = rng.random(n) < args.neg_prob
        final = dp + rev_final
        final[neg] = rng.uniform(-30.0, 3.0, neg.sum())
        rev_final = final - dp
        r = run_day(dn, dp, rev_final, cfg, c_wear=args.wear_eur_mwh, soc_floor=floor,
                    reveal_window=args.reveal_window, action_window=args.action_window,
                    half_spread=args.half_spread, fee=args.fee_per_mwh,
                    reopt_every=args.reopt_every, crate_mwh=args.crate_mwh,
                    reveal_noise=0.3, rng=rng)
        r["day"] = day; rows.append(r)
        print(f"  {day:<12}{r['economic']:>9.0f}{-r['wear_delta']:>8.0f}{r['buyback']:>9.1f}"
              f"{r['resell']:>8.1f}{r['spike_dis']:>7.1f}{r['cheap_chg']:>8.1f}"
              f"{r['soc_min']:>8.1f}{max(r['charge_cycles'],r['discharge_cycles']):>6.2f}"
              f"{r['imbalance_mwh']:>6.1f}")
    econ = np.array([r["economic"] for r in rows]); damr = np.array([r["dam_revenue"] for r in rows])
    print("  " + "─" * 88)
    print(f"  {'MEAN':<12}{econ.mean():>9.0f}")
    print(f"  {'TOTAL':<12}{econ.sum():>9.0f}")
    print(f"\n  Intraday uplift: {econ.sum():,.0f} € on {damr.sum():,.0f} € DAM revenue "
          f"(+{100*econ.sum()/abs(damr.sum()):.2f}%)   positive days {int((econ>0).sum())}/{len(econ)}")
    print(f"  SoC floor respected: min over month = {min(r['soc_min'] for r in rows):.1f} MWh (floor {floor:.0f})   "
          f"max imbalance = {max(r['imbalance_mwh'] for r in rows):.2f} MWh")
    print("═" * 90)


if __name__ == "__main__":
    main()
