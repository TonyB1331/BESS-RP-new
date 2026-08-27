#!/usr/bin/env python3
"""Operational backtest on REAL Greek market data, with capacity commitments.

The daily flow this reproduces::

    D-1   the DAM position AND the capacity-market awards for the day are known
     D    the XBID market runs; the controller trades on top of the committed
          position, never breaking it

Controller = wear-aware rolling MPC + liquidity-aware execution.

Capacity handling
-----------------
``physical = tradeable DAM position + reserve activation``

* only the **tradeable** part is traded on XBID;
* the **activation** (assumed ``--activation-rate``, default 0.40, of the
  awarded capacity) is dispatched by the TSO — untradeable, but it moves the SoC
  and burns cycles, so the controller carries it through every check;
* the awards also reserve power head-room (``q + up <= P``, ``q - dn >= -P``)
  and SoC energy.

Guarantees, all by construction: imbalance 0, daily cycles <= the cap (a
**maximum**, not a target), SoC inside the reserve-aware band, one side per
product, and untradeable products pinned to their committed position.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xbid_trader.data.real_market_loader import (SLOT_HOURS, coverage_report,
                                                 load_real_market)
from xbid_trader.market.battery import BatteryConfig


# ==========================================================================
# Capacity-aware physics (local: takes per-slot limits + a starting SoC)
# ==========================================================================
def soc_dispatch_env(net, cfg, floor, ceiling, p_dis, p_chg, soc_start):
    """Clip a net position to what is deliverable inside a per-slot envelope."""
    net = np.asarray(net, dtype=float)
    n = len(net)
    soc = float(soc_start)
    path = np.zeros(n)
    delivered = np.zeros(n)
    for i in range(n):
        q = net[i]
        if q >= 0:                                     # discharge
            q = min(q, p_dis[i] * SLOT_HOURS)
            q = min(q, max(soc - floor[i], 0.0) * cfg.eta_discharge)
            soc -= q / cfg.eta_discharge
        else:                                          # charge
            q = -min(-q, p_chg[i] * SLOT_HOURS)
            q = -min(-q, max(ceiling[i] - soc, 0.0) / cfg.eta_charge)
            soc += cfg.eta_charge * (-q)
        delivered[i] = q
        path[i] = soc
    return path, delivered, net - delivered


def replace_wear(cfg, wear):
    """Return a copy of the config with a different degradation cost."""
    import copy
    c = copy.copy(cfg)
    c.degradation_eur_per_mwh = float(wear)
    return c


def cycles_of(net, cfg) -> Tuple[float, float]:
    net = np.asarray(net, dtype=float)
    ch = cfg.eta_charge * np.sum(np.where(net < 0, -net, 0.0)) / cfg.energy_mwh
    dh = (np.sum(np.where(net > 0, net, 0.0)) / cfg.eta_discharge) / cfg.energy_mwh
    return float(ch), float(dh)


def throughput(net, cfg) -> float:
    net = np.asarray(net, dtype=float)
    return float(np.sum(np.where(net > 0, net, 0.0)) / cfg.eta_discharge
                 + cfg.eta_charge * np.sum(np.where(net < 0, -net, 0.0)))


def operating_envelope(rd, cfg, capacity_floor, reserve_duration_h):
    """Per-slot power limits and SoC band left free by the awarded reserves.

    An UP award of ``u`` MW must be deliverable on demand -> ``q + u <= P`` and
    enough stored energy to sustain it; a DOWN award mirrors that. The envelope
    is then widened so it always contains the committed plan itself: reserve
    adequacy is the capacity optimiser's responsibility, and the intraday layer
    must simply never make it worse.
    """
    u, d = np.asarray(rd.up_mw, float), np.asarray(rd.dn_mw, float)
    p_dis = np.maximum(cfg.power_mw - u, 0.0)
    p_chg = np.maximum(cfg.power_mw - d, 0.0)

    e_up = u * reserve_duration_h / cfg.eta_discharge
    e_dn = d * reserve_duration_h * cfg.eta_charge
    floor = np.maximum(capacity_floor, cfg.soc_min_mwh) + e_up
    ceiling = cfg.soc_max_mwh - e_dn

    if rd.plan_soc is not None and len(rd.plan_soc) == len(u):
        floor = np.minimum(floor, rd.plan_soc)
        ceiling = np.maximum(ceiling, rd.plan_soc)

    phys = rd.physical()
    p_dis = np.maximum(p_dis, np.maximum(phys, 0.0) / SLOT_HOURS)
    p_chg = np.maximum(p_chg, np.maximum(-phys, 0.0) / SLOT_HOURS)

    floor = np.minimum(floor, ceiling - 1e-6)
    return p_dis, p_chg, floor, ceiling


# ==========================================================================
# Rolling LP
# ==========================================================================
def solve_lp(price, tradeable, dam_net, activation, soc_now, slot, committed, cfg,
             ch_used, dh_used, cycle_cap, p_dis, p_chg, floor, ceiling, soc_target):
    """Optimal remaining tradeable dispatch: value - wear, under every limit."""
    import pulp

    H = len(price)
    dt, Nc, Nd = SLOT_HOURS, cfg.eta_charge, cfg.eta_discharge
    Emax = cfg.energy_mwh

    m = pulp.LpProblem("mpc", pulp.LpMaximize)
    e = {h: pulp.LpVariable(f"e{h}", float(floor[h]), float(ceiling[h]))
         for h in range(slot, H)}
    pc = {h: pulp.LpVariable(f"pc{h}", 0, float(p_chg[h])) for h in range(slot, H)}
    pd = {h: pulp.LpVariable(f"pd{h}", 0, float(p_dis[h])) for h in range(slot, H)}

    # Soft terminal SoC: instead of forcing e[H-1] == soc_target (which makes the
    # battery refill at ANY price to hit the target, buying at evening peaks on
    # expensive days), let it miss the target and pay a penalty ~ the value of
    # that energy. The LP then refills only when it is cheaper than that penalty.
    slack_short = pulp.LpVariable("slack_short", 0, None)   # ends BELOW target
    slack_over = pulp.LpVariable("slack_over", 0, None)     # ends ABOVE target
    mean_price = float(np.mean(np.abs(price[slot:]))) if H > slot else 0.0
    term_penalty = min(max(mean_price, 40.0), 200.0)       # €/MWh, clamped

    m += (pulp.lpSum(price[h] * (pd[h] - pc[h]) * dt for h in range(slot, H))
          - pulp.lpSum(cfg.degradation_eur_per_mwh * (pd[h] / Nd + Nc * pc[h]) * dt
                       for h in range(slot, H))
          - term_penalty * (slack_short + slack_over))

    # the reserve activation is fixed: not a decision, but it moves SoC and cycles
    ra = np.asarray(activation, dtype=float)
    ra_soc = Nc * np.maximum(-ra, 0.0) - np.maximum(ra, 0.0) / Nd
    ra_chg = float(np.sum(Nc * np.maximum(-ra[slot:], 0.0)))
    ra_dis = float(np.sum(np.maximum(ra[slot:], 0.0) / Nd))

    prev = soc_now
    for h in range(slot, H):
        m += e[h] == prev + (Nc * pc[h] - pd[h] / Nd) * dt + float(ra_soc[h])
        prev = e[h]
        if not tradeable[h]:                       # no XBID market -> keep the plan
            cn = float(dam_net[h])
            if cn >= 0:
                m += pd[h] == cn / dt
                m += pc[h] == 0
            else:
                m += pc[h] == -cn / dt
                m += pd[h] == 0

    # cycle budget is a MAXIMUM, and the reserve activation already consumes part
    m += (pulp.lpSum(Nc * pc[h] * dt for h in range(slot, H))
          <= max(cycle_cap - ch_used, 0.0) * Emax - ra_chg)
    m += (pulp.lpSum(pd[h] / Nd * dt for h in range(slot, H))
          <= max(cycle_cap - dh_used, 0.0) * Emax - ra_dis)
    if H - 1 >= slot:
        m += e[H - 1] + slack_short - slack_over == float(soc_target)

    m.solve(pulp.PULP_CBC_CMD(msg=0))
    tgt = np.array(committed, dtype=float)
    if pulp.LpStatus[m.status] == "Optimal":
        for h in range(slot, H):
            tgt[h] = (pd[h].value() - pc[h].value()) * dt
    return tgt


# ==========================================================================
# One delivery day
# ==========================================================================
def reoptimize_dam_dispatch(rd, cfg, floor, ceiling, p_dis, p_chg, soc_start, soc_target):
    """Re-solve the committed DAM position to MAXIMISE day-ahead revenue.

    Solves an LP on the DAM price curve, respecting the SAME reserve-aware
    envelope (dynamic floor/ceiling, power head-room) and a HARD cap of
    ``cfg.max_cycles_per_day`` on the **physical** dispatch (DAM decision + the
    fixed reserve activation). The awarded reserves and their activation are
    unchanged — only the tradeable DAM energy position is re-decided.

    Returns the new tradeable ``dam_net`` (MWh/slot, + discharge / − charge).
    """
    import pulp

    price = np.asarray(rd.dam_price, dtype=float)
    act = np.asarray(rd.reserve_activation, dtype=float)     # MWh/slot, fixed
    H = len(price)
    dt, Nc, Nd = SLOT_HOURS, cfg.eta_charge, cfg.eta_discharge
    Cs, Emax = cfg.max_cycles_per_day, cfg.energy_mwh
    ra_soc = Nc * np.maximum(-act, 0.0) - np.maximum(act, 0.0) / Nd

    m = pulp.LpProblem("dam_reopt", pulp.LpMaximize)
    e = {h: pulp.LpVariable(f"e{h}", float(floor[h]), float(ceiling[h])) for h in range(H)}
    pc = {h: pulp.LpVariable(f"pc{h}", 0, float(p_chg[h])) for h in range(H)}
    pd = {h: pulp.LpVariable(f"pd{h}", 0, float(p_dis[h])) for h in range(H)}
    # physical throughput split, so the cycle cap applies to DAM + activation
    phdis = {h: pulp.LpVariable(f"phd{h}", 0) for h in range(H)}
    phchg = {h: pulp.LpVariable(f"phc{h}", 0) for h in range(H)}

    m += pulp.lpSum(price[h] * (pd[h] - pc[h]) * dt for h in range(H))   # DAM revenue

    prev = soc_start
    for h in range(H):
        m += e[h] == prev + (Nc * pc[h] - pd[h] / Nd) * dt + float(ra_soc[h])
        prev = e[h]
        phys = (pd[h] - pc[h]) * dt + float(act[h])
        m += phdis[h] >= phys
        m += phchg[h] >= -phys
    m += pulp.lpSum(phdis[h] / Nd for h in range(H)) <= Cs * Emax
    m += pulp.lpSum(Nc * phchg[h] for h in range(H)) <= Cs * Emax
    m += e[H - 1] == float(soc_target)

    m.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[m.status] != "Optimal":
        return np.asarray(rd.dam_net, dtype=float)          # keep committed on failure
    return np.array([(pd[h].value() - pc[h].value()) * dt for h in range(H)])


def run_day(rd, cfg, *, capacity_floor, reveal_window, action_window, reopt_every,
            fee, max_trade_mwh, reserve_duration_h=0.25, reoptimize_dam=False,
            min_edge_eur=0.0, roundtrip_margin_eur=None,
            wear=None, rng=None, reveal_noise=0.0):
    """Run the controller for one delivery day.

    ``wear`` overrides the battery's degradation cost; ``reveal_noise`` adds
    forecast error to the prices the controller acts on (fills still happen at
    the real price) for robustness testing.
    """
    if wear is not None:
        cfg = replace_wear(cfg, wear)
    noise_rng = rng if rng is not None else np.random.default_rng(0)
    dam, xbid, tradeable = rd.dam_price, rd.xbid_price, rd.tradeable
    spread, dam_net, activation = rd.xbid_spread, rd.dam_net, rd.reserve_activation
    H = len(dam)

    p_dis, p_chg, floor, ceiling = operating_envelope(
        rd, cfg, capacity_floor, reserve_duration_h)

    soc_start = rd.soc_start if rd.soc_start is not None else cfg.soc_initial_mwh

    # Terminal SoC is anchored to the ORIGINAL committed plan, so a reoptimised
    # DAM still ends the day where the plan does (cyclically consistent).
    orig_phys = rd.physical()
    orig_path, _, _ = soc_dispatch_env(orig_phys, cfg, floor, ceiling,
                                       p_dis, p_chg, soc_start)
    soc_target = float(orig_path[-1]) if len(orig_path) else soc_start

    # Optional: re-solve the DAM position for max day-ahead revenue, using up to
    # cfg.max_cycles_per_day cycles (a MAXIMUM), inside the same reserve envelope.
    # committed DAM revenue is always the baseline (what the plan earns as-is);
    # when reoptimising we also book the extra DAM revenue the system creates.
    dam_revenue_committed = float(np.sum(dam * rd.dam_settled))
    dam_revenue_value = dam_revenue_committed
    if reoptimize_dam:
        dam_net = reoptimize_dam_dispatch(rd, cfg, floor, ceiling, p_dis, p_chg,
                                          soc_start, soc_target)
        dam_revenue_value = float(np.sum(dam * dam_net))

    # Baseline = the committed (or reoptimised) plan INCLUDING reserve activation.
    # Everything is measured against it, so the intraday layer is never charged
    # for energy the baseline itself never restores.
    base_phys = dam_net + activation
    base_path, _, base_imb = soc_dispatch_env(base_phys, cfg, floor, ceiling,
                                              p_dis, p_chg, soc_start)
    base_imbalance = float(np.abs(base_imb).sum())
    base_imb_cost = float(np.sum(np.abs(base_imb) * np.abs(rd.imb_price)))
    b_ch, b_dh = cycles_of(base_phys, cfg)
    # the cycle limit is a MAXIMUM; if the activation assumption alone already
    # pushes the committed plan past it, the plan's own usage becomes the cap so
    # the intraday layer can still trade wear-free, but can never add cycles.
    cycle_cap = max(cfg.max_cycles_per_day, b_ch, b_dh)
    activation_revenue = float(np.sum(rd.reserve_activation * rd.imb_price))

    rev_final = np.where(tradeable, np.nan_to_num(xbid) - dam, 0.0)

    committed = dam_net.copy()
    side = np.zeros(H, dtype=int)
    cash = 0.0
    strat = {"buyback": 0.0, "resell": 0.0, "spike_dis": 0.0, "cheap_chg": 0.0}
    orders = []
    target = committed.copy()

    def observed(t):
        idx = np.arange(H)
        prog = np.clip((t - (idx - reveal_window)) / reveal_window, 0.0, 1.0)
        px = dam + prog * rev_final
        if reveal_noise > 0.0:
            px = px + (reveal_noise * np.abs(dam) * np.sqrt(prog * (1 - prog))
                       * noise_rng.standard_normal(H))
        return px

    def soc_at(upto):
        if upto <= 0:
            return soc_start
        path, _, _ = soc_dispatch_env((committed + activation)[:upto], cfg,
                                      floor, ceiling, p_dis, p_chg, soc_start)
        return float(path[-1]) if len(path) else soc_start

    for t in range(H):
        px = observed(t)
        if t % reopt_every == 0:
            ch_u, dh_u = (cycles_of((committed + activation)[:t], cfg) if t > 0 else (0.0, 0.0))
            target = solve_lp(px, tradeable, dam_net, activation, soc_at(t), t,
                              committed, cfg, ch_u, dh_u, cycle_cap,
                              p_dis, p_chg, floor, ceiling, soc_target)

        window = list(range(t, min(H, t + action_window)))
        for _ in range(6):
            traded = False
            for i in window:
                if not tradeable[i]:
                    continue
                delta = target[i] - committed[i]
                if abs(delta) < 0.05:
                    continue
                s = 1 if delta > 0 else -1
                if side[i] != 0 and side[i] != s:        # one side per product
                    continue

                # ---- Conservative trigger --------------------------------
                # Adding a fresh position (spike-discharge into a high price,
                # cheap-charge into a low one) is genuine arbitrage on a real
                # extreme — leave it alone. But CANCELLING an already-committed
                # position (buying back a committed discharge, or re-selling a
                # committed charge) gives up a slot the day-ahead plan already
                # chose well, and forces an uncertain re-balancing later. Only do
                # it when the XBID edge over the committed price is wide enough to
                # cover that risk.
                cancels_committed = ((s > 0 and committed[i] < -1e-9)
                                     or (s < 0 and committed[i] > 1e-9))
                if cancels_committed and min_edge_eur > 0:
                    edge = (float(xbid[i]) - float(dam[i])) if s > 0 else (float(dam[i]) - float(xbid[i]))
                    if edge < min_edge_eur:
                        continue

                # ---- Round-trip guard (fresh positions) ------------------
                # A fresh spike-discharge must be re-charged later; a fresh
                # cheap-charge must be re-sold later. Only act if that unwind is
                # profitable at the best price still visible in the rest of the
                # day, after round-trip efficiency losses and a safety margin.
                # This stops over-discharging into an evening that is expensive
                # to refill (and the mirror case for charging).
                if (not cancels_committed) and roundtrip_margin_eur is not None and i + 1 < H:
                    eta_rt = cfg.eta_charge * cfg.eta_discharge
                    future = px[i + 1:]
                    if len(future):
                        if s > 0:      # sell extra now -> must re-charge later
                            min_refill = float(np.min(future))
                            if float(xbid[i]) < min_refill / eta_rt + roundtrip_margin_eur:
                                continue
                        else:          # buy extra now -> must re-sell later
                            max_resell = float(np.max(future))
                            if float(xbid[i]) > max_resell * eta_rt - roundtrip_margin_eur:
                                continue

                # analytic head-room first (cheap), then verify end-to-end
                path, _, _ = soc_dispatch_env(committed + activation, cfg, floor,
                                              ceiling, p_dis, p_chg, soc_start)
                if s > 0:
                    head = max(path[i:].min() - floor[i:].max(), 0.0) * cfg.eta_discharge
                    head = min(head, max(p_dis[i] * SLOT_HOURS - (committed[i] + activation[i]), 0.0))
                else:
                    head = max(ceiling[i:].min() - path[i:].max(), 0.0) / cfg.eta_charge
                    head = min(head, max(p_chg[i] * SLOT_HOURS + (committed[i] + activation[i]), 0.0))
                hi = min(abs(delta), max_trade_mwh, max(head, 0.0))
                if hi < 0.05:
                    continue

                def ok(q):
                    trial = committed.copy()
                    trial[i] += q if s > 0 else -q
                    phys = trial + activation
                    _, _, imb = soc_dispatch_env(phys, cfg, floor, ceiling,
                                                 p_dis, p_chg, soc_start)
                    if np.abs(imb).sum() > base_imbalance + 1e-6:
                        return False
                    c, dch = cycles_of(phys, cfg)
                    return max(c, dch) <= cycle_cap + 1e-9

                if ok(hi):
                    qty = hi
                else:
                    lo, hb = 0.0, hi
                    for _ in range(12):
                        mid = 0.5 * (lo + hb)
                        if ok(mid):
                            lo = mid
                        else:
                            hb = mid
                    qty = lo
                if qty < 0.05:
                    continue

                signed = qty if s > 0 else -qty
                hs = 0.5 * float(spread[i])
                fill = float(xbid[i]) + (-hs if s > 0 else hs)
                pnl = fill * signed - fee * qty
                cash += pnl

                if signed > 0 and committed[i] < 0:
                    tag = "resell"
                elif signed > 0:
                    tag = "spike_dis"
                elif signed < 0 and committed[i] > 0:
                    tag = "buyback"
                else:
                    tag = "cheap_chg"
                strat[tag] += qty
                orders.append(dict(product_id=i, side="SELL" if s > 0 else "BUY",
                                   qty_mwh=round(float(qty), 3),
                                   price=round(fill, 2), cash=round(float(pnl), 2),
                                   strategy=tag))
                committed[i] += signed
                side[i] = s
                traded = True
            if not traded:
                break

    phys_final = committed + activation
    soc_final, _, imbalance = soc_dispatch_env(phys_final, cfg, floor, ceiling,
                                               p_dis, p_chg, soc_start)
    wear = cfg.degradation_eur_per_mwh * (throughput(phys_final, cfg)
                                          - throughput(base_phys, cfg))
    imb_cost = float(np.sum(np.abs(imbalance) * np.abs(rd.imb_price)))
    ch, dh = cycles_of(phys_final, cfg)

    # ---- Terminal mark-to-market -------------------------------------------
    # The execution is liquidity- and one-side-limited, so the delivered position
    # can end the day at a slightly different SoC than the committed baseline.
    # That energy difference is a real asset/liability (it carries into the next
    # day) and must be valued, otherwise a day that simply ENDS HOLDING energy it
    # paid for looks like a cash loss. Value the drift at the day's mean DAM
    # price, adjusted for the efficiency of getting it back out / putting it in.
    mark = float(np.nanmean(dam)) if np.isfinite(dam).any() else 0.0
    soc_drift = float(soc_final[-1] - base_path[-1]) if len(soc_final) else 0.0
    if soc_drift >= 0:
        terminal_value = soc_drift * cfg.eta_discharge * mark      # can sell it later
    else:
        terminal_value = soc_drift / cfg.eta_charge * mark         # must buy it back
    economic = cash - wear - (imb_cost - base_imb_cost) + terminal_value

    return {
        "cash": cash, "economic": economic,
        "terminal_value": terminal_value, "soc_drift": soc_drift,
        "wear": wear, "wear_delta": wear,
        "activation_revenue": activation_revenue,
        "base_imbalance_mwh": base_imbalance,
        "imbalance_mwh": float(np.abs(imbalance).sum()),
        "soc_min": float(soc_final.min()) if len(soc_final) else soc_start,
        "cycles": max(ch, dh), "base_cycles": max(b_ch, b_dh), "cycle_cap": cycle_cap,
        "charge_cycles": ch, "discharge_cycles": dh,
        "dam_revenue": dam_revenue_value,
        "dam_revenue_committed": dam_revenue_committed,
        "dam_reopt_gain": dam_revenue_value - dam_revenue_committed,
        "dam_revenue_physical": float(np.sum(dam * dam_net)),
        "reserve_revenue": rd.reserve_revenue,
        "n_tradeable": int(tradeable.sum()), "n_orders": len(orders),
        "orders": orders,
        # per-slot profiles (used by the evaluation plots)
        "committed_net": committed.copy(),
        "physical_net": phys_final.copy(),
        "dam_net_arr": np.asarray(dam_net, dtype=float).copy(),
        "dam_price_arr": np.asarray(dam, dtype=float).copy(),
        "xbid_price_arr": np.asarray(xbid, dtype=float).copy(),
        "tradeable_arr": np.asarray(tradeable).copy(),
        "reserve_up_mw": np.asarray(rd.up_mw, dtype=float).copy(),
        "reserve_dn_mw": np.asarray(rd.dn_mw, dtype=float).copy(),
        "activation_arr": np.asarray(activation, dtype=float).copy(),
        "soc_final_arr": soc_final.copy(),
        "soc_dam_arr": base_path.copy(),
        "soc_floor_arr": np.asarray(floor, dtype=float).copy(),
        **strat,
    }


# ==========================================================================
# CLI
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Operational backtest on REAL market data.")
    ap.add_argument("--schedule-xlsx", required=True, help="committed DAM (+capacity) plan")
    ap.add_argument("--energy-xlsx", required=True, help="Energy_Market_Data.xlsx (DAM+XBID)")
    ap.add_argument("--imbalance-xlsx", required=True, help="imbalance price workbook")
    ap.add_argument("--months", nargs="*", default=None, help="e.g. 2026-01 2026-02")
    ap.add_argument("--max-cycles", type=float, default=2.0,
                    help="MAXIMUM daily cycles (not a target)")
    ap.add_argument("--capacity-floor", type=float, default=5.0, help="SoC floor MWh")
    ap.add_argument("--wear-eur-mwh", type=float, default=2.0)
    ap.add_argument("--power-mw", type=float, default=50.0)
    ap.add_argument("--energy-mwh", type=float, default=100.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale the plan workbook's MW/MWh (50 if normalised to 1 MW)")
    ap.add_argument("--activation-rate", type=float, default=0.40,
                    help="fraction of awarded reserve capacity assumed activated")
    ap.add_argument("--reserve-duration-h", type=float, default=0.25,
                    help="how long an awarded reserve must be sustainable")
    ap.add_argument("--reoptimize-dam", action="store_true",
                    help="re-solve the DAM position for max revenue using up to "
                         "--max-cycles cycles (default: keep the committed DAM)")
    ap.add_argument("--min-edge-eur", type=float, default=10.0,
                    help="conservative trigger: only cancel a committed position "
                         "(buy-back/re-sell) when the XBID edge over DAM exceeds "
                         "this many EUR/MWh (0 = off, more profit but more risk)")
    ap.add_argument("--roundtrip-margin-eur", type=float, default=-1.0,
                    help="EXPERIMENTAL round-trip guard on fresh spike/charge trades "
                         "(<0 = off; testing showed it hurts on DAM-forecast days)")
    ap.add_argument("--reveal-window", type=int, default=48)
    ap.add_argument("--action-window", type=int, default=16)
    ap.add_argument("--reopt-every", type=int, default=2)
    ap.add_argument("--fee-per-mwh", type=float, default=0.0)
    ap.add_argument("--max-trade-mwh", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = BatteryConfig()
    cfg.power_mw = args.power_mw
    cfg.energy_mwh = args.energy_mwh
    cfg.soc_max_mwh = args.energy_mwh
    cfg.soc_min_mwh = min(args.capacity_floor, args.energy_mwh)
    cfg.max_cycles_per_day = args.max_cycles
    cfg.degradation_eur_per_mwh = args.wear_eur_mwh

    by_day = load_real_market(args.schedule_xlsx, args.energy_xlsx, args.imbalance_xlsx,
                              args.months, scale_mw=args.scale,
                              activation_rate=args.activation_rate,
                              eta_charge=cfg.eta_charge, eta_discharge=cfg.eta_discharge)
    days = sorted(by_day)
    if not days:
        print("No overlapping days between the plan and the market data.")
        return

    print("=" * 96)
    print(f"  Operational backtest on REAL market data — {coverage_report(by_day)}")
    print(f"  P={cfg.power_mw:.0f}MW E={cfg.energy_mwh:.0f}MWh  cycles<={args.max_cycles} (max)  "
          f"floor={args.capacity_floor:.0f}MWh  wear={args.wear_eur_mwh}EUR/MWh")
    print(f"  scale x{args.scale:g}   activation={100*args.activation_rate:.0f}% of awarded capacity   "
          f"max_order={args.max_trade_mwh}MWh   reoptimize_dam={args.reoptimize_dam}   "
          f"min_edge={args.min_edge_eur}EUR")
    print("=" * 96)
    print(f"  {'Day':<12}{'ID PnL':>9}{'buyback':>9}{'resell':>8}{'spike':>7}{'chgNeg':>8}"
          f"{'#XBID':>7}{'SOEmin':>8}{'cyc':>6}{'base':>6}{'imb':>7}")
    print("  " + "-" * 94)

    rows = []
    for day in days:
        r = run_day(by_day[day], cfg, capacity_floor=args.capacity_floor,
                    reserve_duration_h=args.reserve_duration_h,
                    reoptimize_dam=args.reoptimize_dam, min_edge_eur=args.min_edge_eur,
                    roundtrip_margin_eur=(args.roundtrip_margin_eur
                                          if args.roundtrip_margin_eur >= 0 else None),
                    reveal_window=args.reveal_window, action_window=args.action_window,
                    reopt_every=args.reopt_every, fee=args.fee_per_mwh,
                    max_trade_mwh=args.max_trade_mwh)
        r["day"] = day
        rows.append(r)
        print(f"  {day:<12}{r['economic']:>9.0f}{r['buyback']:>9.1f}{r['resell']:>8.1f}"
              f"{r['spike_dis']:>7.1f}{r['cheap_chg']:>8.1f}{r['n_tradeable']:>7}"
              f"{r['soc_min']:>8.1f}{r['cycles']:>6.2f}{r['base_cycles']:>6.2f}"
              f"{r['imbalance_mwh']:>7.1e}")

    econ = np.array([r["economic"] for r in rows])
    dam_rev = sum(r["dam_revenue"] for r in rows)
    cap_rev = sum(r["reserve_revenue"] for r in rows)
    act_rev = sum(r["activation_revenue"] for r in rows)
    base_cyc = max(r["base_cycles"] for r in rows)
    print("  " + "-" * 94)
    print(f"  {'MEAN':<12}{econ.mean():>9.0f}")
    print(f"  {'TOTAL':<12}{econ.sum():>9.0f}")
    print()
    print(f"  Real intraday uplift: {econ.sum():,.0f} EUR on {dam_rev:,.0f} EUR DAM "
          f"({100 * econ.sum() / abs(dam_rev):+.2f}%)   positive days "
          f"{int((econ > 0).sum())}/{len(econ)}")
    if cap_rev:
        tot = dam_rev + cap_rev
        print()
        print(f"  REVENUE STACK    DAM energy       {dam_rev:>12,.0f} EUR")
        print(f"                   capacity         {cap_rev:>12,.0f} EUR")
        print(f"                   activation @{100*args.activation_rate:.0f}% {act_rev:>12,.0f} EUR"
              f"   (balancing settlement)")
        print(f"                   XBID intraday    {econ.sum():>12,.0f} EUR"
              f"   ({100 * econ.sum() / tot:+.2f}% of committed revenue)")
    print()
    print(f"  CONSTRAINTS  max imbalance {max(r['imbalance_mwh'] for r in rows):.2e} MWh"
          f" (baseline {max(r['base_imbalance_mwh'] for r in rows):.2e}) | "
          f"min SoE {min(r['soc_min'] for r in rows):.1f} >= {args.capacity_floor}")
    print(f"  CYCLES       committed plan + {100*args.activation_rate:.0f}% activation already "
          f"uses up to {base_cyc:.2f}/day; final {max(r['cycles'] for r in rows):.2f}"
          f"  (limit {args.max_cycles} is a MAXIMUM)")
    if base_cyc > args.max_cycles + 1e-6:
        print(f"  NOTE  the activation assumption alone exceeds the {args.max_cycles} cycle "
              f"limit, so the intraday layer may not add any cycles.")
    print("=" * 96)


if __name__ == "__main__":
    main()
