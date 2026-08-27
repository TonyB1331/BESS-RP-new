#!/usr/bin/env python3
"""Battery intraday-arbitrage evaluation on the real DAM dataset.

Rolls a policy over the given month (default April 2026), starting each day
from the *given* DAM schedule and re-optimising it on the simulated XBID
order book.  For every 15-minute delivery product it prints the orders that
were executed for that quarter (requirement 5), then reports per-day and
aggregate battery KPIs.

Policies
--------
    ppo        : a trained PPO model            (needs --model, --vecnorm)
    threshold  : fixed arbitrage band θ, δ      (heuristic baseline)
    dam_only   : never trades intraday          (validates the DAM baseline)
    random     : random valid actions           (sanity floor)

Examples
--------
    # April, PPO model, print executed orders per quarter
    python scripts/evaluate_battery.py --xlsx DATA.xlsx --month 2026-04 \
        --policy ppo --model out/best_model.zip --vecnorm out/vecnorm.pkl \
        --print-orders --day 2026-04-01

    # Quick check of the DAM baseline (no model needed)
    python scripts/evaluate_battery.py --xlsx DATA.xlsx --policy dam_only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from xbid_trader.data.dam_schedule_loader import load_dam_dataset
from xbid_trader.market.xbid_env import XBIDEnv, MAX_SLOTS
from xbid_trader.market.rl_observation import N_FEATURES, IDX_NET_POSITION, IDX_MID_PRICE, IDX_VALUE_REF
from xbid_trader.market.battery import BatteryConfig


# ──────────────────────────────────────────────────────────────────────────
# Policies
# ──────────────────────────────────────────────────────────────────────────

def _rolling_lp(price, soc_now, slot, committed, cfg, ch_used, dh_used, c_wear, floor):
    """Wear-aware rolling LP: optimal remaining net dispatch from ``slot``."""
    import pulp
    H = len(price); dt = cfg.slot_duration_hours
    Emax = cfg.soc_max_mwh; Nc, Nd = cfg.eta_charge, cfg.eta_discharge
    P = cfg.power_mw; Cs = cfg.max_cycles_per_day
    m = pulp.LpProblem("mpc", pulp.LpMaximize)
    e  = {h: pulp.LpVariable(f"e{h}", floor, Emax) for h in range(slot, H)}
    pc = {h: pulp.LpVariable(f"pc{h}", 0, P) for h in range(slot, H)}
    pd = {h: pulp.LpVariable(f"pd{h}", 0, P) for h in range(slot, H)}
    m += (pulp.lpSum(price[h] * (pd[h] - pc[h]) * dt for h in range(slot, H))
          - pulp.lpSum(c_wear * ((1/Nd)*pd[h] + Nc*pc[h]) * dt for h in range(slot, H)))
    prev = soc_now
    for h in range(slot, H):
        m += e[h] == prev + (Nc*pc[h] - (1/Nd)*pd[h]) * dt; prev = e[h]
    m += pulp.lpSum(Nc*pc[h]*dt for h in range(slot, H)) <= (Cs - ch_used) * Emax
    m += pulp.lpSum((1/Nd)*pd[h]*dt for h in range(slot, H)) <= (Cs - dh_used) * Emax
    if H - 1 >= slot: m += e[H-1] == cfg.soc_target_mwh
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    tgt = np.array(committed, float)
    if pulp.LpStatus[m.status] == "Optimal":
        for h in range(slot, H): tgt[h] = (pd[h].value() - pc[h].value()) * dt
    return tgt


def make_policy(name, env, model=None, vecnorm=None, theta=2.0, delta=0.5,
                c_wear=2.0, floor=None):
    """Return a ``predict(flat_obs) -> action`` for the chosen policy.

    Action = (target_net_frac in [-1,1], aggression in [0,1]) per slot.
    """
    n_act = MAX_SLOTS * 2
    p_slot = env.battery.power_per_slot_mwh

    def _net_frac(flat_obs):
        mat = np.asarray(flat_obs, dtype=np.float32).reshape(MAX_SLOTS, N_FEATURES)
        return np.clip(mat[:, IDX_NET_POSITION] / p_slot, -1.0, 1.0), mat

    if name == "ppo":
        if model is None:
            raise SystemExit("--policy ppo requires --model")

        def predict(flat_obs):
            obs = vecnorm.normalize_obs(flat_obs) if vecnorm is not None else flat_obs
            action, _ = model.predict(obs[None, :], deterministic=True)
            return np.asarray(action[0], dtype=np.float32)
        return predict

    if name == "dam_only":
        # Target == current (DAM) net → zero adjustment → no intraday trades.
        def predict(flat_obs):
            frac, _ = _net_frac(flat_obs)
            a = np.zeros(n_act, dtype=np.float32)
            a[0::2] = frac
            return a
        return predict

    if name == "threshold":
        # Heuristic arbitrage: target full discharge when the ID mid is well
        # above the energy value, full charge when well below, else hold DAM.
        band = float(theta)
        def predict(flat_obs):
            frac, mat = _net_frac(flat_obs)
            mid = mat[:, IDX_MID_PRICE]; vref = mat[:, IDX_VALUE_REF]
            tgt = frac.copy()
            tgt[mid > vref + band] = +1.0   # dear → discharge
            tgt[mid < vref - band] = -1.0   # cheap → charge
            a = np.zeros(n_act, dtype=np.float32)
            a[0::2] = tgt
            a[1::2] = delta
            return a
        return predict

    if name == "random":
        return lambda flat_obs: env.action_space.sample()

    if name == "mpc":
        # Rolling-horizon MPC against the live simulated order book. Reads the
        # observed mid per product (reveals gradually) and re-solves the
        # wear-aware LP each step; the env executes with exact feasibility.
        from xbid_trader.market.battery import soc_dispatch, cycle_usage
        cfg = env.battery
        soc_floor = cfg.soc_min_mwh if floor is None else floor

        def predict(flat_obs):
            n = env._n_slots
            slot = env._session.current_slot if env._session else 0
            net = env._obs_builder._net_position[:n].copy()
            mat = np.asarray(flat_obs, np.float32).reshape(MAX_SLOTS, N_FEATURES)
            price = mat[:n, IDX_MID_PRICE].copy()
            vref  = mat[:n, IDX_VALUE_REF]
            bad = price == 0
            price[bad] = vref[bad]
            sp, _, _ = soc_dispatch(net[:slot], cfg) if slot > 0 else (np.array([cfg.soc_initial_mwh]), None, None)
            soc_now = float(sp[-1]) if slot > 0 and len(sp) else cfg.soc_initial_mwh
            ch_used, dh_used = cycle_usage(net[:slot], cfg) if slot > 0 else (0.0, 0.0)
            tgt = _rolling_lp(price, soc_now, slot, net, cfg, ch_used, dh_used, c_wear, soc_floor)
            a = np.zeros(MAX_SLOTS * 2, np.float32)
            a[0:2*n:2] = np.clip(tgt / cfg.power_per_slot_mwh, -1.0, 1.0)
            a[1:2*n:2] = 0.3
            return a
        return predict

    raise SystemExit(f"unknown policy '{name}'")


# ──────────────────────────────────────────────────────────────────────────
# Rollout with per-quarter order printing
# ──────────────────────────────────────────────────────────────────────────

def slot_time(day: str, slot: int) -> str:
    base = datetime.strptime(day, "%Y-%m-%d")
    t0 = base + timedelta(minutes=15 * slot)
    t1 = t0 + timedelta(minutes=15)
    return f"{t0:%H:%M}-{t1:%H:%M}"


def rollout_day(env, predict, day: str, print_orders: bool = False) -> dict:
    """Run one day; optionally print executed orders per delivery quarter."""
    obs, info = env.reset()
    n = env._n_slots

    # fills_by_product[pid] = list of (side, qty, price)
    fills_by_product = {pid: [] for pid in range(n)}
    terminated = False
    slot = 0

    if print_orders:
        print(f"\n┌─ Executed intraday orders per delivery quarter — {day} "
              f"(battery {env.battery.power_mw:.0f} MW / {env.battery.energy_mwh:.0f} MWh)")

    while not terminated:
        action = predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Record this decision step's fills against their delivery product.
        for trade, sign in zip(info.get("agent_trades", []),
                               info.get("agent_signs", [])):
            side = "DISCHARGE/SELL" if sign > 0 else "CHARGE/BUY"
            fills_by_product[int(trade.product_id)].append(
                (side, float(trade.quantity), float(trade.price))
            )

        # The product that has just reached delivery (gate closed) is `slot`.
        if print_orders and slot < n:
            fills = fills_by_product.get(slot, [])
            hdr = f"│ {slot_time(day, slot)}  (product #{slot:>2})"
            if fills:
                dam = env._obs_builder._dam_position[slot]
                net = env._obs_builder._net_position[slot]
                print(f"{hdr}  DAM={dam:+.2f}  net={net:+.2f} MWh")
                for side, qty, px in fills:
                    print(f"│      • {side:<15} {qty:.2f} MWh @ {px:8.2f} €/MWh")
            # (quarters with no executed intraday order are left silent)
        slot += 1

    if print_orders:
        print("└─ end of day\n")

    return {k: info.get(k, 0.0) for k in (
        "ep_economic_pnl", "ep_realized_pnl", "ep_throughput_mwh",
        "ep_imbalance_mwh", "ep_imbalance_cost", "ep_soc_final",
        "ep_soc_target", "ep_charge_cycles", "ep_discharge_cycles",
        "ep_cycle_penalty", "ep_terminal_penalty",
    )}


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Battery intraday evaluation.")
    ap.add_argument("--xlsx", required=True, help="DAM results workbook (.xlsx)")
    ap.add_argument("--month", default="2026-04", help='e.g. "2026-04" (default) or "all"')
    ap.add_argument("--policy", default="dam_only",
                    choices=["ppo", "threshold", "dam_only", "random", "mpc"])
    ap.add_argument("--model", default=None, help="PPO .zip (for --policy ppo)")
    ap.add_argument("--vecnorm", default=None, help="VecNormalize .pkl (for --policy ppo)")
    ap.add_argument("--theta", type=float, default=2.0, help="threshold-policy band")
    ap.add_argument("--delta", type=float, default=0.5, help="threshold-policy aggressiveness")
    ap.add_argument("--day", default=None, help="restrict to a single day YYYY-MM-DD")
    ap.add_argument("--print-orders", action="store_true",
                    help="print executed orders for each 15-min delivery quarter")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--id-revision-sigma", type=float, default=0.12,
                    help="MUST match the value used in training")
    ap.add_argument("--id-revision-rho", type=float, default=0.8)
    ap.add_argument("--max-cycles", type=float, default=None,
                    help="override daily cycle limit Cs (match training)")
    ap.add_argument("--decision-interval", type=int, default=8,
                    help="re-plan targets every K steps (match training)")
    ap.add_argument("--reveal-window", type=int, default=24)
    ap.add_argument("--spike-prob", type=float, default=0.0, help="XBID RES up-spike prob/product")
    ap.add_argument("--neg-prob", type=float, default=0.0, help="XBID negative-price prob/product")
    ap.add_argument("--capacity-floor", type=float, default=None, help="Capacity-Market SoC floor MWh")
    ap.add_argument("--wear-eur-mwh", type=float, default=2.0, help="wear cost €/MWh for the MPC objective")
    args = ap.parse_args()

    months = None if args.month == "all" else [args.month]
    prices_by_day, provider = load_dam_dataset(args.xlsx, months=months)
    days = provider.days
    if args.day:
        days = [d for d in days if d == args.day]
        if not days:
            raise SystemExit(f"day {args.day} not found in month {args.month}")

    cfg = BatteryConfig()   # real unit defaults (50 MW / 100 MWh / 1.5 cyc)
    if args.max_cycles is not None:
        cfg.max_cycles_per_day = args.max_cycles
    cfg.degradation_eur_per_mwh = args.wear_eur_mwh

    model = vecnorm = None
    if args.policy == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        model = PPO.load(args.model, device="cpu")
        if args.vecnorm:
            dummy = DummyVecEnv([lambda: XBIDEnv(
                dam_prices_by_day=prices_by_day, scenario_provider=provider,
                battery_config=cfg,
                id_revision_sigma_frac=args.id_revision_sigma,
                id_revision_rho=args.id_revision_rho,
                decision_interval=args.decision_interval,
                reveal_window=args.reveal_window,
                spike_prob=args.spike_prob, neg_prob=args.neg_prob)])
            vecnorm = VecNormalize.load(args.vecnorm, dummy)
            vecnorm.training = False
            vecnorm.norm_reward = False

    # One env restricted to the requested day set, iterated day by day.
    env = XBIDEnv(
        dam_prices_by_day={d: prices_by_day[d] for d in days},
        scenario_provider=provider,
        battery_config=cfg,
        forecast_seed=args.seed,
        id_revision_sigma_frac=args.id_revision_sigma,
        id_revision_rho=args.id_revision_rho,
        decision_interval=args.decision_interval,
        reveal_window=args.reveal_window,
        spike_prob=args.spike_prob, neg_prob=args.neg_prob,
    )
    predict = make_policy(args.policy, env, model, vecnorm, args.theta, args.delta,
                          c_wear=args.wear_eur_mwh, floor=args.capacity_floor)

    print("═" * 78)
    print(f"  Battery intraday evaluation — policy={args.policy}  month={args.month}  "
          f"days={len(days)}")
    print("═" * 78)

    rows = []
    for _ in range(len(days)):
        day = env.dam_days[env._day_index % len(env.dam_days)]
        m = rollout_day(env, predict, day, print_orders=args.print_orders)
        m["day"] = day
        rows.append(m)

    # ── Per-day table ─────────────────────────────────────────────────────
    print(f"\n  {'Day':<12}{'Econ €':>10}{'Cash €':>10}{'Thru MWh':>10}"
          f"{'Imb MWh':>9}{'Imb €':>9}{'SOEend':>8}{'cyc c/d':>10}")
    print("  " + "─" * 76)
    for r in rows:
        print(f"  {r['day']:<12}{r['ep_economic_pnl']:>10.0f}{r['ep_realized_pnl']:>10.0f}"
              f"{r['ep_throughput_mwh']:>10.1f}{r['ep_imbalance_mwh']:>9.2f}"
              f"{r['ep_imbalance_cost']:>9.0f}{r['ep_soc_final']:>8.1f}"
              f"{r['ep_charge_cycles']:>5.2f}/{r['ep_discharge_cycles']:<4.2f}")

    econ = np.array([r["ep_economic_pnl"] for r in rows])
    cash = np.array([r["ep_realized_pnl"] for r in rows])
    imb  = np.array([r["ep_imbalance_cost"] for r in rows])
    print("  " + "─" * 76)
    print(f"  {'MEAN':<12}{econ.mean():>10.0f}{cash.mean():>10.0f}"
          f"{'':>10}{'':>9}{imb.mean():>9.0f}")
    print(f"  {'TOTAL':<12}{econ.sum():>10.0f}{cash.sum():>10.0f}")
    if len(econ) > 1:
        sharpe = econ.mean() / (econ.std() + 1e-9)
        worst = np.quantile(econ, 0.1)
        print(f"\n  daily economic PnL:  mean={econ.mean():.0f} €   std={econ.std():.0f} €   "
              f"Sharpe={sharpe:.2f}   CVaR10%={econ[econ<=worst].mean():.0f} €")
    print("═" * 78)


if __name__ == "__main__":
    main()
