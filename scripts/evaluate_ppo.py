"""Evaluation script for the trained XBID PPO agent — INSTRUMENTED.

.. deprecated::
    This is the original *supplier* evaluator (hedge ratio, residual coverage,
    IDA venue benchmark). It is **not applicable to the battery** and references
    removed observation attributes (``_rt``, ``_xbid_fills``, ...). For the
    battery use ``scripts/evaluate_battery.py`` instead, which reports the
    correct KPIs (economic PnL, cycles, imbalance, SoE) and prints the executed
    orders per 15-minute delivery quarter. This file is kept only for reference.

Reports:
  • Trained PPO agent performance on real holdout days.
  • Two control baselines (always-BUY δ=1.5, random θ/δ).
  • Per-day BUY/SELL volume, net XBID position and residual Rt.
  • Hedge ratio (|Rt| reduction).
  • Decision quality: 3×3 confusion matrix classifying every slot as
    BUY / SELL / HOLD (ideal vs actual), plus per-trade hit rates vs
    the imbalance settlement price.
  • IDA benchmark: per-trade comparison vs IDA1/2/3 auction prices.

Run from xbid_hybrid_trader/:
    python scripts/evaluate_ppo.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import data_paths as _dp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from dam_data_manager import DAMDataManager
from historical_data_loader import HistoricalDataLoader
from xbid_trader.market.xbid_env import XBIDEnv, MAX_SLOTS
from xbid_trader.market.rl_observation import N_FEATURES

try:
    from bm_data_loader import BMDataLoader
except ImportError:
    BMDataLoader = None

try:
    from imbalance_price_loader import ImbalancePriceLoader
except ImportError:
    ImbalancePriceLoader = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_ppo")

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH    = Path(os.environ.get(
    "XBID_MODEL_PATH",
    "scripts/training_output_v3/best_model/best_model.zip",
))
VECNORM_PATH  = Path(os.environ.get(
    "XBID_VECNORM_PATH",
    "scripts/training_output_v3/xbid_vecnormalize.pkl",
))
EXCEL_PATH     = _dp.excel_path(required=True)
BM_DATA_PATH   = _dp.bm_path(required=True)
IMBALANCE_PATH = _dp.imbalance_path(required=True)

# Reward weights theta - loaded from the meta-layer winner (best_theta.json).
# Falls back to neutral defaults if absent. theta only affects the env reward
# signal; reported metrics use realized hedging PnL, so loading theta* just
# keeps the eval env consistent with the trained policy.
LAMBDA_POS = 0.5
LAMBDA_SHAPING, LAMBDA_RESIDUAL, LAMBDA_CVAR = 1.0, 0.01, 0.0
_THETA_PATH = Path(os.environ.get(
    "XBID_THETA_PATH", str(MODEL_PATH.parent.parent / "best_theta.json")))
if _THETA_PATH.exists():
    try:
        with open(_THETA_PATH, encoding="utf-8") as _f:
            _theta = json.load(_f).get("theta", {})
        LAMBDA_SHAPING  = float(_theta.get("lambda_shaping",  LAMBDA_SHAPING))
        LAMBDA_RESIDUAL = float(_theta.get("lambda_residual", LAMBDA_RESIDUAL))
        LAMBDA_CVAR     = float(_theta.get("lambda_cvar",     LAMBDA_CVAR))
        logger.info("Loaded theta* from %s: shp=%.3f res=%.4f cvar=%.3f",
                    _THETA_PATH, LAMBDA_SHAPING, LAMBDA_RESIDUAL, LAMBDA_CVAR)
    except Exception as _e:
        logger.warning("Could not parse %s (%s); using default weights.", _THETA_PATH, _e)
else:
    logger.warning("best_theta.json not found at %s; using default reward "
                   "weights (does not affect PnL-based metrics).", _THETA_PATH)

# ── Output directory for structured results ───────────────────────────
RESULTS_DIR = Path(os.environ.get("XBID_RESULTS_DIR", "scripts/evaluation_results"))
SLIPPAGE_COSTS = [0.0, 0.25, 0.50, 0.75, 1.00]


def _save_json(data: dict, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Saved %s", path)
    return path


def _save_csv(rows: List[dict], filename: str, fieldnames: List[str] = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    if not rows:
        return path
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────────
# Instrumented env — logs all agent trades for IDA benchmarking
# ──────────────────────────────────────────────────────────────────────────

class InstrumentedXBIDEnv(XBIDEnv):
    """Tracks BUY/SELL volume, residual Rt and a per-day trade log:
    list of (slot, signed_qty, price)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ep_buy_mwh:  float = 0.0
        self._ep_sell_mwh: float = 0.0
        self._ep_trades:   List[Tuple[int, float, float]] = []

    def reset(self, *, seed=None, options=None):
        self._ep_buy_mwh  = 0.0
        self._ep_sell_mwh = 0.0
        self._ep_trades   = []
        return super().reset(seed=seed, options=options)

    def step(self, action):
        prev_fills = self._obs_builder._xbid_fills.copy()
        obs, reward, terminated, truncated, info = super().step(action)

        delta = self._obs_builder._xbid_fills - prev_fills
        self._ep_buy_mwh  += float(np.sum(np.maximum( delta, 0.0)))
        self._ep_sell_mwh += float(np.sum(np.maximum(-delta, 0.0)))

        # Record this step's trades for IDA benchmark
        for trade, sign in zip(info.get("agent_trades", []),
                               info.get("agent_signs", [])):
            self._ep_trades.append((
                int(trade.product_id),
                float(sign),               # +qty for BUY, -qty for SELL
                float(trade.price),
            ))

        if terminated:
            n = self._n_slots
            ob = self._obs_builder
            info["ep_buy_mwh"]          = self._ep_buy_mwh
            info["ep_sell_mwh"]         = self._ep_sell_mwh
            info["ep_xbid_position"]    = float(np.sum(ob._xbid_fills[:n]))
            info["ep_realized_pnl"]     = float(np.sum(ob._realized_pnl[:n]))
            info["ep_dam_position"]     = float(np.sum(ob._dam_position[:n]))
            info["ep_dam_position_abs"] = float(np.sum(np.abs(ob._dam_position[:n])))
            info["ep_rt_initial_abs"]   = float(np.sum(np.abs(ob._rt_initial[:n])))
            info["ep_rt_remaining_abs"] = float(np.sum(np.abs(ob._rt[:n])))
            info["ep_residual_cost"]    = float(np.sum(
                np.abs(ob._rt[:n]) * np.abs(ob._real_imbalance_prices[:n])
            ))
            info["ep_n_slots"]          = int(n)
            info["ep_trades"]           = list(self._ep_trades)

        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────
# Predict functions for baselines
# ──────────────────────────────────────────────────────────────────────────

def make_constant_action_predict(theta: float, delta: float):
    flat = np.empty(MAX_SLOTS * 2, dtype=np.float32)
    flat[0::2] = theta
    flat[1::2] = delta
    flat = flat.reshape(1, -1)

    def predict(obs, deterministic=True, state=None, episode_start=None):
        return flat, None

    return predict


def make_random_predict(seed: int = 0):
    rng = np.random.default_rng(seed)

    def predict(obs, deterministic=True, state=None, episode_start=None):
        a = np.empty(MAX_SLOTS * 2, dtype=np.float32)
        a[0::2] = rng.uniform(0.0, 2.0, size=MAX_SLOTS)
        a[1::2] = rng.uniform(0.0, 2.0, size=MAX_SLOTS)
        return a.reshape(1, -1), None

    return predict


def make_no_hedge_predict():
    """Never trade — leaves everything to BM settlement."""
    # θ = 999 ensures |γ| < θ always → HOLD on every slot
    flat = np.empty(MAX_SLOTS * 2, dtype=np.float32)
    flat[0::2] = 999.0  # impossibly high threshold
    flat[1::2] = 0.0
    flat = flat.reshape(1, -1)

    def predict(obs, deterministic=True, state=None, episode_start=None):
        return flat, None

    return predict


def make_threshold_predict(threshold: float, delta: float = 1.0):
    """Trade when |γ| > threshold, at fixed aggressiveness δ.

    This is a rule-based baseline: 'if the expected BM settlement
    price is far enough from the XBID mid, trade.'
    """
    flat = np.empty(MAX_SLOTS * 2, dtype=np.float32)
    flat[0::2] = threshold
    flat[1::2] = delta
    flat = flat.reshape(1, -1)

    def predict(obs, deterministic=True, state=None, episode_start=None):
        return flat, None

    return predict


# ──────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────

def _new_day_stats() -> dict:
    return {
        "day":              "",
        "rewards":          [],
        "n_trades":         0,
        "buy_mwh":          0.0,
        "sell_mwh":         0.0,
        "xbid_position":    0.0,
        "dam_position":     0.0,
        "dam_position_abs": 0.0,
        "realized_pnl":     0.0,
        "rt_initial_abs":   0.0,
        "rt_remaining_abs": 0.0,
        "episode_reward":   0.0,
        "n_slots":          0,
        "trades":           [],
    }


def run_evaluation(
    eval_env,
    predict_fn: Callable,
    n_days: int,
    label: str,
    eval_days: list,
) -> List[dict]:
    inner_env = eval_env.venv.envs[0].env
    inner_env._day_index = 0
    obs = eval_env.reset()

    daily_stats: List[dict] = []
    current = _new_day_stats()

    logger.info("[%s] running %d episodes…", label, n_days)

    while len(daily_stats) < n_days:
        action, _ = predict_fn(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

        current["rewards"].append(float(reward[0]))
        current["n_trades"] += info[0].get("n_agent_trades", 0)

        if done[0]:
            i = len(daily_stats)
            current["day"]              = str(eval_days[i % len(eval_days)])
            current["realized_pnl"]     = info[0].get("ep_realized_pnl", 0.0)
            current["xbid_position"]    = info[0].get("ep_xbid_position", 0.0)
            current["dam_position"]     = info[0].get("ep_dam_position", 0.0)
            current["dam_position_abs"] = info[0].get("ep_dam_position_abs", 0.0)
            current["buy_mwh"]          = info[0].get("ep_buy_mwh", 0.0)
            current["sell_mwh"]         = info[0].get("ep_sell_mwh", 0.0)
            current["rt_initial_abs"]   = info[0].get("ep_rt_initial_abs", 0.0)
            current["rt_remaining_abs"] = info[0].get("ep_rt_remaining_abs", 0.0)
            current["n_slots"]          = info[0].get("ep_n_slots", 0)
            current["trades"]           = info[0].get("ep_trades", [])
            current["episode_reward"]   = float(np.sum(current["rewards"]))
            daily_stats.append(current)
            current = _new_day_stats()

    return daily_stats


# ──────────────────────────────────────────────────────────────────────────
# Reporting — agent vs baselines
# ──────────────────────────────────────────────────────────────────────────

def _print_run(label: str, stats: List[dict]) -> None:
    print(f"\n  ── {label} " + "─" * max(0, 78 - len(label) - 5))
    print(f"  {'Date':<12} {'PnL':>11} {'XBID':>8} {'Buy':>7} {'Sell':>7} "
          f"{'Trades':>7} {'|Rt0|':>8} {'|RtR|':>8}")
    print(f"  {'─'*12} {'─'*11} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")
    for d in stats:
        sign = "+" if d["realized_pnl"] >= 0 else ""
        print(
            f"  {d['day']:<12} "
            f"{sign}{d['realized_pnl']:>9.0f}€ "
            f"{d['xbid_position']:>+7.1f} "
            f"{d['buy_mwh']:>6.1f} "
            f"{d['sell_mwh']:>6.1f} "
            f"{d['n_trades']:>6} "
            f"{d['rt_initial_abs']:>7.1f} "
            f"{d['rt_remaining_abs']:>7.1f}"
        )


def _agg(stats: List[dict], key: str) -> float:
    return float(np.mean([d[key] for d in stats])) if stats else 0.0


def _print_comparison(runs: dict) -> None:
    print("\n" + "═" * 90)
    print("  XBID Evaluation — daily breakdown")
    print("═" * 90)
    print("  PnL=realized €, XBID=net agent position MWh, Buy/Sell=fill MWh,")
    print("  |Rt0|=initial hedging need, |RtR|=residual after agent trades.\n")

    for label, stats in runs.items():
        _print_run(label, stats)

    print("\n" + "  " + "═" * 88)
    print("  AGGREGATE COMPARISON  (mean across days)")
    print("  " + "─" * 88)

    header = f"  {'Metric':<32}" + "".join(f"{lbl:>17}" for lbl in runs)
    print(header)
    print("  " + "─" * 88)

    rows = [
        ("Realized PnL/day (€)",     "realized_pnl",     "{:>+15.0f}  "),
        ("Trades/day",               "n_trades",         "{:>17.0f}  "),
        ("Buy MWh/day",              "buy_mwh",          "{:>17.1f}  "),
        ("Sell MWh/day",             "sell_mwh",         "{:>17.1f}  "),
        ("Net XBID pos/day (MWh)",   "xbid_position",    "{:>+15.1f}  "),
        ("Initial |Rt|/day (MWh)",   "rt_initial_abs",   "{:>17.1f}  "),
        ("Residual |Rt|/day (MWh)",  "rt_remaining_abs", "{:>17.1f}  "),
    ]
    for name, key, fmt in rows:
        line = f"  {name:<32}"
        for stats in runs.values():
            line += fmt.format(_agg(stats, key))
        print(line)

    def per_mwh(stats):
        tot_pnl = sum(d["realized_pnl"] for d in stats)
        tot_vol = sum(d["buy_mwh"] + d["sell_mwh"] for d in stats)
        return tot_pnl / tot_vol if tot_vol > 0 else 0.0

    line = f"  {'PnL per traded MWh (€)':<32}"
    for stats in runs.values():
        line += "{:>+15.3f}  ".format(per_mwh(stats))
    print(line)

    def hedge_ratio(stats):
        dam = sum(d["dam_position_abs"] for d in stats)
        rt0 = sum(d["rt_initial_abs"] for d in stats)
        rtr = sum(d["rt_remaining_abs"] for d in stats)
        hedged = rt0 - rtr
        return (dam + hedged) / (dam + rt0) if (dam + rt0) > 0 else 0.0

    line = f"  {'Hedge ratio (1=perfect)':<32}"
    for stats in runs.values():
        line += "{:>17.3f}  ".format(hedge_ratio(stats))
    print(line)

    def residual_hedge(stats):
        rt0 = sum(d["rt_initial_abs"] for d in stats)
        rtr = sum(d["rt_remaining_abs"] for d in stats)
        return (rt0 - rtr) / rt0 if rt0 > 1e-9 else 0.0

    line = f"  {'Residual coverage (|Rt| hedged)':<32}"
    for stats in runs.values():
        line += "{:>17.3f}  ".format(residual_hedge(stats))
    print(line)

    # Diagnosis
    print("\n  " + "═" * 88)
    print("  DIAGNOSIS")
    print("  " + "─" * 88)
    if "Trained PPO" in runs:
        ppo  = runs["Trained PPO"]
        buy  = _agg(ppo, "buy_mwh")
        sell = _agg(ppo, "sell_mwh")
        rt0  = _agg(ppo, "rt_initial_abs")
        rtR  = _agg(ppo, "rt_remaining_abs")
        ratio = sell / (buy + sell + 1e-9)
        print(f"  PPO sell/(buy+sell) volume ratio:  {ratio*100:5.1f}%")
        print(f"  PPO total volume / hedging need:   "
              f"{(buy + sell) / (rt0 + 1e-9):.2f}x")
        print(f"  PPO hedge ratio (|Rt| reduction):  "
              f"{(1 - rtR / (rt0 + 1e-9)) * 100:5.1f}%")
    print("═" * 90 + "\n")

    # ── Save to files ─────────────────────────────────────────────────
    csv_rows = []
    for label, stats in runs.items():
        for d in stats:
            csv_rows.append({
                "strategy": label, "date": d["day"],
                "realized_pnl": round(d["realized_pnl"], 2),
                "buy_mwh": round(d["buy_mwh"], 2),
                "sell_mwh": round(d["sell_mwh"], 2),
                "n_trades": d["n_trades"],
                "rt_initial_abs": round(d["rt_initial_abs"], 2),
                "rt_remaining_abs": round(d["rt_remaining_abs"], 2),
                "dam_position_abs": round(d.get("dam_position_abs", 0), 2),
            })
    _save_csv(csv_rows, "daily_stats.csv")

    agg_data = {}
    for label, stats in runs.items():
        agg_data[label] = {
            "pnl_per_day": round(_agg(stats, "realized_pnl"), 2),
            "trades_per_day": round(_agg(stats, "n_trades"), 1),
            "pnl_per_mwh": round(per_mwh(stats), 3),
            "hedge_ratio": round(hedge_ratio(stats), 4),
            "residual_hedge": round(residual_hedge(stats), 4),
        }
    _save_json(agg_data, "aggregate_comparison.json")


# ──────────────────────────────────────────────────────────────────────────
# IDA benchmark
# ──────────────────────────────────────────────────────────────────────────

def compute_ida_benchmark(
    daily_stats: List[dict],
    ida_prices_by_day: Dict[date_type, np.ndarray],
    eval_days: List[date_type],
) -> dict:
    """For each agent trade, compute the counter-factual savings had it
    been executed at IDA1/2/3 auction prices instead of XBID.

    Per-trade savings (XBID vs IDA_X):
        savings = q_signed × (p_ida_X − p_xbid)

    Positive  → XBID was a better venue than IDA_X for this fill
    Negative  → IDA_X would have been cheaper for this fill

    Returns
    -------
    dict with:
        per_day  : list with vs_ida1/2/3 totals per day
        totals   : aggregate savings (€) per IDA
        coverage : fraction of agent volume that had a finite IDA price
    """
    per_day = []
    totals  = {"ida1": 0.0, "ida2": 0.0, "ida3": 0.0}
    coverage = {"ida1": [0.0, 0.0], "ida2": [0.0, 0.0], "ida3": [0.0, 0.0]}

    for i, d in enumerate(daily_stats):
        day = eval_days[i % len(eval_days)]
        ida_arr = ida_prices_by_day.get(day)
        if ida_arr is None:
            per_day.append({
                "day": d["day"], "vs_ida1": np.nan,
                "vs_ida2": np.nan, "vs_ida3": np.nan,
                "n_matched_1": 0, "n_matched_2": 0, "n_matched_3": 0,
            })
            continue

        n_slots = ida_arr.shape[0]
        savings = {"ida1": 0.0, "ida2": 0.0, "ida3": 0.0}
        matched = {"ida1": 0,   "ida2": 0,   "ida3": 0}

        for slot, q_signed, p_xbid in d["trades"]:
            if not 0 <= slot < n_slots:
                continue
            qty = abs(q_signed)
            for col, key in enumerate(("ida1", "ida2", "ida3")):
                p_ida = ida_arr[slot, col]
                coverage[key][1] += qty
                if not np.isfinite(p_ida) or p_ida <= 0:
                    continue
                savings[key] += q_signed * (p_ida - p_xbid)
                matched[key] += 1
                coverage[key][0] += qty

        for k in totals:
            totals[k] += savings[k]
        per_day.append({
            "day": d["day"],
            "vs_ida1": savings["ida1"], "n_matched_1": matched["ida1"],
            "vs_ida2": savings["ida2"], "n_matched_2": matched["ida2"],
            "vs_ida3": savings["ida3"], "n_matched_3": matched["ida3"],
        })

    cov_ratio = {
        k: (v[0] / v[1] if v[1] > 0 else 0.0)
        for k, v in coverage.items()
    }
    return {"per_day": per_day, "totals": totals, "coverage": cov_ratio}


def _print_ida_benchmark(daily_stats: List[dict], bench: dict) -> None:
    print("\n" + "═" * 90)
    print("  IDA BENCHMARK — XBID execution vs IDA1/IDA2/IDA3 auction prices")
    print("═" * 90)
    print("  Per-trade savings = q_signed × (p_IDA − p_XBID)")
    print("  Positive  → XBID was a better venue than the IDA auction")
    print("  Negative  → the IDA auction would have given a better price\n")

    print(f"  {'Date':<12} {'PPO €':>10} {'vs IDA1':>11} {'vs IDA2':>11} "
          f"{'vs IDA3':>11} {'#1':>5} {'#2':>5} {'#3':>5}")
    print(f"  {'─'*12} {'─'*10} {'─'*11} {'─'*11} {'─'*11} {'─'*5} {'─'*5} {'─'*5}")

    for stats, row in zip(daily_stats, bench["per_day"]):
        def fmt(x):
            if not np.isfinite(x):
                return f"{'N/A':>11}"
            sign = "+" if x >= 0 else ""
            return f" {sign}{x:>8.0f}€ "
        ppo_pnl_str = (
            f" {'+' if stats['realized_pnl'] >= 0 else ''}"
            f"{stats['realized_pnl']:>7.0f}€"
        )
        print(
            f"  {row['day']:<12} {ppo_pnl_str} "
            f"{fmt(row['vs_ida1'])}"
            f"{fmt(row['vs_ida2'])}"
            f"{fmt(row['vs_ida3'])}"
            f"{row['n_matched_1']:>5} "
            f"{row['n_matched_2']:>5} "
            f"{row['n_matched_3']:>5}"
        )

    print("\n  " + "─" * 88)
    print("  AGGREGATE")
    print("  " + "─" * 88)
    tot_ppo = sum(d["realized_pnl"] for d in daily_stats)
    tot_vol = sum(d["buy_mwh"] + d["sell_mwh"] for d in daily_stats)
    print(f"  Total realized PnL (XBID)            : {tot_ppo:>+12.2f} €")
    print(f"  Total agent traded volume            : {tot_vol:>12.1f} MWh")

    for label, key in (("IDA1", "ida1"), ("IDA2", "ida2"), ("IDA3", "ida3")):
        sav = bench["totals"][key]
        cov = bench["coverage"][key]
        per_mwh = (sav / (tot_vol * cov)) if tot_vol > 0 and cov > 0 else 0.0
        print(
            f"  Total savings vs {label} (XBID − {label}): "
            f"{sav:>+12.2f} €   "
            f"coverage={cov*100:5.1f}%   "
            f"avg={per_mwh:+6.3f} €/MWh"
        )

    print("\n  Interpretation: positive total = the XBID strategy executed at")
    print("  more favourable prices than the corresponding IDA auction would")
    print("  have provided for the same trade direction and slot.")
    print("═" * 90 + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Decision quality analysis
#
# For every (day × slot) the agent has three possible decisions:
#   BUY   — hedge a deficit through XBID
#   SELL  — hedge a surplus through XBID
#   HOLD  — leave the residual for balancing / imbalance settlement
#
# The Greek balancing market uses a single imbalance price per ISP,
# so for each slot the ex-post optimal decision is fully determined by
# the sign of Rt and whether the XBID execution price beat the imbalance
# price.  Per slot:
#   if Rt_i >  τ  and  p_xbid < p_imb_i   →  BUY is optimal
#   if Rt_i < -τ  and  p_xbid > p_imb_i   →  SELL is optimal
#   otherwise                              →  HOLD is optimal
#
# We report:
#   1. A 3×3 confusion matrix (ideal direction based on sign(Rt_0) vs
#      actual net agent direction on that slot).
#   2. Hit rates per side: fraction of BUY (resp. SELL) trades with
#      positive savings vs p_imb_i.
#   3. Average savings per trade (€/MWh), analogous to the Implementation
#      Shortfall metric in RL execution literature.
#   4. Counts of missed hedges and wrong-direction trades.
#
# The p_imb_i values come from scenario.imbalance_prices (= XBID VWAP for
# that slot — the best available ex-post proxy for the balancing price
# given the current dataset).
# ──────────────────────────────────────────────────────────────────────────

def analyze_decision_quality(
    daily_stats: List[dict],
    scenario_provider,
    eval_days: list,
    rt_threshold: float = 0.5,
) -> dict:
    """Classify every slot decision as BUY/SELL/HOLD and compute per-trade
    savings versus the slot's imbalance-settlement price.

    ``rt_threshold`` is the absolute |Rt| below which a slot needs no
    hedging action (matches the direction-mask used in the env).
    """
    labels = ["BUY", "SELL", "HOLD"]
    conf = np.zeros((3, 3), dtype=int)   # rows = ideal, cols = actual

    all_buy_savings:  List[float] = []
    all_sell_savings: List[float] = []

    # Covered volume by direction (for hedging effectiveness)
    total_deficit_need   = 0.0
    total_surplus_need   = 0.0
    covered_deficit_mwh  = 0.0
    covered_surplus_mwh  = 0.0

    for i, d in enumerate(daily_stats):
        day = eval_days[i % len(eval_days)]
        try:
            scenario = scenario_provider.get_scenario(day)
        except Exception:
            continue

        rt0    = np.asarray(scenario.rt, dtype=float)
        p_imb  = np.asarray(scenario.imbalance_prices, dtype=float)
        n_slots = len(rt0)

        # Aggregate net signed volume per slot
        slot_net      = np.zeros(n_slots)
        slot_buy_qty  = np.zeros(n_slots)
        slot_sell_qty = np.zeros(n_slots)

        for slot, signed_qty, p_xbid in d["trades"]:
            if not 0 <= slot < n_slots:
                continue
            slot_net[slot] += signed_qty
            qty = abs(signed_qty)
            p_i = p_imb[slot]

            if signed_qty > 0:
                slot_buy_qty[slot] += qty
                all_buy_savings.append(qty * (p_i - p_xbid))
            else:
                slot_sell_qty[slot] += qty
                all_sell_savings.append(qty * (p_xbid - p_i))

        # Hedging effectiveness — how much of the real need was covered
        # in the correct direction.
        deficit_mask = rt0 >  rt_threshold
        surplus_mask = rt0 < -rt_threshold
        total_deficit_need += float(np.sum(rt0[deficit_mask]))
        total_surplus_need += float(np.sum(-rt0[surplus_mask]))
        covered_deficit_mwh += float(np.sum(
            np.minimum(slot_buy_qty[deficit_mask], rt0[deficit_mask])
        ))
        covered_surplus_mwh += float(np.sum(
            np.minimum(slot_sell_qty[surplus_mask], -rt0[surplus_mask])
        ))

        # 3×3 confusion matrix
        for s in range(n_slots):
            if rt0[s] >  rt_threshold:
                ideal = 0
            elif rt0[s] < -rt_threshold:
                ideal = 1
            else:
                ideal = 2

            if slot_net[s] >  0.05:
                actual = 0
            elif slot_net[s] < -0.05:
                actual = 1
            else:
                actual = 2

            conf[ideal, actual] += 1

    def _stats(arr: List[float]) -> Tuple[int, float, float, float]:
        if not arr:
            return 0, 0.0, 0.0, 0.0
        a = np.asarray(arr)
        return (
            int(len(a)),
            float(np.sum(a > 0) / len(a)),   # hit rate
            float(np.mean(a)),                # mean
            float(np.median(a)),              # median
        )

    n_buy, hit_buy, mean_buy, med_buy  = _stats(all_buy_savings)
    n_sell, hit_sell, mean_sell, med_sell = _stats(all_sell_savings)

    return {
        "labels":         labels,
        "confusion":      conf,
        "n_buy":          n_buy,
        "n_sell":         n_sell,
        "hit_rate_buy":   hit_buy,
        "hit_rate_sell":  hit_sell,
        "mean_buy_sav":   mean_buy,
        "mean_sell_sav":  mean_sell,
        "median_buy_sav": med_buy,
        "median_sell_sav":med_sell,
        "deficit_need":   total_deficit_need,
        "surplus_need":   total_surplus_need,
        "covered_deficit":covered_deficit_mwh,
        "covered_surplus":covered_surplus_mwh,
    }


def _print_decision_quality(label: str, dq: dict) -> None:
    labels = dq["labels"]
    conf   = dq["confusion"]
    total  = int(conf.sum())
    diag   = int(np.trace(conf))

    print(f"\n  ── {label} " + "─" * max(0, 78 - len(label) - 5))
    print(f"  Confusion matrix  (rows = ideal based on Rt sign, "
          f"cols = actual agent action)")
    print(f"                  actual BUY    actual SELL   actual HOLD       total")
    row_totals = conf.sum(axis=1)
    for r, lbl in enumerate(labels):
        rt = int(row_totals[r])
        parts = []
        for c in range(3):
            pct = 100 * conf[r, c] / rt if rt > 0 else 0
            parts.append(f"{conf[r,c]:>5} ({pct:>4.0f}%)")
        print(f"  ideal {lbl:<4}   {parts[0]}  {parts[1]}  {parts[2]}      {rt:>5}")

    acc = 100 * diag / total if total > 0 else 0
    print(f"  " + "─" * 80)
    print(f"  Overall accuracy (diagonal):  {diag}/{total} = {acc:5.1f}%")

    # Directional sub-metrics
    ideal_buy_n   = int(conf[0].sum())
    hit_buy_slots = int(conf[0, 0])
    miss_buy      = int(conf[0, 2])
    flip_buy      = int(conf[0, 1])

    ideal_sell_n   = int(conf[1].sum())
    hit_sell_slots = int(conf[1, 1])
    miss_sell      = int(conf[1, 2])
    flip_sell      = int(conf[1, 0])

    print(f"\n  Deficit slots (ideal=BUY):   {ideal_buy_n:>5}  "
          f"→ hedged {hit_buy_slots} ({100*hit_buy_slots/ideal_buy_n if ideal_buy_n else 0:4.0f}%)   "
          f"held {miss_buy}   wrong-way {flip_buy}")
    print(f"  Surplus slots (ideal=SELL):  {ideal_sell_n:>5}  "
          f"→ hedged {hit_sell_slots} ({100*hit_sell_slots/ideal_sell_n if ideal_sell_n else 0:4.0f}%)   "
          f"held {miss_sell}   wrong-way {flip_sell}")
    print(f"  Balanced slots (ideal=HOLD): {int(conf[2].sum()):>5}  "
          f"→ held {int(conf[2, 2])}   "
          f"unneeded BUY {int(conf[2, 0])}   unneeded SELL {int(conf[2, 1])}")

    # Volume coverage
    d_need  = dq["deficit_need"]
    s_need  = dq["surplus_need"]
    d_cov   = dq["covered_deficit"]
    s_cov   = dq["covered_surplus"]
    d_pct   = 100 * d_cov / d_need if d_need > 0 else 0
    s_pct   = 100 * s_cov / s_need if s_need > 0 else 0
    print(f"\n  Volume coverage of real hedging need")
    print(f"    Deficit need: {d_need:7.1f} MWh   hedged {d_cov:7.1f} MWh  ({d_pct:5.1f}%)")
    print(f"    Surplus need: {s_need:7.1f} MWh   hedged {s_cov:7.1f} MWh  ({s_pct:5.1f}%)")

    # Per-trade hit rate vs imbalance price
    n_b = dq["n_buy"]
    n_s = dq["n_sell"]
    print(f"\n  Per-trade savings vs imbalance price")
    if n_b > 0:
        print(f"    BUYs :  {n_b:>5} trades   "
              f"hit rate {100*dq['hit_rate_buy']:5.1f}%   "
              f"mean savings {dq['mean_buy_sav']:+7.3f} €/MWh   "
              f"median {dq['median_buy_sav']:+7.3f}")
    if n_s > 0:
        print(f"    SELLs:  {n_s:>5} trades   "
              f"hit rate {100*dq['hit_rate_sell']:5.1f}%   "
              f"mean savings {dq['mean_sell_sav']:+7.3f} €/MWh   "
              f"median {dq['median_sell_sav']:+7.3f}")
    print("  (hit rate > 50% → agent timing beats random selection)")


def _print_decision_quality_all(runs_dq: dict) -> None:
    print("\n" + "═" * 90)
    print("  DECISION QUALITY — is the agent making correct BUY/SELL/HOLD choices?")
    print("═" * 90)
    print("  Every (day × slot) is classified by the sign of the initial Rt_i:")
    print("    Rt_i >  0.5 MWh  → ideal action is BUY  (hedge the deficit)")
    print("    Rt_i < -0.5 MWh  → ideal action is SELL (hedge the surplus)")
    print("    |Rt_i| ≤ 0.5 MWh → ideal action is HOLD (accept imbalance)")
    print("  Per-trade savings = q_signed × (p_imbalance − p_xbid) for BUYs")
    print("                    = |q|      × (p_xbid − p_imbalance) for SELLs")
    print("  Positive savings ⇒ the XBID fill beat the imbalance settlement.\n")

    for label, dq in runs_dq.items():
        _print_decision_quality(label, dq)
    print("═" * 90 + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════
# Advanced Financial Metrics & Slippage
# ══════════════════════════════════════════════════════════════════════════

def compute_financial_metrics(runs: dict) -> dict:
    """Sharpe, MaxDD, Win Rate, Profit Factor, Slippage break-even."""
    print("\n" + "═" * 90)
    print("  ADVANCED FINANCIAL METRICS")
    print("═" * 90)

    all_metrics = {}
    for label, stats in runs.items():
        pnl_arr = np.array([d["realized_pnl"] for d in stats])
        vol_arr = np.array([d["buy_mwh"] + d["sell_mwh"] for d in stats])
        n = len(pnl_arr)
        mean_pnl = pnl_arr.mean()
        std_pnl  = pnl_arr.std(ddof=1) if n > 1 else 1e-9
        sharpe   = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 1e-9 else 0.0
        cum_pnl = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_dd = float(drawdowns.min())
        dd_end   = int(np.argmin(drawdowns))
        dd_start = int(np.argmax(cum_pnl[:dd_end + 1])) if dd_end > 0 else 0
        win_rate = float(np.mean(pnl_arr > 0)) * 100
        loss_rate = float(np.mean(pnl_arr < 0)) * 100
        flat_rate = float(np.mean(pnl_arr == 0)) * 100
        gross_profit = float(pnl_arr[pnl_arr > 0].sum()) if (pnl_arr > 0).any() else 0.0
        gross_loss   = float(abs(pnl_arr[pnl_arr < 0].sum())) if (pnl_arr < 0).any() else 1e-9
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")
        annual_pnl = mean_pnl * 252
        calmar = annual_pnl / abs(max_dd) if abs(max_dd) > 1e-9 else float("inf")
        total_pnl = float(pnl_arr.sum())
        total_vol = float(vol_arr.sum())
        breakeven = total_pnl / total_vol if total_vol > 0 else 0.0

        print(f"\n  ── {label} " + "─" * max(0, 78 - len(label) - 5))
        print(f"    Sharpe Ratio (annualized):    {sharpe:>+8.2f}")
        print(f"    Maximum Drawdown:             {max_dd:>+8.0f} € (day {dd_start+1}→{dd_end+1})")
        print(f"    Calmar Ratio:                 {calmar:>8.2f}")
        print(f"    Win Rate:                     {win_rate:>7.1f}%  (loss {loss_rate:.1f}%, flat {flat_rate:.1f}%)")
        print(f"    Profit Factor:                {profit_factor:>8.2f}")
        print(f"    Mean daily PnL:               {mean_pnl:>+8.1f} €")
        print(f"    Std daily PnL:                {std_pnl:>8.1f} €")
        print(f"    Cumulative PnL:               {cum_pnl[-1]:>+8.0f} €  ({n} days)")

        print(f"\n    Slippage sensitivity:")
        print(f"    {'Cost €/MWh':>12}  {'Adj PnL/day':>12}  {'Adj Cum PnL':>12}  {'Still +?':>8}")
        print(f"    {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}")
        slippage_data = []
        for cost in SLIPPAGE_COSTS:
            adj_pnl = pnl_arr - cost * vol_arr
            adj_mean = float(adj_pnl.mean())
            adj_cum  = float(adj_pnl.sum())
            sign = "YES" if adj_mean > 0 else "NO"
            print(f"    {cost:>12.2f}  {adj_mean:>+12.1f}  {adj_cum:>+12.0f}  {sign:>8}")
            slippage_data.append({"cost_per_mwh": cost, "adj_pnl_per_day": round(adj_mean, 2),
                                  "adj_cum_pnl": round(adj_cum, 2), "still_positive": adj_mean > 0})
        print(f"    Break-even cost: {breakeven:>.2f} €/MWh")

        all_metrics[label] = {
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_eur": round(float(max_dd), 2),
            "max_dd_start_day": int(dd_start + 1), "max_dd_end_day": int(dd_end + 1),
            "calmar_ratio": round(float(calmar), 4) if not np.isinf(calmar) else None,
            "win_rate_pct": round(float(win_rate), 2),
            "loss_rate_pct": round(float(loss_rate), 2),
            "flat_rate_pct": round(float(flat_rate), 2),
            "profit_factor": round(float(profit_factor), 4) if not np.isinf(profit_factor) else None,
            "mean_daily_pnl": round(float(mean_pnl), 2),
            "std_daily_pnl": round(float(std_pnl), 2),
            "cumulative_pnl": round(float(cum_pnl[-1]), 2),
            "n_days": int(n),
            "breakeven_cost_per_mwh": round(float(breakeven), 4),
            "total_pnl": round(float(total_pnl), 2),
            "total_volume_mwh": round(float(total_vol), 2),
            "slippage_sensitivity": slippage_data,
        }

    print("\n" + "═" * 90 + "\n")
    _save_json(all_metrics, "financial_metrics.json")
    return all_metrics


# ══════════════════════════════════════════════════════════════════════════
# Walk-Forward / Rolling Window Testing
# ══════════════════════════════════════════════════════════════════════════

def run_walk_forward(model, all_prices, candidate_days, scenario_provider,
                     engine_config, vecnorm_path, window_size=14, step_size=14):
    print("\n" + "═" * 90)
    print("  WALK-FORWARD / ROLLING WINDOW TESTING")
    print("═" * 90)
    print(f"  Window size: {window_size} days, step: {step_size} days")
    print(f"  Total days available: {len(candidate_days)}\n")

    windows = []
    for start in range(0, len(candidate_days) - window_size + 1, step_size):
        windows.append(candidate_days[start:start + window_size])
    if not windows:
        print("  Not enough data for walk-forward testing.\n")
        return

    results = []
    print(f"  {'Window':>6}  {'Period':<27}  {'PnL/day':>9}  {'Trades':>7}  "
          f"{'Hedge%':>7}  {'PnL/MWh':>8}  {'Sharpe':>7}  {'MaxDD':>8}  {'Win%':>6}")
    print("  " + "─" * 100)

    for wi, w_days in enumerate(windows):
        w_prices = {d: all_prices[d] for d in w_days}

        def _make_wf_env(wp=w_prices):
            return Monitor(InstrumentedXBIDEnv(
                dam_prices_by_day=wp, engine_config=engine_config,
                scenario_provider=scenario_provider,
                lambda_shaping=LAMBDA_SHAPING, lambda_pos=LAMBDA_POS,
                lambda_residual=LAMBDA_RESIDUAL, scenario_seed=42))

        try:
            wf_env = VecNormalize.load(str(vecnorm_path), DummyVecEnv([_make_wf_env]))
            wf_env.training = False
            wf_env.norm_reward = False
            stats = run_evaluation(wf_env, model.predict, len(w_days), f"WF-{wi}", w_days)
        except Exception as exc:
            logger.warning("Walk-forward window %d failed: %s", wi, exc)
            continue

        pnl_arr = np.array([d["realized_pnl"] for d in stats])
        vol_arr = np.array([d["buy_mwh"] + d["sell_mwh"] for d in stats])
        rt0_sum = sum(d["rt_initial_abs"] for d in stats)
        rtr_sum = sum(d["rt_remaining_abs"] for d in stats)
        dam_sum = sum(d.get("dam_position_abs", 0) for d in stats)
        hedged  = rt0_sum - rtr_sum

        mean_pnl = float(pnl_arr.mean())
        std_pnl  = float(pnl_arr.std(ddof=1)) if len(pnl_arr) > 1 else 1e-9
        sharpe   = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 1e-9 else 0.0
        hedge    = (dam_sum + hedged) / (dam_sum + rt0_sum) if (dam_sum + rt0_sum) > 0 else 0.0
        trades   = float(np.mean([d["n_trades"] for d in stats]))
        tot_pnl  = float(pnl_arr.sum())
        tot_vol  = float(vol_arr.sum())
        pnl_mwh  = tot_pnl / tot_vol if tot_vol > 0 else 0.0
        cum = np.cumsum(pnl_arr)
        max_dd = float((cum - np.maximum.accumulate(cum)).min())
        win_rate = float(np.mean(pnl_arr > 0)) * 100

        period = f"{w_days[0]} → {w_days[-1]}"
        print(f"  {wi+1:>6}  {period:<27}  {mean_pnl:>+9.0f}  {trades:>7.0f}  "
              f"{hedge*100:>6.1f}%  {pnl_mwh:>+8.2f}  {sharpe:>+7.2f}  "
              f"{max_dd:>+8.0f}  {win_rate:>5.1f}%")
        results.append({"window": wi+1, "period": period, "mean_pnl": round(mean_pnl, 2),
                         "trades": round(trades, 1), "hedge": round(hedge, 4),
                         "pnl_mwh": round(pnl_mwh, 4), "sharpe": round(sharpe, 4),
                         "max_dd": round(max_dd, 2), "win_rate": round(win_rate, 2)})

    if not results:
        print("  No windows completed.\n")
        return

    pnls    = [r["mean_pnl"] for r in results]
    hedges  = [r["hedge"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    print("  " + "─" * 100)
    print(f"\n  Walk-Forward Summary ({len(results)} windows):")
    print(f"    PnL/day:      mean={np.mean(pnls):>+.1f} €, std={np.std(pnls):.1f}, "
          f"min={np.min(pnls):>+.0f}, max={np.max(pnls):>+.0f}")
    print(f"    Hedge ratio:  mean={np.mean(hedges):.3f}, std={np.std(hedges):.3f}")
    print(f"    Sharpe:       mean={np.mean(sharpes):>+.2f}, std={np.std(sharpes):.2f}")
    positive_windows = sum(1 for p in pnls if p > 0)
    print(f"    Profitable windows: {positive_windows}/{len(results)} "
          f"({positive_windows/len(results)*100:.0f}%)")
    print("\n" + "═" * 90 + "\n")

    _save_json({"windows": results, "summary": {
        "pnl_mean": round(float(np.mean(pnls)), 2),
        "pnl_std": round(float(np.std(pnls)), 2),
        "hedge_mean": round(float(np.mean(hedges)), 4),
        "profitable_windows": positive_windows,
        "profitable_pct": round(positive_windows / len(results) * 100, 1),
    }}, "walk_forward.json")
    _save_csv(results, "walk_forward.csv")


# ══════════════════════════════════════════════════════════════════════════
# main()
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("Loading DAM price data…")
    manager = DAMDataManager(cache_dir="scripts/dam_cache")

    hist_loader = None
    if EXCEL_PATH is not None and EXCEL_PATH.exists():
        hist_loader = HistoricalDataLoader(str(EXCEL_PATH))
        hist_loader.load()
        manager.load_from_excel(hist_loader, overwrite=False)

    manager.fetch_all()
    all_prices = manager.get_prices()
    if not all_prices:
        logger.error("No cached DAM data found.")
        return

    all_days = sorted(all_prices.keys())

    # Pre-fetch IDA prices and build the historical scenario provider
    # BEFORE choosing eval days, so we can filter to the days that
    # actually have real ISP / IDA data.
    ida_prices_by_day: Dict[date_type, np.ndarray] = {}
    if hist_loader is not None:
        try:
            ida_prices_by_day = hist_loader.get_ida_prices_by_day()
            logger.info("Loaded IDA prices for %d days", len(ida_prices_by_day))
        except Exception as exc:
            logger.warning("Could not load IDA prices: %s", exc)

    scenario_provider = None
    bm_loader = None
    if BMDataLoader is not None and BM_DATA_PATH is not None and BM_DATA_PATH.exists():
        try:
            bm_loader = BMDataLoader(BM_DATA_PATH)
            bm_loader.load()
            logger.info(
                "BMDataLoader loaded — %d BM days available",
                len(bm_loader.available_dates()),
            )
        except Exception as exc:
            logger.warning("Could not load BM data (%s)", exc)
            bm_loader = None

    # Imbalance settlement price loader
    imbalance_loader = None
    if ImbalancePriceLoader is not None and IMBALANCE_PATH is not None and IMBALANCE_PATH.exists():
        try:
            imbalance_loader = ImbalancePriceLoader(IMBALANCE_PATH)
            imbalance_loader.load()
            logger.info(
                "ImbalancePriceLoader loaded — %d days available",
                len(imbalance_loader),
            )
        except Exception as exc:
            logger.warning("Could not load imbalance prices (%s)", exc)
            imbalance_loader = None

    if hist_loader is not None:
        try:
            from xbid_trader.scenario.historical_scenario_provider import (
                HistoricalScenarioProvider,
            )
            scenario_provider = HistoricalScenarioProvider(
                hist_loader, bm_loader=bm_loader,
                imbalance_loader=imbalance_loader,
            )
            logger.info(
                "HistoricalScenarioProvider loaded — %d days available%s",
                len(scenario_provider),
                " (with BM data)" if bm_loader is not None else "",
            )
            calibrated = hist_loader.calibrate_engine_config()
            if calibrated:
                ENGINE_CONFIG["background_agent_configs"] = calibrated
                logger.info("Background agents calibrated from XBID data.")
        except Exception as exc:
            logger.warning("Could not load HistoricalScenarioProvider (%s)", exc)

    if scenario_provider is None:
        logger.warning("Eval will run on SYNTHETIC scenario data.")
        candidate_days = all_days
    else:
        # Restrict to days that exist in the Excel — otherwise the env
        # silently falls back to ScenarioGenerator and the IDA benchmark
        # has no data.
        hist_days_set = set(scenario_provider.available_dates())
        candidate_days = [d for d in all_days if d in hist_days_set]
        n_dropped = len(all_days) - len(candidate_days)
        if n_dropped:
            logger.info(
                "Filtered out %d HEnEx-only days without historical "
                "scenario data (%d → %d).",
                n_dropped, len(all_days), len(candidate_days),
            )

    if len(candidate_days) < N_EVAL_DAYS:
        logger.error(
            "Only %d real-data days available — need %d for evaluation.",
            len(candidate_days), N_EVAL_DAYS,
        )
        return

    eval_days   = candidate_days[-N_EVAL_DAYS:]
    eval_prices = {d: all_prices[d] for d in eval_days}

    logger.info(
        "Evaluating on %d real-data holdout days: %s → %s",
        len(eval_days), eval_days[0], eval_days[-1],
    )

    def make_eval_env():
        return Monitor(InstrumentedXBIDEnv(
            dam_prices_by_day=eval_prices,
            engine_config=ENGINE_CONFIG,
            scenario_provider=scenario_provider,
            lambda_shaping=LAMBDA_SHAPING,
            lambda_pos=LAMBDA_POS,
            lambda_residual=LAMBDA_RESIDUAL,
            scenario_seed=42,
        ))

    eval_env = VecNormalize.load(str(VECNORM_PATH), DummyVecEnv([make_eval_env]))
    eval_env.training    = False
    eval_env.norm_reward = False

    if not MODEL_PATH.exists():
        logger.error("Model not found at %s", MODEL_PATH)
        return
    logger.info("Loading model from %s", MODEL_PATH)
    model = PPO.load(str(MODEL_PATH), env=eval_env)

    runs = {}
    runs["Trained PPO"] = run_evaluation(
        eval_env, model.predict, len(eval_days), "Trained PPO", eval_days
    )
    runs["BUY δ=1.5"] = run_evaluation(
        eval_env,
        make_constant_action_predict(theta=0.0, delta=1.5),
        len(eval_days),
        "Always-BUY δ=1.5",
        eval_days,
    )
    runs["Random"] = run_evaluation(
        eval_env,
        make_random_predict(seed=123),
        len(eval_days),
        "Random θ/δ",
        eval_days,
    )
    runs["No-Hedging"] = run_evaluation(
        eval_env,
        make_no_hedge_predict(),
        len(eval_days),
        "No-Hedging (BM only)",
        eval_days,
    )
    runs["Threshold δ*=2.0"] = run_evaluation(
        eval_env,
        make_threshold_predict(threshold=2.0, delta=1.0),
        len(eval_days),
        "Threshold δ*=2.0",
        eval_days,
    )

    _print_comparison(runs)

    # ── Decision quality (BUY / SELL / HOLD correctness) ─────────────
    if scenario_provider is not None:
        runs_dq = {
            label: analyze_decision_quality(stats, scenario_provider, eval_days)
            for label, stats in runs.items()
        }
        _print_decision_quality_all(runs_dq)
    else:
        logger.info("Skipping decision quality — no scenario provider.")

    # ── IDA benchmark ─────────────────────────────────────────────────
    # IDA prices only exist for days loaded from the Excel.  Recent eval
    # days may have come from HEnEx (no IDA) → run a SEPARATE eval pass
    # on the last 14 days that DO have IDA prices.
    if not ida_prices_by_day:
        logger.warning("Skipping IDA benchmark — no IDA prices available.")
        return

    def _has_finite_ida(arr: np.ndarray) -> bool:
        return arr is not None and np.isfinite(arr).any() and (arr > 0).any()

    ida_eligible = sorted([
        d for d in all_days
        if d in ida_prices_by_day and _has_finite_ida(ida_prices_by_day[d])
    ])
    if not ida_eligible:
        logger.warning(
            "Skipping IDA benchmark — no overlap between DAM days and "
            "IDA-priced days."
        )
        return

    ida_eval_days = ida_eligible[-N_EVAL_DAYS:]
    logger.info(
        "Running IDA benchmark on %d days with IDA data: %s → %s",
        len(ida_eval_days), ida_eval_days[0], ida_eval_days[-1],
    )

    # Build a fresh eval env scoped to those days only
    ida_eval_prices = {d: all_prices[d] for d in ida_eval_days}

    def make_ida_eval_env():
        return Monitor(InstrumentedXBIDEnv(
            dam_prices_by_day=ida_eval_prices,
            engine_config=ENGINE_CONFIG,
            scenario_provider=scenario_provider,
            lambda_shaping=LAMBDA_SHAPING,
            lambda_pos=LAMBDA_POS,
            lambda_residual=LAMBDA_RESIDUAL,
            scenario_seed=42,
        ))

    ida_eval_env = VecNormalize.load(
        str(VECNORM_PATH), DummyVecEnv([make_ida_eval_env])
    )
    ida_eval_env.training    = False
    ida_eval_env.norm_reward = False

    ida_model = PPO.load(str(MODEL_PATH), env=ida_eval_env)
    ida_stats = run_evaluation(
        ida_eval_env, ida_model.predict, len(ida_eval_days),
        "PPO on IDA days", ida_eval_days,
    )

    bench = compute_ida_benchmark(ida_stats, ida_prices_by_day, ida_eval_days)
    _print_ida_benchmark(ida_stats, bench)

    # Decision quality on the IDA-days pass (more reliable metric —
    # real Rt and real imbalance prices)
    if scenario_provider is not None:
        dq_ida = analyze_decision_quality(
            ida_stats, scenario_provider, ida_eval_days
        )
        print("  IDA-window decision quality  (trained PPO only)")
        _print_decision_quality("Trained PPO (IDA days)", dq_ida)
        print("═" * 90 + "\n")

    # ── Advanced Financial Metrics ────────────────────────────────────
    compute_financial_metrics(runs)

    # ── Walk-Forward Testing ──────────────────────────────────────────
    run_walk_forward(
        model=model,
        all_prices=all_prices,
        candidate_days=candidate_days,
        scenario_provider=scenario_provider,
        engine_config=ENGINE_CONFIG,
        vecnorm_path=VECNORM_PATH,
        window_size=14,
        step_size=14,
    )

    # ── Final summary ─────────────────────────────────────────────────
    logger.info("All results saved to %s/", RESULTS_DIR)
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.rglob("*")):
            if f.is_file():
                logger.info("  %s  (%d KB)", f.name, f.stat().st_size // 1024)


if __name__ == "__main__":
    main()
