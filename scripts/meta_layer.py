"""Phase 3 — Meta-layer: Bayesian optimization of reward weights.

Architecture
------------
A shared base PPO is trained ONCE (BASE_STEPS, cached on disk).  Each
Optuna trial then CLONES that base and fine-tunes it under the trial's θ.

Outer loop (Optuna TPE + Hyperband pruner):
    For each trial, sample θ = (λ_shaping, λ_residual, λ_CVaR) and
    fine-tune the cloned base under those weights.  True multi-fidelity:
    fine-tuning is incremental across cumulative rungs (200K → 350K →
    500K steps); the SAME model is promoted between rungs and the
    Hyperband pruner stops unpromising trials early.

Inner loop (per trial):
    1. Build XBIDEnv + CVaRRewardWrapper with the sampled θ
    2. Fine-tune the cloned base PPO on the TRAIN set (130 days)
    3. Evaluate deterministically on the VAL set (29 days) after each rung
    4. Compute Score(θ) = mean(economic PnL) − w_imb × mean(imbalance cost)
                        − ρ × CVaR_0.1(PnL) − ρ_reg × ||θ||²
    5. Report the rung Score to Optuna (pruning between rungs)

The winning θ* is finally retrained FROM SCRATCH on train+val for
FINAL_STEPS.  Everything is seeded (XBID_SEED) for reproducibility.

Data split
----------
    173 usable days (Oct 2025 → Mar 2026)
    ├── Train:  130 days  (2025-10-01 → 2026-02-07)
    ├── Val:     29 days  (2026-02-08 → 2026-03-08)
    └── Test:    14 days  (2026-03-09 → 2026-03-22)

Output
------
    scripts/training_output_v3/
        optuna_study.db          — Optuna study (SQLite, resumable)
        best_theta.json          — θ* = optimal weights
        best_model/best_model.zip — final model (retrained on train+val)
        xbid_vecnormalize.pkl

Run from xbid_hybrid_trader/:
    python scripts/meta_layer.py

Evaluate the Phase 3 model:
    XBID_MODEL_PATH=scripts/training_output_v3/best_model/best_model.zip \\
    XBID_VECNORM_PATH=scripts/training_output_v3/xbid_vecnormalize.pkl \\
    python scripts/evaluate_ppo.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dam_data_manager import DAMDataManager
from historical_data_loader import HistoricalDataLoader
from xbid_trader.market.xbid_env import XBIDEnv
from xbid_trader.utils import set_seed

try:
    from xbid_trader.scenario.historical_scenario_provider import (
        HistoricalScenarioProvider,
    )
except ImportError:
    HistoricalScenarioProvider = None

try:
    from bm_data_loader import BMDataLoader
except ImportError:
    BMDataLoader = None

try:
    from imbalance_price_loader import ImbalancePriceLoader
except ImportError:
    ImbalancePriceLoader = None

from shared_policy import SharedEncoderPolicy
import data_paths as _dp
from cvar_wrapper import CVaRRewardWrapper

import optuna
from optuna.samplers import TPESampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meta_layer")


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

EXCEL_PATH     = _dp.excel_path(required=True)
BM_DATA_PATH   = _dp.bm_path(required=True)
IMBALANCE_PATH = _dp.imbalance_path(required=True)

OUTPUT_DIR = Path("scripts/training_output_v3")
STUDY_DB   = OUTPUT_DIR / "optuna_study.db"

# Data split sizes
N_TEST_DAYS = 14   # held out entirely from meta-layer
N_VAL_DAYS  = 29   # meta-layer evaluates here

# Optuna settings — report targets 100–125 trials (override via XBID_N_TRIALS)
N_TRIALS       = int(os.environ.get("XBID_N_TRIALS", "100"))
STUDY_NAME     = "xbid_meta_v3"
BASE_SEED      = int(os.environ.get("XBID_SEED", "42"))

# ── True multi-fidelity fine-tuning (report §5.2) ─────────────────────
# A single shared base PPO is trained once for BASE_STEPS and cached.
# Every trial CLONES that base and FINE-TUNES it under the trial's θ.
# Fine-tuning is incremental across CUMULATIVE_RUNGS (total fine-tune
# steps reached at each rung); the SAME model is promoted between rungs
# (successive halving), and Optuna's Hyperband pruner stops unpromising
# trials early.  This replaces the previous code that retrained from
# scratch at every rung and did no pruning.
BASE_STEPS       = int(os.environ.get("XBID_BASE_STEPS", "2000000"))
CUMULATIVE_RUNGS = [200_000, 350_000, 500_000]   # report: 200K–500K / trial
FINAL_STEPS      = 2_000_000   # winner retrained from scratch on train+val
FORCE_BASE       = os.environ.get("XBID_FORCE_BASE", "0") not in ("0", "", "false", "False")

# θ used to train the shared base model (neutral; the meta-layer searches
# around these).
BASE_THETA = {"lambda_shaping": 1.0, "lambda_residual": 0.01, "lambda_cvar": 0.0}

# Score function weights
W_IMB = 0.5        # weight on mean imbalance cost in Score
RHO_CVAR = 2.0     # weight on CVaR tail penalty
RHO_REG  = 0.1     # L2 regularization on θ

# CVaR wrapper settings (fixed across trials — only λ_CVaR varies)
CVAR_ALPHA   = 0.10
CVAR_WINDOW  = 200
CVAR_WARMUP  = 50

# PPO hyperparameters (fixed, same as Phase 2)
ENGINE_CONFIG = dict(
    event_granularity_seconds=60.0,
    gate_closure_offset_minutes=0.0,
    seed_books_on_init=True,
    initial_half_spread=0.5,
    initial_levels=1,
    initial_level_qty=10.0,
    background_agent_configs=None,
)

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=96 * 16,
    batch_size=96 * 4,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=0,
)


# ══════════════════════════════════════════════════════════════════════
# Data loading (done once, shared across all trials)
# ══════════════════════════════════════════════════════════════════════

def load_data() -> Tuple[
    Dict, Dict, Dict,           # train/val/test prices
    List, List, List,            # train/val/test day lists
    Optional[object],            # scenario_provider
    dict,                        # engine_config (with calibrated agents)
]:
    """Load all data sources and split into train/val/test."""
    logger.info("Loading data (one-time)…")

    manager = DAMDataManager(cache_dir="scripts/dam_cache")
    hist_loader = None
    if EXCEL_PATH is not None and EXCEL_PATH.exists():
        hist_loader = HistoricalDataLoader(str(EXCEL_PATH))
        hist_loader.load()
        manager.load_from_excel(hist_loader, overwrite=False)
    manager.fetch_all()
    all_prices = manager.get_prices()
    if not all_prices:
        raise RuntimeError("No DAM data available")
    all_days = sorted(all_prices.keys())

    # BM loader
    bm_loader = None
    if BMDataLoader is not None and BM_DATA_PATH is not None and BM_DATA_PATH.exists():
        try:
            bm_loader = BMDataLoader(BM_DATA_PATH)
            bm_loader.load()
            logger.info("BMDataLoader: %d days", len(bm_loader.available_dates()))
        except Exception as exc:
            logger.warning("Could not load BM data (%s)", exc)
            bm_loader = None

    # Scenario provider
    # Imbalance settlement price loader
    imbalance_loader = None
    if ImbalancePriceLoader is not None and IMBALANCE_PATH is not None and IMBALANCE_PATH.exists():
        try:
            imbalance_loader = ImbalancePriceLoader(IMBALANCE_PATH)
            imbalance_loader.load()
            logger.info("ImbalancePriceLoader: %d days", len(imbalance_loader))
        except Exception as exc:
            logger.warning("Could not load imbalance prices (%s)", exc)
            imbalance_loader = None

    engine_config = dict(ENGINE_CONFIG)
    scenario_provider = None
    if HistoricalScenarioProvider is not None and hist_loader is not None:
        try:
            scenario_provider = HistoricalScenarioProvider(
                hist_loader, bm_loader=bm_loader,
                imbalance_loader=imbalance_loader,
            )
            logger.info(
                "ScenarioProvider: %d days%s",
                len(scenario_provider),
                " (with BM)" if bm_loader else "",
            )
            calibrated = hist_loader.calibrate_engine_config()
            if calibrated:
                engine_config["background_agent_configs"] = calibrated
        except Exception as exc:
            logger.warning("Could not load ScenarioProvider (%s)", exc)

    # ── Loud summary of the data mode actually in effect ──────────────
    # A run that silently drops to the legacy reward (no BM dual pricing)
    # or proxy settlement prices will NOT match the report — make it
    # impossible to miss, and hard-fail under XBID_STRICT=1.
    degraded = []
    if scenario_provider is None:
        degraded.append("no HistoricalScenarioProvider → SYNTHETIC scenarios")
    if bm_loader is None:
        degraded.append("no BM data → reward uses legacy lambda_pos*|Rt| "
                        "(not BM dual pricing)")
    if imbalance_loader is None:
        degraded.append("no imbalance file → settlement uses XBID/DAM proxy "
                        "(not real ISP prices)")
    if degraded:
        _dp._banner(["DATA MODE IS DEGRADED — results will NOT match the report:"]
                    + [f"  - {d}" for d in degraded])
        if _dp.STRICT:
            raise RuntimeError("Degraded data mode under XBID_STRICT=1; "
                               "provide BM + imbalance + scenario data.")
    else:
        logger.info("Data mode OK: historical scenarios + BM dual pricing + "
                    "real imbalance settlement prices.")

    # Filter to intersection
    if scenario_provider is not None:
        sp_set = set(scenario_provider.available_dates())
        all_days = [d for d in all_days if d in sp_set]

    n_total = len(all_days)
    if n_total < N_TEST_DAYS + N_VAL_DAYS + 10:
        raise RuntimeError(
            f"Only {n_total} days available — need at least "
            f"{N_TEST_DAYS + N_VAL_DAYS + 10}"
        )

    # Split: [...train... | ...val... | ...test...]
    test_days  = all_days[-N_TEST_DAYS:]
    val_days   = all_days[-(N_TEST_DAYS + N_VAL_DAYS):-N_TEST_DAYS]
    train_days = all_days[:-(N_TEST_DAYS + N_VAL_DAYS)]

    logger.info(
        "Split: train=%d (%s→%s), val=%d (%s→%s), test=%d (%s→%s)",
        len(train_days), train_days[0], train_days[-1],
        len(val_days),   val_days[0],   val_days[-1],
        len(test_days),  test_days[0],  test_days[-1],
    )

    train_prices = {d: all_prices[d] for d in train_days}
    val_prices   = {d: all_prices[d] for d in val_days}
    test_prices  = {d: all_prices[d] for d in test_days}

    return (
        train_prices, val_prices, test_prices,
        train_days, val_days, test_days,
        scenario_provider,
        engine_config,
    )


# ══════════════════════════════════════════════════════════════════════
# Training + evaluation helpers
# ══════════════════════════════════════════════════════════════════════

def _make_train_env_fn(theta, prices, scenario_provider, engine_config, seed):
    def _fn():
        base = XBIDEnv(
            dam_prices_by_day=prices,
            engine_config=engine_config,
            scenario_provider=scenario_provider,
            lambda_shaping=theta["lambda_shaping"],
            lambda_pos=0.5,
            lambda_residual=theta["lambda_residual"],
            forecast_seed=seed,
        )
        return Monitor(CVaRRewardWrapper(
            base, alpha=CVAR_ALPHA, lambda_cvar=theta["lambda_cvar"],
            window=CVAR_WINDOW, warmup_episodes=CVAR_WARMUP,
        ))
    return _fn


def _make_eval_env_fn(theta, prices, scenario_provider, engine_config, seed=42):
    def _fn():
        base = XBIDEnv(
            dam_prices_by_day=prices,
            engine_config=engine_config,
            scenario_provider=scenario_provider,
            lambda_shaping=theta["lambda_shaping"],
            lambda_pos=0.5,
            lambda_residual=theta["lambda_residual"],
            scenario_seed=42,
            forecast_seed=seed,
        )
        return Monitor(base)        # NB: no CVaR wrapper during evaluation
    return _fn


def _evaluate_model(model, eval_env, eval_days) -> Dict[str, float]:
    """Deterministic rollout on eval_env; returns battery PnL/risk metrics."""
    eval_env.training    = False
    eval_env.norm_reward = False

    pnl_list, imb_list, thru_list = [], [], []
    obs = eval_env.reset()
    for _ in range(len(eval_days) * 100 + 100):   # up to 100 slots/day (DST)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done[0]:
            ep = info[0]
            # Economic € result of the day (cash flow − degradation −
            # imbalance − terminal-SoC), and its cost drivers.
            pnl_list.append(ep.get("ep_economic_pnl", ep.get("ep_realized_pnl", 0.0)))
            imb_list.append(ep.get("ep_imbalance_cost", 0.0))
            thru_list.append(ep.get("ep_throughput_mwh", 0.0))
            obs = eval_env.reset()
            if len(pnl_list) >= len(eval_days):
                break

    pnl_arr  = np.array(pnl_list)  if pnl_list  else np.array([0.0])
    imb_arr  = np.array(imb_list)  if imb_list  else np.array([0.0])
    thru_arr = np.array(thru_list) if thru_list else np.array([0.0])
    return {
        "mean_pnl":            float(pnl_arr.mean()),
        "mean_imbalance_cost": float(imb_arr.mean()),
        "mean_throughput":     float(thru_arr.mean()),
        "pnl_list":            pnl_arr.tolist(),
        "imbalance_list":      imb_arr.tolist(),
    }


def train_base_model(train_prices, scenario_provider, engine_config) -> Tuple[Path, Path]:
    """Train (or reuse) the shared base PPO model — report §5.2.

    Every Optuna trial fine-tunes a clone of THIS model, so it is trained
    exactly once and cached on disk.  Reproducible via BASE_SEED.
    """
    base_model   = OUTPUT_DIR / "base_model.zip"
    base_vecnorm = OUTPUT_DIR / "base_vecnorm.pkl"
    if base_model.exists() and base_vecnorm.exists() and not FORCE_BASE:
        logger.info("Reusing cached base model: %s", base_model)
        return base_model, base_vecnorm

    logger.info("Training shared base model: %dK steps (seed=%d)…",
                BASE_STEPS // 1000, BASE_SEED)
    train_env = VecNormalize(
        DummyVecEnv([_make_train_env_fn(
            BASE_THETA, train_prices, scenario_provider, engine_config,
            seed=BASE_SEED)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
    )
    model = PPO(policy=SharedEncoderPolicy, env=train_env,
                seed=BASE_SEED, **PPO_KWARGS)
    model.learn(total_timesteps=BASE_STEPS, progress_bar=False)
    model.save(str(base_model))
    train_env.save(str(base_vecnorm))
    del model, train_env
    gc.collect()
    logger.info("Base model saved → %s", base_model)
    return base_model, base_vecnorm


def finetune_trial(
    theta: Dict[str, float],
    trial,
    train_prices: Dict,
    eval_prices: Dict,
    eval_days: List,
    scenario_provider,
    engine_config: dict,
    base_model: Path,
    base_vecnorm: Path,
    seed: int,
) -> Dict[str, float]:
    """Multi-fidelity fine-tuning of the base model under θ (report §5.2).

    The base model is cloned and fine-tuned incrementally across
    CUMULATIVE_RUNGS.  After each rung the policy is evaluated on the
    validation set and the intermediate Score is reported to Optuna; the
    Hyperband pruner may stop unpromising trials early.  The SAME model is
    promoted between rungs (true successive-halving), not retrained.
    """
    # Train env continues from the base normalization statistics.
    train_env = VecNormalize.load(
        str(base_vecnorm),
        DummyVecEnv([_make_train_env_fn(
            theta, train_prices, scenario_provider, engine_config, seed=seed)]),
    )
    train_env.training = True
    train_env.norm_reward = True

    model = PPO.load(str(base_model), env=train_env, device="auto")
    model.set_random_seed(seed)

    eval_env = VecNormalize.load(
        str(base_vecnorm),
        DummyVecEnv([_make_eval_env_fn(
            theta, eval_prices, scenario_provider, engine_config, seed=seed)]),
    )

    results: Dict[str, float] = {}
    prev_steps = 0
    for rung_idx, cum_steps in enumerate(CUMULATIVE_RUNGS):
        delta = cum_steps - prev_steps
        prev_steps = cum_steps
        model.learn(total_timesteps=delta, progress_bar=False,
                    reset_num_timesteps=(rung_idx == 0))

        # Sync normalization stats into the eval env, then evaluate.
        eval_env.obs_rms = train_env.obs_rms
        results = _evaluate_model(model, eval_env, eval_days)
        score = compute_score(results, theta)
        logger.info("  Trial %d rung %d (%dK steps): Score=%.1f PnL=%.1f "
                    "imb=%.1f thru=%.1f", trial.number, rung_idx,
                    cum_steps // 1000, score, results["mean_pnl"],
                    results["mean_imbalance_cost"], results["mean_throughput"])

        trial.report(score, step=cum_steps)
        if trial.should_prune():
            del model, train_env, eval_env
            gc.collect()
            raise optuna.TrialPruned()

    del model, train_env, eval_env
    gc.collect()
    return results


def train_from_scratch_and_eval(
    theta: Dict[str, float],
    train_prices: Dict,
    eval_prices: Dict,
    eval_days: List,
    scenario_provider,
    engine_config: dict,
    n_steps: int,
    trial_dir: Path,
    seed: int = BASE_SEED,
) -> Dict[str, float]:
    """Train a fresh PPO under θ from scratch and evaluate (used for the
    final winner retrain on train+val — report §5.3)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    train_env = VecNormalize(
        DummyVecEnv([_make_train_env_fn(
            theta, train_prices, scenario_provider, engine_config, seed=seed)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
    )
    model = PPO(policy=SharedEncoderPolicy, env=train_env,
                seed=seed, **PPO_KWARGS)
    model.learn(total_timesteps=n_steps, progress_bar=False)

    model_path   = trial_dir / "model.zip"
    vecnorm_path = trial_dir / "vecnorm.pkl"
    model.save(str(model_path))
    train_env.save(str(vecnorm_path))

    eval_env = VecNormalize.load(
        str(vecnorm_path),
        DummyVecEnv([_make_eval_env_fn(
            theta, eval_prices, scenario_provider, engine_config, seed=seed)]),
    )
    results = _evaluate_model(model, eval_env, eval_days)

    del model, train_env, eval_env
    gc.collect()
    return results


def compute_score(
    results: Dict[str, float],
    theta: Dict[str, float],
) -> float:
    """Compute the meta-layer objective (battery variant).

    Score = mean(economic_PnL)
          − W_IMB   × mean(imbalance_cost)
          − RHO_CVAR × CVaR_0.1(economic_PnL lower tail)
          − RHO_REG  × ||θ||²
    """
    pnl_arr = np.array(results["pnl_list"])
    mean_pnl = float(pnl_arr.mean())
    imb_mean = float(results.get("mean_imbalance_cost", 0.0))

    # CVaR of daily economic PnL (lower tail = worst days)
    if len(pnl_arr) >= 3:
        alpha_q = np.quantile(pnl_arr, CVAR_ALPHA)
        tail = pnl_arr[pnl_arr <= alpha_q]
        cvar = float(tail.mean()) if len(tail) > 0 else float(alpha_q)
    else:
        cvar = mean_pnl

    # L2 regularization on θ (distance from defaults)
    theta_vec = np.array([
        theta["lambda_shaping"],
        theta["lambda_residual"] * 100,   # scale up so it's ~1.0
        theta["lambda_cvar"],
    ])
    reg = float(np.sum(theta_vec ** 2))

    score = (
        mean_pnl
        - W_IMB * imb_mean
        - RHO_CVAR * abs(cvar)
        - RHO_REG * reg
    )

    logger.info(
        "  Score=%.1f  (PnL=%.1f, imb=%.1f, CVaR=%.1f, reg=%.2f)  "
        "θ=(shp=%.2f, res=%.4f, cvar=%.2f)",
        score, mean_pnl, imb_mean, cvar, reg,
        theta["lambda_shaping"], theta["lambda_residual"],
        theta["lambda_cvar"],
    )
    return score


# ══════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════

# These will be set in main() before the study starts
_data_cache = {}


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: sample θ, fine-tune the base model with
    multi-fidelity + pruning, return the validation Score."""
    theta = {
        "lambda_shaping":  trial.suggest_float("lambda_shaping", 1.0, 5.0, log=True),
        "lambda_residual": trial.suggest_float("lambda_residual", 0.001, 0.1, log=True),
        "lambda_cvar":     trial.suggest_float("lambda_cvar", 0.0, 1.0),
    }

    logger.info(
        "Trial %d: θ=(shp=%.3f, res=%.4f, cvar=%.3f)",
        trial.number, theta["lambda_shaping"],
        theta["lambda_residual"], theta["lambda_cvar"],
    )

    t0 = time.time()
    # Per-trial seed → reproducible yet distinct fine-tuning per trial.
    trial_seed = BASE_SEED + 1 + trial.number
    results = finetune_trial(
        theta=theta,
        trial=trial,
        train_prices=_data_cache["train_prices"],
        eval_prices=_data_cache["val_prices"],
        eval_days=_data_cache["val_days"],
        scenario_provider=_data_cache["scenario_provider"],
        engine_config=_data_cache["engine_config"],
        base_model=_data_cache["base_model"],
        base_vecnorm=_data_cache["base_vecnorm"],
        seed=trial_seed,
    )
    score = compute_score(results, theta)
    logger.info("  Trial %d done (%.1f min): Score=%.1f PnL=%.1f imb=%.1f",
                trial.number, (time.time() - t0) / 60, score,
                results["mean_pnl"], results.get("mean_imbalance_cost", 0.0))

    trial.set_user_attr("final_score", score)
    trial.set_user_attr("mean_pnl", results["mean_pnl"])
    trial.set_user_attr("mean_imbalance_cost", results.get("mean_imbalance_cost", 0.0))
    trial.set_user_attr("mean_throughput", results.get("mean_throughput", 0.0))
    trial.set_user_attr("theta", theta)
    return score


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. Global reproducibility ─────────────────────────────────────
    set_seed(BASE_SEED)

    # ── 1. Load data once ─────────────────────────────────────────────
    (
        train_prices, val_prices, test_prices,
        train_days, val_days, test_days,
        scenario_provider, engine_config,
    ) = load_data()

    # ── 1b. Train (or reuse) the shared base model — report §5.2 ──────
    base_model, base_vecnorm = train_base_model(
        train_prices, scenario_provider, engine_config)

    # Store in module-level cache for the objective function
    _data_cache["train_prices"]      = train_prices
    _data_cache["val_prices"]        = val_prices
    _data_cache["test_prices"]       = test_prices
    _data_cache["train_days"]        = train_days
    _data_cache["val_days"]          = val_days
    _data_cache["test_days"]         = test_days
    _data_cache["scenario_provider"] = scenario_provider
    _data_cache["engine_config"]     = engine_config
    _data_cache["base_model"]        = base_model
    _data_cache["base_vecnorm"]      = base_vecnorm

    # ── 2. Run Optuna study ───────────────────────────────────────────
    logger.info("═" * 70)
    logger.info("Starting Optuna meta-layer optimization")
    logger.info("  Trials: %d, fine-tune rungs (cumulative steps): %s",
                N_TRIALS, [f"{s//1000}K" for s in CUMULATIVE_RUNGS])
    logger.info("  Search: λ_shaping∈[1,5] (log), λ_residual∈[0.001,0.1] "
                "(log), λ_CVaR∈[0,1]")
    logger.info("  Score = PnL − %.1f×imbalance − %.1f×|CVaR| − %.2f×||θ||²",
                W_IMB, RHO_CVAR, RHO_REG)
    logger.info("  Data: train=%d, val=%d, test=%d days",
                len(train_days), len(val_days), len(test_days))
    logger.info("═" * 70)

    if STUDY_DB.exists():
        STUDY_DB.unlink()
        logger.info("Deleted old study DB (fresh start)")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(seed=BASE_SEED, n_startup_trials=5),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=CUMULATIVE_RUNGS[0],
            max_resource=CUMULATIVE_RUNGS[-1],
            reduction_factor=2,
        ),
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=False,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ── 3. Extract best θ ─────────────────────────────────────────────
    best = study.best_trial
    best_theta = best.user_attrs.get("theta", {
        "lambda_shaping":  best.params["lambda_shaping"],
        "lambda_residual": best.params["lambda_residual"],
        "lambda_cvar":     best.params["lambda_cvar"],
    })

    logger.info("═" * 70)
    logger.info("Best trial: #%d", best.number)
    logger.info("  θ* = (shp=%.3f, res=%.4f, cvar=%.3f)",
                best_theta["lambda_shaping"],
                best_theta["lambda_residual"],
                best_theta["lambda_cvar"])
    logger.info("  Score = %.2f", best.value)
    logger.info("  PnL = %.1f, imbalance_cost = %.1f, throughput = %.1f",
                best.user_attrs.get("mean_pnl", 0),
                best.user_attrs.get("mean_imbalance_cost", 0),
                best.user_attrs.get("mean_throughput", 0))
    logger.info("═" * 70)

    # Save best theta
    theta_path = OUTPUT_DIR / "best_theta.json"
    with open(theta_path, "w") as f:
        json.dump({
            "theta": best_theta,
            "score": best.value,
            "trial_number": best.number,
            "val_mean_pnl": best.user_attrs.get("mean_pnl", None),
            "val_mean_imbalance_cost": best.user_attrs.get("mean_imbalance_cost", None),
            "val_mean_throughput": best.user_attrs.get("mean_throughput", None),
        }, f, indent=2)
    logger.info("Saved best θ to %s", theta_path)

    # ── 4. Retrain winner on train+val with full budget ───────────────
    logger.info("Retraining best θ* on train+val (%d days), %dK steps…",
                len(train_days) + len(val_days), FINAL_STEPS // 1000)

    trainval_prices = {**train_prices, **val_prices}
    trainval_days   = train_days + val_days

    final_dir = OUTPUT_DIR / "best_model"
    final_results = train_from_scratch_and_eval(
        theta=best_theta,
        train_prices=trainval_prices,
        eval_prices=test_prices,
        eval_days=test_days,
        scenario_provider=scenario_provider,
        engine_config=engine_config,
        n_steps=FINAL_STEPS,
        trial_dir=final_dir,
        seed=BASE_SEED,
    )

    final_score = compute_score(final_results, best_theta)

    # Move final artifacts to expected locations
    src_model   = final_dir / "model.zip"
    src_vecnorm = final_dir / "vecnorm.pkl"
    dst_model   = final_dir / "best_model.zip"
    dst_vecnorm = OUTPUT_DIR / "xbid_vecnormalize.pkl"

    if src_model.exists():
        shutil.copy2(src_model, dst_model)
    if src_vecnorm.exists():
        shutil.copy2(src_vecnorm, dst_vecnorm)

    # ── 5. Print final report ─────────────────────────────────────────
    logger.info("═" * 70)
    logger.info("PHASE 3 COMPLETE — Meta-Layer Results")
    logger.info("═" * 70)
    logger.info("  Best θ*: λ_shaping=%.3f, λ_residual=%.4f, λ_CVaR=%.3f",
                best_theta["lambda_shaping"],
                best_theta["lambda_residual"],
                best_theta["lambda_cvar"])
    logger.info("  Val score:  %.2f", best.value)
    logger.info("  Test score: %.2f", final_score)
    logger.info("  Test PnL/day:        %.1f €", final_results["mean_pnl"])
    logger.info("  Test imbalance cost: %.1f  (primary KPI: undeliverable settlement €)",
                final_results.get("mean_throughput", 0.0))
    logger.info("  Test throughput:     %.1f  (MWh traded intraday)",
                final_results.get("mean_imbalance_cost", 0.0))

    if final_results["pnl_list"]:
        pnl_arr = np.array(final_results["pnl_list"])
        logger.info("  Test PnL stats:  mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
                     pnl_arr.mean(), pnl_arr.std(), pnl_arr.min(), pnl_arr.max())

    logger.info("")
    logger.info("  Model:     %s", dst_model)
    logger.info("  VecNorm:   %s", dst_vecnorm)
    logger.info("  Theta:     %s", theta_path)
    logger.info("  Study DB:  %s", STUDY_DB)
    logger.info("")
    logger.info("To evaluate on full test suite:")
    logger.info("  XBID_MODEL_PATH=%s \\", dst_model)
    logger.info("  XBID_VECNORM_PATH=%s \\", dst_vecnorm)
    logger.info("  python scripts/evaluate_ppo.py")
    logger.info("═" * 70)

    # ── 6. Print top-5 trials summary ─────────────────────────────────
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value or float("-inf"), reverse=True)

    print("\n  Top-5 trials:")
    print("  " + "─" * 75)
    print(f"  {'#':>4}  {'Score':>8}  {'PnL':>8}  {'Imb€':>8}  "
          f"{'λ_shp':>8}  {'λ_res':>8}  {'λ_cvar':>8}")
    print("  " + "─" * 75)
    for t in completed[:5]:
        th = t.user_attrs.get("theta", t.params)
        print(
            f"  {t.number:4d}  {t.value:8.1f}  "
            f"{t.user_attrs.get('mean_pnl', 0):8.1f}  "
            f"{t.user_attrs.get('mean_imbalance_cost', 0):8.1f}  "
            f"{th.get('lambda_shaping', t.params.get('lambda_shaping', 0)):8.3f}  "
            f"{th.get('lambda_residual', t.params.get('lambda_residual', 0)):8.4f}  "
            f"{th.get('lambda_cvar', t.params.get('lambda_cvar', 0)):8.3f}"
        )
    print("  " + "─" * 75)
    print()


if __name__ == "__main__":
    main()
