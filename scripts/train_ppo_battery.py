"""PPO training for the BESS (battery) XBID intraday agent.

Battery counterpart of ``train_ppo_v2.py``.  Reuses the data loading, the
shared per-slot encoder policy, and the market engine unchanged, and swaps:

  * XBIDEnv (supplier / Rt hedging)  → BatteryXBIDEnv (SoC arbitrage)
  * CVaRRewardWrapper (residual cost) → BatteryCVaRRewardWrapper (downside PnL)

The HistoricalScenarioProvider is reused as-is: the battery env consumes only
its ``prices`` / ``imbalance_prices`` / ``xbid_volume`` fields (the supplier
``rt`` / ``dam_position`` / BM fields are ignored) and builds its own day-ahead
schedule from DAM prices via ``dam_arbitrage_schedule``.

Run from meta_xbid_hybrid_trader/ :

    python scripts/train_ppo_battery.py

Outputs:
    scripts/training_output_battery/best_model/best_model.zip
    scripts/training_output_battery/battery_vecnormalize.pkl
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))

from dam_data_manager import DAMDataManager
from historical_data_loader import HistoricalDataLoader
from xbid_trader.battery import BatteryXBIDEnv, BatterySpec
from battery_cvar_wrapper import BatteryCVaRRewardWrapper
from shared_policy import SharedEncoderPolicy
from xbid_trader.utils import set_seed

try:
    from xbid_trader.scenario.historical_scenario_provider import HistoricalScenarioProvider
except ImportError:
    HistoricalScenarioProvider = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ppo_battery")

SEED            = int(os.environ.get("XBID_SEED", "42"))
N_EVAL_DAYS     = 14
TOTAL_TIMESTEPS = int(os.environ.get("XBID_TIMESTEPS", "2000000"))
CHECKPOINT_FREQ = 50_000
OUT             = Path("scripts/training_output_battery")
LOG_DIR         = OUT / "logs"
CKPT_DIR        = OUT / "checkpoints"
BEST_MODEL_DIR  = OUT / "best_model"

# ── Battery spec + reward weights (see configs/battery.yaml) ────────────────
BATTERY = BatterySpec()
LAMBDA_SHAPING  = 0.1
LAMBDA_DEG      = 1.0
LAMBDA_RESIDUAL = 0.01
LAMBDA_TERMINAL = 1.0

CVAR_ALPHA, CVAR_LAMBDA, CVAR_WINDOW, CVAR_WARMUP = 0.10, 0.5, 200, 50

ENGINE_CONFIG = dict(
    event_granularity_seconds=60.0, gate_closure_offset_minutes=0.0,
    seed_books_on_init=True, initial_half_spread=0.5,
    initial_levels=1, initial_level_qty=10.0, background_agent_configs=None,
)

PPO_CONFIG = dict(
    learning_rate=3e-4, n_steps=96 * 16, batch_size=96 * 4, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, verbose=1, tensorboard_log=str(LOG_DIR),
)


def main() -> None:
    set_seed(SEED)
    for d in (LOG_DIR, CKPT_DIR, BEST_MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DAM price data…")
    manager = DAMDataManager(cache_dir="scripts/dam_cache")
    hist_loader = None
    excel_path = Path(os.environ.get(
        "XBID_EXCEL_PATH", "scripts/Διπλωματική_-_XBID_Trading.xlsx"))
    if excel_path.exists():
        hist_loader = HistoricalDataLoader(str(excel_path))
        hist_loader.load()
        manager.load_from_excel(hist_loader, overwrite=False)
    manager.fetch_all()
    all_prices = manager.get_prices()
    if not all_prices:
        logger.error("No DAM data available.")
        return
    all_days = sorted(all_prices.keys())

    scenario_provider = None
    if HistoricalScenarioProvider is not None and hist_loader is not None:
        try:
            scenario_provider = HistoricalScenarioProvider(hist_loader)
            logger.info("HistoricalScenarioProvider loaded — %d days.",
                        len(scenario_provider))
            calibrated = hist_loader.calibrate_engine_config()
            if calibrated:
                ENGINE_CONFIG["background_agent_configs"] = calibrated
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scenario provider unavailable (%s)", exc)

    if scenario_provider is not None:
        days_set = set(scenario_provider.available_dates())
        all_days = [d for d in all_days if d in days_set] or all_days

    train_days, eval_days = all_days[:-N_EVAL_DAYS], all_days[-N_EVAL_DAYS:]
    train_prices = {d: all_prices[d] for d in train_days}
    eval_prices  = {d: all_prices[d] for d in eval_days}
    logger.info("Train %d days | Eval %d days", len(train_days), len(eval_days))

    def make_train_env():
        base = BatteryXBIDEnv(
            dam_prices_by_day=train_prices, battery_spec=BATTERY,
            engine_config=ENGINE_CONFIG, scenario_provider=scenario_provider,
            lambda_shaping=LAMBDA_SHAPING, lambda_deg=LAMBDA_DEG,
            lambda_residual=LAMBDA_RESIDUAL, lambda_terminal=LAMBDA_TERMINAL,
            forecast_seed=SEED,
        )
        return Monitor(BatteryCVaRRewardWrapper(
            base, alpha=CVAR_ALPHA, lambda_cvar=CVAR_LAMBDA,
            window=CVAR_WINDOW, warmup_episodes=CVAR_WARMUP))

    def make_eval_env():
        base = BatteryXBIDEnv(
            dam_prices_by_day=eval_prices, battery_spec=BATTERY,
            engine_config=ENGINE_CONFIG, scenario_provider=scenario_provider,
            lambda_shaping=LAMBDA_SHAPING, lambda_deg=LAMBDA_DEG,
            lambda_residual=LAMBDA_RESIDUAL, lambda_terminal=LAMBDA_TERMINAL,
            scenario_seed=42, forecast_seed=SEED,
        )
        return Monitor(base)

    train_env = VecNormalize(
        DummyVecEnv([make_train_env]),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    eval_env = VecNormalize(
        DummyVecEnv([make_eval_env]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms

    model = PPO(SharedEncoderPolicy, train_env, seed=SEED, **PPO_CONFIG)
    logger.info("Training battery PPO (shared encoder + downside-CVaR)…")

    callbacks = [
        CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=str(CKPT_DIR),
                           name_prefix="battery_ppo"),
        EvalCallback(eval_env, best_model_save_path=str(BEST_MODEL_DIR),
                     log_path=str(LOG_DIR), eval_freq=CHECKPOINT_FREQ,
                     deterministic=True, render=False),
    ]
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    model.save(str(BEST_MODEL_DIR / "final_model"))
    train_env.save(str(OUT / "battery_vecnormalize.pkl"))
    logger.info("Done. Model + VecNormalize saved under %s", OUT)


if __name__ == "__main__":
    main()
