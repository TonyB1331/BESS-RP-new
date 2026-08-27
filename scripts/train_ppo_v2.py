"""Phase 2 training script: Custom PPO with shared per-slot encoder
and CVaR reward shaping.

This is a drop-in successor to train_ppo.py.  It reuses all the env
plumbing and data loading from the Phase 1 script, and only swaps:
  * MlpPolicy                 → SharedEncoderPolicy (see shared_policy.py)
  * base XBIDEnv              → CVaRRewardWrapper(XBIDEnv)
  * output paths              → training_output_v2/

All other hyperparameters, reward coefficients, and data filtering
remain identical so the comparison to Phase 1 is apples-to-apples.

Run from xbid_hybrid_trader/ :

    python scripts/train_ppo_v2.py

Outputs:
    scripts/training_output_v2/best_model/best_model.zip
    scripts/training_output_v2/xbid_vecnormalize.pkl
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))

from dam_data_manager import DAMDataManager
from historical_data_loader import HistoricalDataLoader
from xbid_trader.market.xbid_env import XBIDEnv

try:
    from xbid_trader.scenario.historical_scenario_provider import HistoricalScenarioProvider
except ImportError:
    HistoricalScenarioProvider = None

try:
    from bm_data_loader import BMDataLoader
except ImportError:
    BMDataLoader = None

from shared_policy import SharedEncoderPolicy
from cvar_wrapper  import CVaRRewardWrapper
from xbid_trader.utils import set_seed

# Global seed for reproducibility (override via XBID_SEED).
SEED = int(os.environ.get("XBID_SEED", "42"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ppo_v2")


# ── Config ─────────────────────────────────────────────────────────────────

N_EVAL_DAYS     = 14
TOTAL_TIMESTEPS = 2_000_000
CHECKPOINT_FREQ = 50_000
LOG_DIR         = Path("scripts/training_output_v2/logs")
CKPT_DIR        = Path("scripts/training_output_v2/checkpoints")
BEST_MODEL_DIR  = Path("scripts/training_output_v2/best_model")

BM_DATA_PATH    = Path(
    os.environ.get("XBID_BM_PATH", "scripts/Balancing_Market_Data.xlsx")
)

LAMBDA_SHAPING  = 1.0
LAMBDA_POS      = 0.5    # fallback — used when no BM data
LAMBDA_RESIDUAL = 0.01   # scale on real BM economic residual cost

# CVaR shaping weights — see cvar_wrapper.py
CVAR_ALPHA       = 0.10
CVAR_LAMBDA      = 0.5     # reduced from 1.0 — less conservative, more trades
CVAR_WINDOW      = 200
CVAR_WARMUP      = 50

ENGINE_CONFIG = dict(
    event_granularity_seconds=60.0,
    gate_closure_offset_minutes=0.0,
    seed_books_on_init=True,
    initial_half_spread=0.5,
    initial_levels=1,
    initial_level_qty=10.0,
    background_agent_configs=None,
)

PPO_CONFIG = dict(
    learning_rate   = 3e-4,
    n_steps         = 96 * 16,    # 16 day-length rollouts per update
    batch_size      = 96 * 4,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
    tensorboard_log = str(LOG_DIR),
)


def main() -> None:
    set_seed(SEED)
    # ── 1. Data loading (identical to Phase 1) ────────────────────────
    logger.info("Loading DAM price data…")
    manager = DAMDataManager(cache_dir="scripts/dam_cache")

    hist_loader = None
    excel_path_env = Path(
        os.environ.get("XBID_EXCEL_PATH", "scripts/Διπλωματική_-_XBID_Trading.xlsx")
    )
    if excel_path_env.exists():
        hist_loader = HistoricalDataLoader(str(excel_path_env))
        hist_loader.load()
        manager.load_from_excel(hist_loader, overwrite=False)
    manager.fetch_all()
    all_prices = manager.get_prices()
    if not all_prices:
        logger.error("No DAM data available.")
        return
    all_days = sorted(all_prices.keys())

    # BM loader
    bm_loader = None
    if BMDataLoader is not None and BM_DATA_PATH.exists():
        try:
            bm_loader = BMDataLoader(BM_DATA_PATH)
            bm_loader.load()
            logger.info("BMDataLoader loaded — %d BM days available",
                        len(bm_loader.available_dates()))
        except Exception as exc:
            logger.warning("Could not load BM data (%s)", exc)
            bm_loader = None

    # Scenario provider
    scenario_provider = None
    if HistoricalScenarioProvider is not None and hist_loader is not None:
        try:
            scenario_provider = HistoricalScenarioProvider(
                hist_loader, bm_loader=bm_loader
            )
            logger.info(
                "HistoricalScenarioProvider loaded — %d days available%s",
                len(scenario_provider),
                " (with BM data)" if bm_loader is not None else "",
            )
            calibrated_agents = hist_loader.calibrate_engine_config()
            if calibrated_agents:
                ENGINE_CONFIG["background_agent_configs"] = calibrated_agents
                logger.info("Background agents calibrated from XBID data.")
        except Exception as exc:
            logger.warning("Could not load HistoricalScenarioProvider (%s)", exc)

    # Filter days to intersection with Excel
    if scenario_provider is not None:
        hist_days_set = set(scenario_provider.available_dates())
        usable_days = [d for d in all_days if d in hist_days_set]
        n_dropped = len(all_days) - len(usable_days)
        if n_dropped > 0:
            logger.info(
                "Filtered out %d days without historical scenario data "
                "(%d → %d).", n_dropped, len(all_days), len(usable_days),
            )
        if len(usable_days) <= N_EVAL_DAYS:
            logger.error(
                "Only %d historical days available — need > %d.",
                len(usable_days), N_EVAL_DAYS,
            )
            return
        all_days = usable_days

    train_days = all_days[:-N_EVAL_DAYS]
    eval_days  = all_days[-N_EVAL_DAYS:]
    logger.info("Training days: %s → %s (%d days)",
                train_days[0], train_days[-1], len(train_days))
    logger.info("Evaluation days: %s → %s (%d days)",
                eval_days[0], eval_days[-1], len(eval_days))
    train_prices = {d: all_prices[d] for d in train_days}
    eval_prices  = {d: all_prices[d] for d in eval_days}

    # ── 2. Env builders — CVaR wrapped ────────────────────────────────
    def make_train_env():
        base = XBIDEnv(
            dam_prices_by_day=train_prices,
            engine_config=ENGINE_CONFIG,
            scenario_provider=scenario_provider,
            lambda_shaping=LAMBDA_SHAPING,
            lambda_pos=LAMBDA_POS,
            lambda_residual=LAMBDA_RESIDUAL,
            forecast_seed=SEED,
        )
        return Monitor(CVaRRewardWrapper(
            base,
            alpha=CVAR_ALPHA,
            lambda_cvar=CVAR_LAMBDA,
            window=CVAR_WINDOW,
            warmup_episodes=CVAR_WARMUP,
        ))

    def make_eval_env():
        # Eval env does NOT apply CVaR shaping — we want to see the real
        # economic return, not the CVaR-adjusted one
        base = XBIDEnv(
            dam_prices_by_day=eval_prices,
            engine_config=ENGINE_CONFIG,
            scenario_provider=scenario_provider,
            lambda_shaping=LAMBDA_SHAPING,
            lambda_pos=LAMBDA_POS,
            lambda_residual=LAMBDA_RESIDUAL,
            scenario_seed=42,
            forecast_seed=SEED,
        )
        return Monitor(base)

    train_env = VecNormalize(
        DummyVecEnv([make_train_env]),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
    )
    eval_env = VecNormalize(
        DummyVecEnv([make_eval_env]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
    )

    # ── 3. Sanity check (on unwrapped env) ────────────────────────────
    logger.info("Running environment sanity check...")
    try:
        check_env(make_train_env(), warn=True)
        logger.info("Environment check passed.")
    except Exception as e:
        logger.warning("Environment check warning: %s", e)

    # ── 4. Dirs + callbacks ───────────────────────────────────────────
    for d in [LOG_DIR, CKPT_DIR, BEST_MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    class SyncVecNormalizeCallback(EvalCallback):
        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms
            return super()._on_step()

    eval_callback = SyncVecNormalizeCallback(
        eval_env,
        best_model_save_path=str(BEST_MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=CHECKPOINT_FREQ,
        n_eval_episodes=len(eval_days),
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CKPT_DIR),
        name_prefix="xbid_ppo_v2",
        verbose=1,
    )

    # ── 5. Build and train model with SharedEncoderPolicy ─────────────
    logger.info("Building PPO v2 model (shared encoder + CVaR shaping)…")
    model = PPO(
        policy=SharedEncoderPolicy,
        env=train_env,
        seed=SEED,
        **PPO_CONFIG,
    )

    logger.info(
        "Starting training — %d timesteps (~%d episodes), "
        "CVaR α=%.2f λ=%.2f window=%d",
        TOTAL_TIMESTEPS, TOTAL_TIMESTEPS // 96,
        CVAR_ALPHA, CVAR_LAMBDA, CVAR_WINDOW,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    final_path   = Path("scripts/training_output_v2/xbid_ppo_final")
    vecnorm_path = Path("scripts/training_output_v2/xbid_vecnormalize.pkl")
    model.save(str(final_path))
    train_env.save(str(vecnorm_path))
    logger.info("Training complete. Model: %s  VecNormalize: %s",
                final_path, vecnorm_path)

    # ── 6. Quick evaluation summary ───────────────────────────────────
    logger.info("Running final evaluation on holdout days...")
    eval_env.obs_rms = train_env.obs_rms
    obs = eval_env.reset()
    episode_returns = []
    current_return  = 0.0
    for _ in range(len(eval_days) * 96):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        current_return += float(reward[0])
        if done[0]:
            episode_returns.append(current_return)
            current_return = 0.0

    if episode_returns:
        r = np.asarray(episode_returns)
        logger.info(
            "Eval: mean=%.2f std=%.2f min=%.2f max=%.2f  "
            "CVaR_0.1(empirical)=%.2f",
            r.mean(), r.std(), r.min(), r.max(),
            r[r <= np.quantile(r, 0.1)].mean() if len(r) >= 10 else float("nan"),
        )


if __name__ == "__main__":
    main()
