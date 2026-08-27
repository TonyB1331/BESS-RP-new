#!/usr/bin/env python3
"""Train the battery intraday-arbitrage PPO agent on the real DAM dataset.

Splits the workbook by month: trains on Jan–Mar 2026, validates on a held-out
slice, and saves ``model.zip`` + ``vecnorm.pkl`` for evaluation on April.

Reuses the repo's shared-slot encoder policy (``scripts/shared_policy.py``)
and the battery downside-loss CVaR wrapper (``scripts/cvar_wrapper.py``).

Kaggle usage — see the step-by-step guide printed by ``--help`` and the
README section. Minimal run::

    python scripts/train_battery.py \
        --xlsx /kaggle/input/<dataset>/1_5cyclejan_to_apr_26Results.xlsx \
        --train-months 2026-01 2026-02 2026-03 \
        --timesteps 300000 --out /kaggle/working/battery_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from xbid_trader.data.dam_schedule_loader import load_dam_dataset
from xbid_trader.market.xbid_env import XBIDEnv
from xbid_trader.market.battery import BatteryConfig


def build_env(prices, provider, cfg, cvar_kwargs, seed, monitor=True):
    """Factory returning a Monitor(CVaR(XBIDEnv)) callable."""
    from stable_baselines3.common.monitor import Monitor
    from cvar_wrapper import CVaRRewardWrapper

    def _fn():
        base = XBIDEnv(
            dam_prices_by_day=prices,
            scenario_provider=provider,
            battery_config=cfg,
            forecast_seed=seed,
        )
        wrapped = CVaRRewardWrapper(base, **cvar_kwargs)
        return Monitor(wrapped) if monitor else wrapped
    return _fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Train battery intraday PPO agent.")
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--train-months", nargs="+", default=["2026-01", "2026-02", "2026-03"])
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of train days held out for validation")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="battery_out")
    ap.add_argument("--seed", type=int, default=42)
    # reward weights θ (can be overridden by the meta-layer's best_theta.json)
    ap.add_argument("--lambda-shaping", type=float, default=0.0)
    ap.add_argument("--lambda-residual", type=float, default=0.01)
    ap.add_argument("--lambda-cvar", type=float, default=0.2)
    ap.add_argument("--cvar-alpha", type=float, default=0.1)
    ap.add_argument("--id-revision-sigma", type=float, default=0.12,
                    help="ID price revision std as fraction of |price| (0 = ID echoes DAM)")
    ap.add_argument("--id-revision-rho", type=float, default=0.8,
                    help="AR(1) persistence of the ID revision")
    ap.add_argument("--max-cycles", type=float, default=None,
                    help="override daily cycle limit Cs (DAM uses 1.5; e.g. 2.0 "
                         "gives the intraday layer 0.5 cycle of headroom)")
    ap.add_argument("--reveal-window", type=int, default=24,
                    help="quarters before gate over which a price reveals")
    ap.add_argument("--decision-interval", type=int, default=8,
                    help="re-plan targets every K steps (K>1 prevents wash-trade "
                         "churn; 8 = re-plan every 2 hours)")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback
    from shared_policy import SharedEncoderPolicy

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────
    prices, provider = load_dam_dataset(args.xlsx, months=args.train_months)
    days = provider.days
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(days))
    n_val = max(1, int(len(days) * args.val_frac))
    val_days = [days[i] for i in perm[:n_val]]
    train_days = [days[i] for i in perm[n_val:]]
    train_prices = {d: prices[d] for d in train_days}
    val_prices   = {d: prices[d] for d in val_days}
    print(f"Loaded {len(days)} days  →  train={len(train_days)}  val={len(val_days)}")

    cfg = BatteryConfig()   # real 50 MW / 100 MWh / 1.5-cycle unit
    if args.max_cycles is not None:
        cfg.max_cycles_per_day = args.max_cycles
    cvar_kwargs = dict(alpha=args.cvar_alpha, lambda_cvar=args.lambda_cvar,
                       window=64, warmup_episodes=16)

    # NOTE: lambda_shaping/residual are XBIDEnv reward weights → pass via a
    # small wrapper factory that sets them on the base env.
    def env_factory(pr, pv, seed):
        from stable_baselines3.common.monitor import Monitor
        from cvar_wrapper import CVaRRewardWrapper

        def _fn():
            base = XBIDEnv(
                dam_prices_by_day=pr, scenario_provider=pv, battery_config=cfg,
                forecast_seed=seed,
                lambda_shaping=args.lambda_shaping,
                lambda_residual=args.lambda_residual,
                id_revision_sigma_frac=args.id_revision_sigma,
                id_revision_rho=args.id_revision_rho,
                decision_interval=args.decision_interval,
                reveal_window=args.reveal_window,
            )
            return Monitor(CVaRRewardWrapper(base, **cvar_kwargs))
        return _fn

    train_env = VecNormalize(
        DummyVecEnv([env_factory(train_prices, provider, args.seed)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0,
    )
    val_env = VecNormalize(
        DummyVecEnv([env_factory(val_prices, provider, args.seed + 1)]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
    )

    ppo_cfg = dict(
        n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.999, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, learning_rate=3e-4, vf_coef=0.5,
        max_grad_norm=0.5, verbose=1, device="cpu",
    )
    model = PPO(policy=SharedEncoderPolicy, env=train_env, seed=args.seed, **ppo_cfg)

    class SyncVN(EvalCallback):
        def _on_step(self):
            try:
                self.eval_env.obs_rms = self.training_env.obs_rms
            except Exception:
                pass
            return super()._on_step()

    eval_cb = SyncVN(val_env, best_model_save_path=str(out),
                     log_path=str(out), eval_freq=10_000,
                     n_eval_episodes=max(3, len(val_days)), deterministic=True)

    print(f"Training PPO for {args.timesteps} timesteps (~{args.timesteps//96} episodes)…")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)

    model.save(str(out / "model"))
    train_env.save(str(out / "vecnorm.pkl"))
    print(f"\nSaved model → {out/'model.zip'}  and VecNormalize → {out/'vecnorm.pkl'}")
    print(f"A best_model.zip (highest val reward) is also in {out}/ if eval ran.")


if __name__ == "__main__":
    main()
