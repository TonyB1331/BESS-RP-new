#!/usr/bin/env python3
"""Imitation-learning warm-start for the battery agent, then PPO fine-tune.

RL-from-scratch struggles on this 200-dim continuous-control problem (the agent
churns). But we have a perfect-foresight ORACLE (the day-ahead LP applied to the
intraday price curve), so we can:

  1. generate oracle target dispatches for every training day;
  2. roll out each day applying those targets, recording (observation, target)
     pairs;
  3. behaviourally-clone (supervised MSE) the policy to output the oracle
     targets;
  4. optionally fine-tune with PPO from that warm start.

Because the dispatch is idempotent (aim at a stable target → trade once → stop),
a policy that reproduces the oracle target realises the arbitrage directly.

Outputs ``model.zip`` + ``vecnorm.pkl`` compatible with ``evaluate_battery.py``.
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
from xbid_trader.market.xbid_env import XBIDEnv, MAX_SLOTS
from xbid_trader.market.battery import BatteryConfig, lp_optimal_dispatch
from xbid_trader.market.rl_observation import N_FEATURES


def oracle_action_for_day(env, day_prices, cfg, aggression=0.5):
    """Return the full (MAX_SLOTS*2,) oracle target action for the current day.

    Must be called right after ``env.reset()`` so ``env._id_reference`` holds
    the intraday curve the order book is anchored to.
    """
    n = env._n_slots
    id_ref = env._id_reference[:n]
    net = lp_optimal_dispatch(id_ref, cfg)          # optimal net MWh per slot
    p_slot = env.battery.power_per_slot_mwh
    frac = np.clip(net / p_slot, -1.0, 1.0)
    a = np.zeros(MAX_SLOTS * 2, dtype=np.float32)
    a[0:2 * n:2] = frac
    a[1:2 * n:2] = aggression
    return a


def collect_bc_dataset(prices, provider, cfg, days, seed, decision_interval,
                       id_sigma, id_rho):
    """Roll out each day with oracle targets; return (obs_array, action_array)."""
    obs_list, act_list = [], []
    for day in days:
        env = XBIDEnv(
            dam_prices_by_day={day: prices[day]}, scenario_provider=provider,
            battery_config=cfg, forecast_seed=seed,
            id_revision_sigma_frac=id_sigma, id_revision_rho=id_rho,
            decision_interval=decision_interval,
        )
        obs, _ = env.reset()
        target = oracle_action_for_day(env, prices[day], cfg)
        done = False
        while not done:
            obs_list.append(obs.astype(np.float32))
            act_list.append(target.astype(np.float32))
            obs, _, done, _, _ = env.step(target)
    return np.asarray(obs_list), np.asarray(act_list)


def main() -> None:
    ap = argparse.ArgumentParser(description="BC warm-start + PPO fine-tune.")
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--train-months", nargs="+", default=["2026-01", "2026-02", "2026-03"])
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--bc-batch", type=int, default=256)
    ap.add_argument("--bc-lr", type=float, default=3e-4)
    ap.add_argument("--finetune-steps", type=int, default=0,
                    help="PPO timesteps after BC (0 = BC only)")
    ap.add_argument("--id-revision-sigma", type=float, default=0.12)
    ap.add_argument("--id-revision-rho", type=float, default=0.8)
    ap.add_argument("--max-cycles", type=float, default=2.0)
    ap.add_argument("--decision-interval", type=int, default=8)
    ap.add_argument("--lambda-shaping", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="battery_bc_out")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from shared_policy import SharedEncoderPolicy
    from cvar_wrapper import CVaRRewardWrapper

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    prices, provider = load_dam_dataset(args.xlsx, months=args.train_months)
    days = provider.days
    cfg = BatteryConfig()
    cfg.max_cycles_per_day = args.max_cycles

    # ── 1-2. Build the BC dataset from oracle targets ─────────────────────
    print(f"Generating oracle BC dataset over {len(days)} days …")
    obs_arr, act_arr = collect_bc_dataset(
        prices, provider, cfg, days, args.seed,
        args.decision_interval, args.id_revision_sigma, args.id_revision_rho)
    print(f"  collected {len(obs_arr):,} (obs, target) pairs")

    # ── Build the (vec-normalised) env the PPO model will use ─────────────
    def env_factory():
        base = XBIDEnv(
            dam_prices_by_day=prices, scenario_provider=provider, battery_config=cfg,
            forecast_seed=args.seed, lambda_shaping=args.lambda_shaping,
            id_revision_sigma_frac=args.id_revision_sigma,
            id_revision_rho=args.id_revision_rho,
            decision_interval=args.decision_interval,
                reveal_window=args.reveal_window,
        )
        return Monitor(CVaRRewardWrapper(base, alpha=0.1, lambda_cvar=0.2,
                                         window=64, warmup_episodes=16))

    venv = VecNormalize(DummyVecEnv([env_factory]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)
    # Seed obs-normalisation stats from the BC dataset so BC and PPO agree.
    venv.obs_rms.mean = obs_arr.mean(axis=0)
    venv.obs_rms.var = obs_arr.var(axis=0) + 1e-8
    venv.obs_rms.count = float(len(obs_arr))

    model = PPO(policy=SharedEncoderPolicy, env=venv, seed=args.seed,
                n_steps=2048, batch_size=256, learning_rate=3e-4,
                gamma=0.999, ent_coef=0.005, verbose=0, device="cpu")

    # ── 3. Behavioural cloning: regress policy mean → oracle target ───────
    device = model.policy.device
    obs_n = np.clip((obs_arr - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var),
                    -10.0, 10.0).astype(np.float32)
    X = torch.as_tensor(obs_n, device=device)
    Y = torch.as_tensor(act_arr, device=device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=args.bc_lr)

    print(f"Behavioural cloning: {args.bc_epochs} epochs on {len(X):,} samples …")
    n = len(X)
    for ep in range(args.bc_epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.bc_batch):
            idx = perm[i:i + args.bc_batch]
            dist = model.policy.get_distribution(X[idx])
            mean = dist.distribution.mean
            loss = F.mse_loss(mean, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{args.bc_epochs}  MSE={tot/n:.4f}")

    model.save(str(out / "model"))
    venv.save(str(out / "vecnorm.pkl"))
    print(f"\nBC warm-start saved → {out/'model.zip'} , {out/'vecnorm.pkl'}")

    # ── 4. Optional PPO fine-tune from the warm start ─────────────────────
    if args.finetune_steps > 0:
        print(f"PPO fine-tuning for {args.finetune_steps} timesteps …")
        model.learn(total_timesteps=args.finetune_steps, progress_bar=True)
        model.save(str(out / "model"))
        venv.save(str(out / "vecnorm.pkl"))
        print(f"Fine-tuned model saved → {out/'model.zip'}")


if __name__ == "__main__":
    main()
