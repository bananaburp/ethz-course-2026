"""
Parallelized SAC hyperparameter grid search.

Runs multiple training trials in parallel and summarizes which hyperparameter
combinations yield the best eval returns.

Usage:
    # Run grid search with 4 parallel workers
    python scripts/sac_grid_search.py --num_workers 4

    # Quick search with fewer iterations per trial
    python scripts/sac_grid_search.py --num_workers 4 --total_iterations 300

    # Print summary from a previous run without retraining
    python scripts/sac_grid_search.py --summarize_only --results_dir logs/sac_grid_search/2026_04_07_...
"""

import argparse
import csv
import itertools
import json
import os
import sys
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Hyperparameter grid — edit this to control what gets searched
# ---------------------------------------------------------------------------
PARAM_GRID = {
    # (train_freq, gradient_steps) are paired: UTD = gradient_steps / train_freq
    "train_freq_gradient_steps": [
        (50, 50),    # UTD = 1.0  — high-frequency updates
        (100, 100),  # UTD = 1.0  — medium frequency
        (500, 200),  # UTD = 0.4  — original config
    ],
    "hidden_sizes": [
        [256, 128, 128],   # original
        [256, 256, 256],   # larger
    ],
    "actor_lr": [3e-4, 1e-4],
    "init_alpha": [0.1, 0.2],
}

# Base config — values not in PARAM_GRID stay fixed
BASE_CONFIG = {
    "batch_size": 256,
    "learning_start_steps": 500,
    "eval_freq": 50,       # eval every N training iterations (not env steps)
    "replay_size": 200_000,
    "gamma": 0.99,
    "tau": 0.005,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "target_entropy": None,
}
# ---------------------------------------------------------------------------


def _expand_grid(grid: dict) -> list[dict]:
    """Expand PARAM_GRID into a flat list of config dicts."""
    # Handle paired params separately
    paired = grid.pop("train_freq_gradient_steps")
    keys = list(grid.keys())
    values = list(grid.values())

    combos = []
    for pair in paired:
        for combo in itertools.product(*values):
            cfg = dict(zip(keys, combo))
            cfg["train_freq"], cfg["gradient_steps"] = pair
            combos.append(cfg)

    # Restore for potential re-use
    grid["train_freq_gradient_steps"] = paired
    return combos


def _evaluate_policy(env, agent, device, num_episodes=10):
    import torch
    import numpy as np

    returns = []
    tracking_errors = []
    agent.eval_mode()
    with torch.inference_mode():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
                action = agent.predict_action(obs_t)
                obs, reward, terminated, truncated, info = env.step(
                    action.cpu().numpy().squeeze(0)
                )
                ep_return += reward
                done = terminated or truncated
            returns.append(ep_return)
            tracking_errors.append(info["ee_tracking_error"])
    return float(np.mean(returns)), float(np.mean(tracking_errors))


def run_trial(args: tuple) -> dict:
    """
    Worker function. Runs one SAC training trial and returns a result dict.
    Designed to be called in a subprocess via multiprocessing.Pool.
    """
    trial_id, config, run_dir_str, total_iterations, seed = args
    run_dir = Path(run_dir_str)
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "trial_id": trial_id,
        "config": config,
        "status": "failed",
        "best_return": None,
        "final_return": None,
        "best_tracking_error": None,
        "eval_returns": [],
        "run_dir": run_dir_str,
    }

    try:
        import numpy as np
        import torch
        from envs.so100_rl_env import SO100RLEnv
        from exercises.ex4_sac import SACAgent, SACUpdateStats
        from rl.buffers import ReplayBuffer
        from rl.common import set_seed

        set_seed(seed)
        device = torch.device("cpu")  # CPU per worker to avoid GPU contention

        xml_path = ROOT_DIR / "assets" / "mujoco" / "so100_pos_ctrl.xml"
        env = SO100RLEnv(xml_path=xml_path, render_mode=None)
        eval_env = SO100RLEnv(xml_path=xml_path, render_mode=None)

        agent = SACAgent(
            obs_dim=env.state_dim,
            act_dim=env.action_dim,
            hidden_sizes=config["hidden_sizes"],
            actor_lr=config["actor_lr"],
            critic_lr=config["critic_lr"],
            alpha_lr=config["alpha_lr"],
            gamma=config["gamma"],
            tau=config["tau"],
            init_alpha=config["init_alpha"],
            target_entropy=config["target_entropy"],
            device=device,
        )

        replay_buffer = ReplayBuffer(
            obs_dim=env.state_dim,
            act_dim=env.action_dim,
            max_size=config["replay_size"],
            device=device,
        )

        train_freq = config["train_freq"]
        gradient_steps = config["gradient_steps"]
        batch_size = config["batch_size"]
        learning_start_steps = config["learning_start_steps"]
        eval_freq = config["eval_freq"]  # in training iterations

        obs, _ = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float, device=device).unsqueeze(0)

        step = 0
        it = 0
        best_return = -float("inf")
        best_tracking_error = float("inf")
        eval_returns = []

        while it < total_iterations:
            agent.train_mode()
            with torch.no_grad():
                step += 1
                if step < learning_start_steps:
                    action = torch.empty(env.action_dim, dtype=torch.float, device=device).uniform_(-1.0, 1.0).unsqueeze(0)
                else:
                    action = agent.sample_action(obs)

                next_obs, reward, terminated, truncated, info = env.step(
                    action.cpu().numpy().squeeze(0)
                )
                next_obs = torch.as_tensor(next_obs, dtype=torch.float, device=device).unsqueeze(0)
                done = terminated or truncated

                replay_buffer.store(
                    obs=obs.squeeze(0),
                    act=action.squeeze(0),
                    rew=reward,
                    next_obs=next_obs.squeeze(0),
                    done=done,
                )

                obs = next_obs
                if done:
                    obs, _ = env.reset()
                    obs = torch.as_tensor(obs, dtype=torch.float, device=device).unsqueeze(0)

            if step >= learning_start_steps and step % train_freq == 0:
                for _ in range(gradient_steps):
                    batch = replay_buffer.sample_batch(batch_size=batch_size)
                    agent.update(batch)
                it += 1

                if it % eval_freq == 0 or it == total_iterations:
                    mean_return, mean_tracking_error = _evaluate_policy(
                        eval_env, agent, device, num_episodes=5
                    )
                    eval_returns.append({"it": it, "return": mean_return, "tracking_error": mean_tracking_error})

                    if mean_return > best_return:
                        best_return = mean_return
                        best_tracking_error = mean_tracking_error

                    print(
                        f"[trial {trial_id:03d}] it={it}/{total_iterations} "
                        f"return={mean_return:.2f} (best={best_return:.2f}) "
                        f"err={mean_tracking_error:.4f}"
                    )

        env.close()
        eval_env.close()

        final_return = eval_returns[-1]["return"] if eval_returns else None
        result.update({
            "status": "ok",
            "best_return": best_return,
            "final_return": final_return,
            "best_tracking_error": best_tracking_error,
            "eval_returns": eval_returns,
        })

    except Exception:
        result["traceback"] = traceback.format_exc()
        print(f"[trial {trial_id:03d}] FAILED:\n{result['traceback']}")

    # Save per-trial JSON result
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def summarize_results(results: list[dict], output_dir: Path) -> None:
    """Print a ranked summary table and save as CSV."""
    valid = [r for r in results if r["status"] == "ok" and r["best_return"] is not None]
    failed = [r for r in results if r["status"] != "ok"]

    if not valid:
        print("No successful trials to summarize.")
        return

    valid.sort(key=lambda r: r["best_return"], reverse=True)

    # --- Console table ---
    col_w = {
        "rank": 5, "trial": 6, "best": 8, "final": 8, "err": 8,
        "train_freq": 11, "grad_steps": 11, "hidden": 18,
        "actor_lr": 10, "init_alpha": 11,
    }
    header = (
        f"{'Rank':>{col_w['rank']}} "
        f"{'Trial':>{col_w['trial']}} "
        f"{'Best':>{col_w['best']}} "
        f"{'Final':>{col_w['final']}} "
        f"{'TrackErr':>{col_w['err']}} "
        f"{'TrainFreq':>{col_w['train_freq']}} "
        f"{'GradSteps':>{col_w['grad_steps']}} "
        f"{'HiddenSizes':<{col_w['hidden']}} "
        f"{'ActorLR':>{col_w['actor_lr']}} "
        f"{'InitAlpha':>{col_w['init_alpha']}}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("SAC GRID SEARCH RESULTS (ranked by best eval return)")
    print(sep)
    print(header)
    print(sep)

    csv_rows = []
    for rank, r in enumerate(valid, 1):
        cfg = r["config"]
        row_str = (
            f"{rank:>{col_w['rank']}} "
            f"{r['trial_id']:>{col_w['trial']}} "
            f"{r['best_return']:>{col_w['best']}.2f} "
            f"{r['final_return']:>{col_w['final']}.2f} "
            f"{r['best_tracking_error']:>{col_w['err']}.4f} "
            f"{cfg['train_freq']:>{col_w['train_freq']}} "
            f"{cfg['gradient_steps']:>{col_w['grad_steps']}} "
            f"{str(cfg['hidden_sizes']):<{col_w['hidden']}} "
            f"{cfg['actor_lr']:>{col_w['actor_lr']}.0e} "
            f"{cfg['init_alpha']:>{col_w['init_alpha']}.2f}"
        )
        print(row_str)
        csv_rows.append({
            "rank": rank,
            "trial_id": r["trial_id"],
            "best_return": round(r["best_return"], 4),
            "final_return": round(r["final_return"], 4),
            "best_tracking_error": round(r["best_tracking_error"], 6),
            "train_freq": cfg["train_freq"],
            "gradient_steps": cfg["gradient_steps"],
            "hidden_sizes": str(cfg["hidden_sizes"]),
            "actor_lr": cfg["actor_lr"],
            "init_alpha": cfg["init_alpha"],
            "run_dir": r["run_dir"],
        })

    print(sep)
    if failed:
        print(f"  {len(failed)} trial(s) failed — check result.json in their run dirs")
    print(f"{'='*len(header)}\n")

    # --- CSV ---
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Summary saved to: {csv_path}")

    # --- Best config snippet ---
    best = valid[0]
    cfg = best["config"]
    print("\nBest config to copy into ex4_sac_config.py:")
    print("-" * 50)
    print(f'    "hidden_sizes": {cfg["hidden_sizes"]},')
    print(f'    "train_freq": {cfg["train_freq"]},')
    print(f'    "gradient_steps": {cfg["gradient_steps"]},')
    print(f'    "actor_lr": {cfg["actor_lr"]},')
    print(f'    "init_alpha": {cfg["init_alpha"]},')
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="SAC hyperparameter grid search")
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel training processes (tune to your CPU; default: 4)"
    )
    parser.add_argument(
        "--total_iterations", type=int, default=400,
        help="Training iterations per trial (default: 400; use fewer for a quick scan)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed; each trial gets seed + trial_id (default: 42)"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Path to an existing grid search output dir (skips training, just prints summary)"
    )
    parser.add_argument(
        "--summarize_only", action="store_true",
        help="Only regenerate the summary from existing result.json files in --results_dir"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Summarize-only mode
    # -----------------------------------------------------------------------
    if args.summarize_only:
        if args.results_dir is None:
            parser.error("--summarize_only requires --results_dir")
        results_dir = Path(args.results_dir)
        results = []
        for result_file in sorted(results_dir.glob("trial_*/result.json")):
            with open(result_file) as f:
                results.append(json.load(f))
        print(f"Loaded {len(results)} results from {results_dir}")
        summarize_results(results, results_dir)
        return

    # -----------------------------------------------------------------------
    # Build trials
    # -----------------------------------------------------------------------
    grid = {k: list(v) for k, v in PARAM_GRID.items()}
    param_combos = _expand_grid(grid)

    # Merge with base config
    trials = []
    for trial_id, combo in enumerate(param_combos):
        cfg = {**BASE_CONFIG, **combo}
        # hidden_sizes must be a plain list (JSON-serializable)
        cfg["hidden_sizes"] = list(cfg["hidden_sizes"])
        trials.append(cfg)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_root = ROOT_DIR / "logs" / "sac_grid_search" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    # Save full grid spec for reference
    with open(run_root / "grid_spec.json", "w") as f:
        json.dump({"total_iterations": args.total_iterations, "trials": trials}, f, indent=2)

    worker_args = [
        (
            trial_id,
            cfg,
            str(run_root / f"trial_{trial_id:03d}"),
            args.total_iterations,
            args.seed + trial_id,
        )
        for trial_id, cfg in enumerate(trials)
    ]

    n_workers = min(args.num_workers, len(worker_args))
    print(f"Grid search: {len(worker_args)} trials, {n_workers} parallel workers")
    print(f"Results dir: {run_root}\n")

    # -----------------------------------------------------------------------
    # Run in parallel
    # -----------------------------------------------------------------------
    if n_workers == 1:
        # Single-process mode — easier to debug
        results = [run_trial(a) for a in worker_args]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(run_trial, worker_args)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summarize_results(results, run_root)


if __name__ == "__main__":
    main()
