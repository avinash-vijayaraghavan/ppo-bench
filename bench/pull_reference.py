"""
Pull reference learning curves (episodic return vs global step) from the openrlbenchmark wandb projects.

    uv run python bench/pull_reference.py            # pull every missing CSV
    uv run python bench/pull_reference.py --force    # re-pull everything

Writes bench/reference/<source>_<algo>_<env_id>.csv with columns: run, global_step, charts/episodic_return.
  cleanrl:   openrlbenchmark/cleanrl, filtered by config.env_id + config.exp_name (all seeds, 2022 + 2023 batches)
  baselines: openrlbenchmark/baselines, runs named <env_id>__baselines-ppo2-{cnn,mlp}__<seed>__<time> (5 seeds)
"""

import argparse
import os

import pandas as pd
import wandb

OUT_DIR = os.path.join(os.path.dirname(__file__), "reference")
COLS = ["run", "global_step", "charts/episodic_return"]

# (source, algo, env_id): wandb filter
REFERENCES = {
    # classic control (baselines has no classic-control runs)
    ("cleanrl", "ppo", "CartPole-v1"): {"config.env_id": "CartPole-v1", "config.exp_name": "ppo"},
    ("cleanrl", "ppo", "Acrobot-v1"): {"config.env_id": "Acrobot-v1", "config.exp_name": "ppo"},
    ("cleanrl", "ppo", "MountainCar-v0"): {"config.env_id": "MountainCar-v0", "config.exp_name": "ppo"},
    # Atari, 10M steps
    **{
        ("cleanrl", "ppo_atari", env): {"config.env_id": env, "config.exp_name": "ppo_atari"}
        for env in ["BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
    },
    **{
        ("baselines", "ppo2", env): {"display_name": {"$regex": f"^{env}__baselines-ppo2-cnn__"}}
        for env in ["BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
    },
    # MuJoCo, 1M steps (CleanRL ran v4; baselines ran v2 with mujoco-py)
    **{
        ("cleanrl", "ppo_continuous_action", env): {"config.env_id": env, "config.exp_name": "ppo_continuous_action"}
        for env in ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
    },
    **{
        ("baselines", "ppo2", env): {"display_name": {"$regex": f"^{env}__baselines-ppo2-mlp__"}}
        for env in ["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2"]
    },
}


def pull(api, source, filters):
    rows = []
    for r in api.runs(f"openrlbenchmark/{source}", filters=filters, per_page=50):
        h = r.history(keys=["global_step", "charts/episodic_return"], samples=50_000, pandas=True).dropna()
        h["run"] = r.name
        rows.append(h[COLS])
        print(f"  {r.name:65s} {len(h):6d} episodes, max step {int(h.global_step.max()):,}")
    return pd.concat(rows).sort_values(["run", "global_step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-pull CSVs that already exist")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    api = wandb.Api(timeout=120)
    for (source, algo, env_id), filters in REFERENCES.items():
        path = os.path.join(OUT_DIR, f"{source}_{algo}_{env_id}.csv")
        if os.path.exists(path) and not args.force:
            print(f"skip {path} (exists)")
            continue
        print(f"pull {source} {algo} {env_id}")
        df = pull(api, source, filters)
        df.to_csv(path, index=False)
        print(f"  -> {path}: {df.run.nunique()} runs")


if __name__ == "__main__":
    main()
