"""
Compare our bench runs (runs/<env>__bench_<mode>__<seed>__<time>/episodes.csv) with the reference CSVs.

    uv run python bench/compare.py                 # every env that has bench runs
    uv run python bench/compare.py Hopper-v4       # one env

Metric: mean episodic return over the `window` env steps ending at each checkpoint, per run;
then mean ± std over runs (n). "match" = our final mean lies inside the reference's min..max over runs.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF = os.path.join(ROOT, "bench", "reference")

# our env_id: (checkpoints, window, reference CSVs)
ENVS = {
    "CartPole-v1": ([100_000, 250_000, 500_000], 25_000, ["cleanrl_ppo_CartPole-v1"]),
    "Acrobot-v1": ([100_000, 250_000, 500_000], 25_000, ["cleanrl_ppo_Acrobot-v1"]),
    "MountainCar-v0": ([100_000, 250_000, 500_000], 25_000, ["cleanrl_ppo_MountainCar-v0"]),
    **{
        f"ALE/{game}-v5": (
            [2_500_000, 5_000_000, 7_500_000, 10_000_000],
            250_000,
            [f"cleanrl_ppo_atari_{game}NoFrameskip-v4", f"baselines_ppo2_{game}NoFrameskip-v4"],
        )
        for game in ["Breakout", "Pong", "BeamRider"]
    },
    **{
        f"{env}-v4": (
            [250_000, 500_000, 750_000, 1_000_000],
            50_000,
            [f"cleanrl_ppo_continuous_action_{env}-v4", f"baselines_ppo2_{env}-v2"],
        )
        for env in ["Hopper", "Walker2d", "HalfCheetah"]
    },
}


def load_ours(env_id):
    """{"ours bench_<mode>": DataFrame(run, global_step, r)} from runs/<flat env>__bench_<mode>__<seed>__<time>/."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", f"{env_id.replace('/', '-')}__bench_*", "episodes.csv"))):
        run = os.path.basename(os.path.dirname(path))
        mode = run.split("__")[1]
        df = pd.read_csv(path).rename(columns={"episodic_return": "r"})
        df["run"] = run
        groups.setdefault(f"ours {mode}", []).append(df[["run", "global_step", "r"]])
    return {k: pd.concat(v) for k, v in groups.items()}


def load_ref(name):
    df = pd.read_csv(os.path.join(REF, f"{name}.csv")).rename(columns={"charts/episodic_return": "r"})
    return name.split("_")[0], df  # "cleanrl" / "baselines"


def window_means(df, checkpoint, window):
    """One number per run: mean return over (checkpoint - window, checkpoint]. NaN if the run has no episodes there."""
    return np.array(
        [g.r[(g.global_step > checkpoint - window) & (g.global_step <= checkpoint)].mean() for _, g in df.groupby("run")]
    )


def fmt(vals):
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return "-"
    return f"{vals.mean():.1f} ± {vals.std():.1f} ({len(vals)})"


def compare(env_id):
    checkpoints, window, ref_names = ENVS[env_id]
    sources = {**load_ours(env_id), **dict(load_ref(n) for n in ref_names)}
    print(f"\n== {env_id}: mean return over the {window:,} steps before each checkpoint, mean ± std (n runs)")
    print(f"{'source':22s}" + "".join(f"{c / 1e6:>22.2f}M" for c in checkpoints))
    for name, df in sources.items():
        print(f"{name:22s}" + "".join(f"{fmt(window_means(df, c, window)):>23s}" for c in checkpoints))

    final = checkpoints[-1]
    for ours in [n for n in sources if n.startswith("ours")]:
        mine = window_means(sources[ours], final, window)
        mine = mine[~np.isnan(mine)]
        if len(mine) == 0:
            continue
        for ref in [n for n in sources if not n.startswith("ours")]:
            vals = window_means(sources[ref], final, window)
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            ok = lo <= mine.mean() <= hi
            print(f"  {ours} final {mine.mean():.1f} vs {ref} {lo:.1f}..{hi:.1f}: {'match' if ok else 'OUTSIDE'}")


def main():
    if len(sys.argv) > 1:
        env_ids = sys.argv[1:]
    else:
        env_ids = [e for e in ENVS if load_ours(e)]
        if not env_ids:
            print("no bench runs found in runs/ (expected runs/<env>__bench_<mode>__<seed>__<time>/episodes.csv)")
    for env_id in env_ids:
        compare(env_id)


if __name__ == "__main__":
    main()
