# ppo-bench

My own PPO, written from scratch as single files in CleanRL's style, benchmarked against **CleanRL** and **OpenAI Baselines** (ppo2) on 9 environments, 3 seeds each.

It has two goals:
1. **Fidelity:** with the same settings as the references, my PPO should land inside their range of results.
2. **One fix:** handle episodes cut off by a time limit correctly, and measure what that's worth.

## Results

**Score:** mean episodic return over the last stretch of training (classic control: 25K steps, MuJoCo: 50K, Atari: 250K), then mean ± std over 3 seeds. **"Match"** means my mean falls between the reference's worst and best runs.

### Fidelity mode (time limits handled like CleanRL and Baselines)

| Environment | Steps | Mine | CleanRL | OpenAI Baselines |
|---|---|---|---|---|
| CartPole-v1 | 500K | 494.0 ± 4.5 | 487.5 ± 7.9 | — |
| Acrobot-v1 | 500K | −82.8 ± 1.0 | −83.9 ± 1.2 | — |
| MountainCar-v0 | 500K | −200 | −200 | — |
| Hopper-v4 | 1M | 2192 ± 649 | 2214 ± 362 | 2255 ± 305 |
| Walker2d-v4 | 1M | 2925 ± 877 | 2933 ± 707 | 3458 ± 253 |
| HalfCheetah-v4 | 1M | 2077 ± 900 | 2181 ± 1094 | 1436 ± 68 |
| Breakout | 10M | 428.5 ± 20.9 | 416.4 ± 20.4 | 406.4 ± 12.9 |
| Pong | 10M | 20.6 ± 0.0 | 20.5 ± 0.2 | 20.5 ± 0.7 |
| BeamRider | 10M | 2046.5 ± 485 | 2232.9 ± 463 | 2851.6 ± 381 |

**All 9 match CleanRL.** Where I'm outside Baselines' range (Walker2d and BeamRider below, Breakout above), CleanRL differs from Baselines in the same way.

### The time-limit fix

MuJoCo episodes stop at 1,000 steps even when the agent is doing well. That's a **truncation**, not a failure. CleanRL and Baselines treat it as the end of the episode, so the future after the cut-off counts as 0.

**Why that hurts.** Say Hopper earns ~3 per step and γ = 0.99. A good state is then worth ~3 / (1 − 0.99) ≈ 300 of future reward. At the cut-off, the target says ~3. The critic can't see the clock, so the same good state gets two very different targets, and it learns a blurred value everywhere.

**The fix, `--truncation`**, does two things, and both are needed:
- **(A)** still mark the episode boundary, so GAE doesn't mix in the next episode's values;
- **(B)** add γ · V(final observation) to the last reward. The final observation is the state before the automatic reset.

| Environment | Fidelity mode | With the fix |
|---|---|---|
| Hopper-v4 | 2192 ± 649 | **2853 ± 367** |
| Walker2d-v4 | 2925 ± 877 | **3646 ± 279** |
| HalfCheetah-v4 | 2077 ± 900 | **2414 ± 964** |

- The fix is better on 8 of 9 same-seed pairs; the 9th was about equal.
- The critic's explained variance rises with it, e.g. HalfCheetah seed 1: 0.07 → 0.74.
- This is the known correct handling from Pardo et al., ["Time Limits in Reinforcement Learning"](https://arxiv.org/abs/1712.00378) (ICML 2018).

## Repo layout

| File | What it is |
|---|---|
| `ppo_classic.py` | PPO for discrete-action classic control (CartPole, Acrobot, MountainCar) |
| `ppo_atari.py` | PPO for Atari: Nature-CNN, frame stacking, uint8 observations. It uses the standard Atari wrappers from Stable-Baselines3 |
| `ppo_mujoco.py` | PPO for continuous control: Gaussian policy, observation and reward normalization |
| `bench/classic.sh`, `bench/mujoco.sh` | 3 envs × 3 seeds × 2 modes, on CPU |
| `bench/atari.sh`, `bench/modal_atari.py` | Atari runs, locally or on Modal (T4) |
| `bench/pull_reference.py` | Downloads CleanRL and Baselines learning curves (openrlbenchmark) into `bench/reference/` |
| `bench/compare.py` | Side-by-side table of my runs and the references |
| `render_mujoco.py` | Records a video of a trained MuJoCo policy |

## How to run

```bash
uv sync

# one run
uv run python ppo_mujoco.py --env-id Hopper-v4 --seed 1 --truncation      # with the fix
uv run python ppo_mujoco.py --env-id Hopper-v4 --seed 1 --no-truncation   # fidelity mode

# the benchmark
bash bench/classic.sh
bash bench/mujoco.sh
uvx modal run --detach bench/modal_atari.py::bench   # Atari on Modal: 3 games x 3 seeds
uvx modal volume get ppo-bench-runs runs/ .          # copy the results into ./runs/
uv run python bench/pull_reference.py       # reference curves
uv run python bench/compare.py              # the tables above
```

Runs write TensorBoard logs and an `episodes.csv` under `runs/`.

## Compute

- Classic control and MuJoCo ran on a laptop CPU, one thread per run (~1,000 steps/s on MuJoCo, ~17 min per 1M-step run).
- Atari ran on Modal: a T4 with 4 CPU cores, ~1,200–1,600 steps/s, ~2 h per 10M-step run. **The 9 Atari runs cost $14.92 in total.**

## Limits

- **3 seeds.** HalfCheetah's runs split into a weak and a strong gait in both CleanRL and here, so its std is large.
- **Environment versions differ slightly from the references:**
  - MuJoCo v4 here vs v2 for Baselines;
  - ALE v5 with `frameskip=1` and `repeat_action_probability=0` here, the same settings as the references' NoFrameskip-v4.
- **The fix was benchmarked on classic control and MuJoCo only.** In Atari, time limits almost never bind.
- **MountainCar is uninformative:** PPO never reaches the flag in 500K steps, mine or CleanRL's.
