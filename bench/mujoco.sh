#!/usr/bin/env bash
# MuJoCo benchmark: 3 envs x 3 seeds x 2 truncation modes, 1M steps each (script defaults).
#   terminate = time limit treated as termination (CleanRL / baselines), bootstrap = our A+B.
# One wave of 9 runs per mode, 1 CPU thread each.   Usage: bash bench/mujoco.sh
cd "$(dirname "$0")/.." && mkdir -p runs/logs

for mode in terminate bootstrap; do
  flag=$([ $mode = bootstrap ] && echo --truncation || echo --no-truncation)
  for env in Hopper-v4 Walker2d-v4 HalfCheetah-v4; do
    for seed in 1 2 3; do
      OMP_NUM_THREADS=1 uv run python -u ppo_mujoco.py --env-id $env --seed $seed $flag \
        --exp-name bench_$mode > runs/logs/${env}__bench_${mode}__${seed}.log 2>&1 &
    done
  done
  wait
done
