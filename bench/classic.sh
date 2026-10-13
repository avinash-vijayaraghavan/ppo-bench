#!/usr/bin/env bash
# Classic-control benchmark: 3 envs x 3 seeds x 2 truncation modes, 500k steps each (script defaults), CPU.
#   terminate = time limit treated as termination (CleanRL / baselines), bootstrap = our A+B.
# One wave of 9 runs per mode, 1 CPU thread each.   Usage: bash bench/classic.sh
cd "$(dirname "$0")/.." && mkdir -p runs/logs

for mode in terminate bootstrap; do
  flag=$([ $mode = bootstrap ] && echo --truncation || echo --no-truncation)
  for env in CartPole-v1 Acrobot-v1 MountainCar-v0; do
    for seed in 1 2 3; do
      OMP_NUM_THREADS=1 uv run python -u ppo_classic.py --env-id $env --seed $seed $flag --no-cuda \
        --exp-name bench_$mode > runs/logs/${env}__bench_${mode}__${seed}.log 2>&1 &
    done
  done
  wait
done
