#!/usr/bin/env bash
# Atari benchmark: 3 games x 3 seeds, terminate mode only (= CleanRL / baselines), 10M steps each (script defaults).
# ~3 h per run on the laptop, so meant for a cloud box. One wave of 3 seeds per game.   Usage: bash bench/atari.sh
cd "$(dirname "$0")/.." && mkdir -p runs/logs

for env in ALE/Breakout-v5 ALE/Pong-v5 ALE/BeamRider-v5; do
  for seed in 1 2 3; do
    uv run python -u ppo_atari.py --env-id $env --seed $seed --no-truncation --async-envs \
      --exp-name bench_terminate > runs/logs/${env/\//-}__bench_terminate__${seed}.log 2>&1 &
  done
  wait
done
