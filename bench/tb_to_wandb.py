"""
Upload finished runs' TensorBoard scalars to wandb (for runs trained without --track, e.g. on Modal).

    uv run python bench/tb_to_wandb.py runs/ALE-Pong-v5__bench_terminate__1__*  [--project ppo-bench]

One wandb run per folder, named after the folder; every scalar is logged against global_step, and the
hyperparameters text table becomes the run config.
"""

import argparse
import os
from collections import defaultdict

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_hparams(ea):
    try:
        text = ea.Tensors("hyperparameters/text_summary")[0].tensor_proto.string_val[0].decode()
    except (KeyError, IndexError):
        return {}
    rows = [line.strip("|").split("|") for line in text.splitlines()[2:]]  # skip header + separator
    return {k: v for k, v in rows if k}


def upload(run_dir, project, entity):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0, "tensors": 0})
    ea.Reload()
    by_step = defaultdict(dict)
    for tag in ea.Tags()["scalars"]:
        for e in ea.Scalars(tag):
            by_step[e.step][tag] = e.value
    name = os.path.basename(run_dir.rstrip("/"))
    run = wandb.init(project=project, entity=entity, name=name, config=parse_hparams(ea), reinit="finish_previous")
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    for step in sorted(by_step):
        run.log({"global_step": step, **by_step[step]})
    run.finish()
    print(f"{name}: {len(by_step)} steps, {len(ea.Tags()['scalars'])} scalars -> {run.url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--project", default="ppo-bench")
    parser.add_argument("--entity", default=None)
    args = parser.parse_args()
    for d in args.run_dirs:
        upload(d, args.project, args.entity)


if __name__ == "__main__":
    main()
