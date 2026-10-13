"""
Run ppo_atari.py on Modal: one container per run (T4 + 4 CPU cores for the 8 async envs).

    uvx modal run --detach bench/modal_atari.py::smoke    # ~5 min of Pong (SPS check)
    uvx modal run --detach bench/modal_atari.py::bench    # 3 games x 3 seeds, terminate mode, 10M steps
    uvx modal volume get ppo-bench-runs runs/ .           # copy the results into ./runs/

Runs write straight into the `ppo-bench-runs` Volume (cwd=/vol), so partial results survive a timeout or a
dropped local client. Cost: T4 $0.59/h + 4 cores $0.19/h + 8 GiB $0.06/h = ~$0.84/h per run (Modal bills the
requested resources); a 10M run takes ~2 h at ~1,430 SPS. timeout=2.5 h caps a run at ~$2.1.
"""

import os
import subprocess

import modal

ROOT = os.path.join(os.path.dirname(__file__), "..")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(  # same pins as uv.lock; opencv headless (no libGL) for ResizeObservation
        "torch==2.14.0",
        "gymnasium[atari]==1.3.0",
        "ale-py==0.12.1",
        "stable-baselines3==2.9.0",
        "numpy==2.4.6",
        "tensorboard",
        "opencv-python-headless",
    )
    .add_local_file(os.path.join(ROOT, "ppo_atari.py"), "/root/ppo_atari.py")
)
app = modal.App("ppo-bench-atari", image=image)
volume = modal.Volume.from_name("ppo-bench-runs", create_if_missing=True)


@app.function(  # max_containers: Starter plan allows 10 GPUs; Modal spun up idle spares without a cap
    gpu="T4", cpu=4, memory=8192, timeout=int(2.5 * 3600), volumes={"/vol": volume}, max_containers=9
)
def train(args, seconds=None):
    """Run ppo_atari.py with `args` (optionally killed after `seconds`); outputs land in /vol/runs/<run_name>/."""
    cmd = ["python", "-u", "/root/ppo_atari.py", *args]
    if seconds:
        cmd = ["timeout", str(seconds), *cmd]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd="/vol")  # stdout streams to `modal run` / the app logs
    volume.commit()


@app.local_entrypoint()
def smoke():
    train.remote(["--env-id", "ALE/Pong-v5", "--async-envs", "--no-truncation", "--exp-name", "smoke_modal"], 300)


@app.local_entrypoint()
def bench():
    calls = [
        train.spawn(
            ["--env-id", env, "--seed", str(seed), "--async-envs", "--no-truncation", "--exp-name", "bench_terminate"]
        )
        for env in ["ALE/Breakout-v5", "ALE/Pong-v5", "ALE/BeamRider-v5"]
        for seed in [1, 2, 3]
    ]
    for c in calls:
        c.get()
