"""
Render a trained ppo_mujoco.py agent to an MP4.

    MUJOCO_GL=osmesa uv run python render_mujoco.py runs/<run_name>/agent.pt [--out video.mp4] [--seed 0]

Loads the saved weights + observation-normalization stats, runs one episode with the policy's MEAN action
(no exploration noise), and writes the frames to an MP4. Prints the episode's raw return.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless rendering (software OpenGL)

import gymnasium as gym
import imageio
import numpy as np
import torch

from ppo_mujoco import get_actor_and_critic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=None, help="default: <checkpoint dir>/episode.mp4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    env_id = ckpt["args"]["env_id"]
    env = gym.make(env_id, render_mode="rgb_array", width=640, height=480)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    agent = get_actor_and_critic(obs_dim=obs_dim, act_dim=act_dim)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    def normalize(obs):  # same as NormalizeObservation + clip(-10, 10) during training, with frozen stats
        return np.clip((obs - ckpt["obs_mean"]) / np.sqrt(ckpt["obs_var"] + 1e-8), -10, 10)

    obs, _ = env.reset(seed=args.seed)
    frames, total, steps, done = [env.render()], 0.0, 0, False
    while not done:
        with torch.no_grad():
            mean_action = agent.actor(torch.as_tensor(normalize(obs), dtype=torch.float32)[None])[0][0].numpy()
        obs, reward, terminated, truncated, _ = env.step(np.clip(mean_action, env.action_space.low, env.action_space.high))
        total += float(reward)
        steps += 1
        done = terminated or truncated
        frames.append(env.render())

    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "episode.mp4")
    fps = env.metadata.get("render_fps", 30)
    imageio.mimsave(out, frames, fps=fps)
    print(f"{env_id}: return {total:.0f} over {steps} steps -> {out} ({len(frames)} frames, {fps} fps)")


if __name__ == "__main__":
    main()
