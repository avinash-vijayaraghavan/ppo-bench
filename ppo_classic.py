"""
PPO for classic-control (discrete actions), as a single self-contained file, CleanRL-style.
Envs: CartPole-v1, Acrobot-v1, MountainCar-v0 (gymnasium 1.3 vector API).
"""

import argparse
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    # Experiment
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).removesuffix(".py"),
        help="the name of this experiment (goes into the run name)")
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the gymnasium environment id")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
        help="total env steps across all envs; the benchmark budget (num_iterations is derived from it)")
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="set torch.backends.cudnn.deterministic")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True,
        help="use CUDA if available")

    # Logging
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=False,
        help="also mirror the TensorBoard metrics to wandb")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-bench",
        help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="wandb entity (team); None = your default")

    # Rollout
    parser.add_argument("--num-envs", type=int, default=4,
        help="number of parallel envs")
    parser.add_argument("--num-steps-per-rollout", type=int, default=128,
        help="steps per env per rollout (T)")

    # Optimization
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="Adam learning rate")
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True,
        help="linearly anneal the learning rate to 0 over training")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="minibatches per epoch")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="passes over each rollout's batch (K)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="global gradient-norm clip")
    parser.add_argument("--target-kl", type=float, default=None,
        help="early-stop the epochs of an update when approx_kl exceeds this; default off")

    # RL / PPO
    parser.add_argument("--gamma", type=float, default=0.99,
        help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE lambda")
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True,
        help="normalize advantages per minibatch (policy loss only)")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="surrogate clipping epsilon")
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True,
        help="clipped value loss (CleanRL default, always on in baselines ppo2); removed late instability on CartPole")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="entropy bonus coefficient (c2)")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="value loss coefficient (c1)")
    parser.add_argument("--truncation", action=argparse.BooleanOptionalAction, default=True,
        help="at time limits, bootstrap with gamma * V(final_obs) and mark the episode boundary (A+B); --no-truncation = ignore it")
    # fmt: on
    args = parser.parse_args()

    # Derived (not flags)
    args.batch_size = args.num_envs * args.num_steps_per_rollout
    assert (
        args.batch_size % args.num_minibatches == 0
    ), "batch_size must be divisible by num_minibatches"
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = (
        args.total_timesteps // args.batch_size
    )  # rollout + learning cycles (pi_old refreshes)
    args.num_gradient_steps = (
        args.num_iterations * args.update_epochs * args.num_minibatches
    )  # optimizer steps (before target_kl early stops)
    return args


# ---------------------------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------------------------
def make_env(env_id, num_envs, seed):
    """
    gymnasium vector env with SAME_STEP autoreset (numpy in, numpy out; move to torch in the loop).

    step(actions) -> obs, reward, terminated, truncated, info    (arrays of shape (num_envs, ...))
      - autoreset in the SAME step: for envs that finished, obs is already the reset obs
      - info["final_obs"]: object array (num_envs,), the pre-reset obs where info["_final_obs"] is True, else None
            e.g. np.stack(info["final_obs"][mask]) for a boolean mask of truncated envs
      - info["final_info"]["episode"]: {"r", "l", "t"} arrays (num_envs,), valid where
            info["final_info"]["_episode"] is True (RecordEpisodeStatistics: raw return / length per episode)
    """
    envs = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        wrappers=[gym.wrappers.RecordEpisodeStatistics],
    )
    envs.action_space.seed(seed)
    return envs  # first call: envs.reset(seed=seed)


# ---------------------------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------------------------
# Architecture: SEPARATE actor and critic MLPs, each 64-64 tanh (CleanRL; baselines value_network='copy').
#   layer_init: orthogonal weights, std sqrt(2) for hidden layers, bias 0; actor output std 0.01, critic output std 1.0
#   get_value(x) -> (B,), get_action_and_value(x, action=None) -> action, logprob, entropy, value (B,)
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic networks (CleanRL / baselines value_network='copy'); no shared trunk.
    A shared trunk learned much slower on CartPole: the large value loss dominated the shared features (see notes.txt).
    """

    def __init__(self, actor, critic) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)

    def get_value(self, obs):
        return self.critic(obs).flatten()

    def get_action_and_value(self, obs, action=None):
        # action: (num_envs,) indices
        action_logits, value = self(obs)
        action_dist = Categorical(logits=action_logits)
        if action is None:
            action = action_dist.sample()
        logprob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action, logprob, entropy, value.flatten()


def get_actor_and_critic(obs_shape: int, num_actions: int):
    def mlp(out_features, out_std):
        return nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, out_features), std=out_std),
        )

    actor = mlp(num_actions, 0.01)  # near-uniform initial policy
    critic = mlp(1, 1.0)
    return ActorCritic(actor, critic)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,  # wandb picks up everything written to the SummaryWriter below
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    envs = make_env(env_id=args.env_id, num_envs=args.num_envs, seed=args.seed)
    state, _ = envs.reset(seed=args.seed)  # seeds every env (env i gets seed + i)
    state = torch.as_tensor(state, dtype=torch.float32, device=device)
    done = torch.zeros(args.num_envs, device=device)
    obs_shape = int(np.prod(envs.single_observation_space.shape))
    num_actions = envs.single_action_space.n
    agent = get_actor_and_critic(obs_shape=obs_shape, num_actions=num_actions).to(
        device
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs = torch.zeros(
        (args.num_steps_per_rollout, args.num_envs)
        + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        args.num_steps_per_rollout, args.num_envs, device=device, dtype=torch.int64
    )
    logprobs = torch.zeros(args.num_steps_per_rollout, args.num_envs, device=device)
    rewards = torch.zeros(args.num_steps_per_rollout, args.num_envs, device=device)
    dones = torch.zeros(args.num_steps_per_rollout, args.num_envs, device=device)
    values = torch.zeros(args.num_steps_per_rollout, args.num_envs, device=device)

    global_step = 0  # env steps so far, summed over envs (the x-axis for logging)
    start_time = time.time()

    episode_log = open(f"runs/{run_name}/episodes.csv", "w")
    episode_log.write("global_step,episodic_return,episodic_length\n")
    recent_returns = deque(maxlen=100)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            lrnow = (
                1.0 - 1.0 / args.num_iterations * (iteration - 1)
            ) * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps_per_rollout):
            global_step += args.num_envs

            # dont reset after every rollout -> this will prevent learning states beyond the rollout steps
            obs[step] = state
            dones[step] = done

            with torch.no_grad():
                # we are using the old (frozen) policy
                action, logprob, _, value = agent.get_action_and_value(state, None)

            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            # step the env (works)
            state, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            state = torch.as_tensor(state, dtype=torch.float32).to(device)
            reward = torch.as_tensor(reward, dtype=torch.float32).to(device)
            boundary = terminated
            truncated_not_terminated = truncated & ~terminated
            if args.truncation and truncated_not_terminated.any():
                final_obs = torch.as_tensor(
                    np.stack(info["final_obs"][truncated_not_terminated]),
                    dtype=obs.dtype,
                ).to(device)
                with torch.no_grad():
                    reward[truncated_not_terminated] += args.gamma * agent.get_value(
                        final_obs
                    )
                boundary = (
                    terminated | truncated
                )  # treat truncation also as a episodic boundary

            done = torch.as_tensor(boundary, dtype=dones.dtype).to(device)
            rewards[step] = reward

            if "episode" in info.get("final_info", {}):
                episode = info["final_info"]["episode"]
                for i in np.flatnonzero(info["final_info"]["_episode"]):
                    ep_return, ep_length = float(episode["r"][i]), int(episode["l"][i])
                    recent_returns.append(ep_return)
                    episode_log.write(f"{global_step},{ep_return},{ep_length}\n")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(state)  # for bootstrapping
            advantages = torch.zeros_like(rewards)

            for t in reversed(range(args.num_steps_per_rollout)):
                if t == args.num_steps_per_rollout - 1:
                    nextterminal = done
                    nextvalues = next_value
                else:
                    nextterminal = dones[t + 1]
                    nextvalues = values[t + 1]
                delta_t = (
                    rewards[t]
                    + args.gamma * (1 - nextterminal) * nextvalues
                    - values[t]
                )
                if t == args.num_steps_per_rollout - 1:
                    advantages[t] = delta_t
                else:
                    advantages[t] = (
                        delta_t
                        + args.gamma
                        * args.gae_lambda
                        * (1 - nextterminal)
                        * advantages[t + 1]
                    )
            returns = advantages + values

        # 4. Update using shuffled minibatches
        b_obs = obs.view(args.batch_size, *envs.single_observation_space.shape)
        b_actions = actions.view(args.batch_size)
        b_logprobs = logprobs.view(args.batch_size)
        b_advantages = advantages.view(args.batch_size)
        b_returns = returns.view(args.batch_size)
        b_values = values.view(args.batch_size)
        b_inds = np.arange(args.batch_size)
        clipfracs = []  # fraction of samples whose ratio was clipped (diagnostic)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # a new minibatch split every epoch
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs=b_obs[mb_inds], action=b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(
                    logratio
                )  # pi_theta / pi_old; exactly 1 on the first minibatch of the first epoch

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1) - logratio
                    ).mean()  # k3 estimator of KL(pi_old || pi_theta), always >= 0
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                # (d) advantages for the policy loss: normalized per minibatch if args.norm_adv (+1e-8 in the std)
                if args.norm_adv:
                    nb_advantages = (
                        b_advantages[mb_inds] - b_advantages[mb_inds].mean()
                    ) / (b_advantages[mb_inds].std() + 1e-8)
                else:
                    nb_advantages = b_advantages[mb_inds]
                clipped_ratio = torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = -torch.min(
                    ratio * nb_advantages, clipped_ratio * nb_advantages
                ).mean()

                mb_returns, mb_values = b_returns[mb_inds], b_values[mb_inds]
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                if args.clip_vloss:
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values, -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # (h) gradient step with global grad-norm clipping
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))
        metrics = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/SPS": sps,
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)),
            "losses/explained_variance": explained_var,
        }
        for tag, value in metrics.items():
            writer.add_scalar(tag, value, global_step)
        if iteration % 20 == 0 or iteration == args.num_iterations:
            mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
            print(
                f"iter {iteration}/{args.num_iterations}  step {global_step}  "
                f"return(last {len(recent_returns)}) {mean_ret:.1f}  "
                f"v_loss {metrics['losses/value_loss']:.3f}  kl {metrics['losses/approx_kl']:.4f}  "
                f"clipfrac {metrics['losses/clipfrac']:.3f}  ev {explained_var:.2f}  SPS {sps}"
            )

    envs.close()
    episode_log.close()
    writer.close()
    if args.track:
        wandb.finish()
