import argparse
import pickle
import re
from pathlib import Path
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from xarm_pendulum_env import XArmPendulumEnv


def load_policy(env, train_cfg, log_dir: Path, checkpoint_name: str | None):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    cpu_only = not torch.cuda.is_available()
    if checkpoint_name:
        checkpoint_path = log_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
    else:
        checkpoints = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
        _, checkpoint_path = max(((int(re.findall(r"\d+", f.stem)[0]), f) for f in checkpoints), key=lambda tup: tup[0])
    if cpu_only:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        state_dict = (
            checkpoint.get("actor_critic_state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("module_state_dict")
        )
        if state_dict is None:
            raise KeyError("Policy weights not found in checkpoint for CPU restore.")
        runner.alg.actor_critic.load_state_dict(state_dict)
    else:
        runner.load(checkpoint_path)
    print(f"Loaded checkpoint {checkpoint_path}")
    return runner.get_inference_policy(device=gs.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="xarm-pendulum")
    parser.add_argument("-B", "--num_envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=32, help="Total number of evaluation episodes to collect")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint filename to load")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer during evaluation")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")

    log_dir = Path("logs") / args.exp_name
    cfg_path = log_dir / "cfgs.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find configuration file at {cfg_path}")

    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    env_cfg["num_envs"] = args.num_envs

    env = XArmPendulumEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )

    policy = load_policy(env, train_cfg, log_dir, args.checkpoint)
    if hasattr(policy, "eval"):
        policy.eval()

    obs, _ = env.reset()
    episode_returns = torch.zeros((args.num_envs,), device=gs.device)
    completed_returns = []
    max_steps = env.max_episode_length

    with torch.no_grad():
        while len(completed_returns) < args.episodes:
            for _ in range(max_steps):
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                episode_returns += rewards

                done_idx = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                if done_idx.numel() > 0:
                    completed_returns.extend(episode_returns[done_idx].tolist())
                    episode_returns[done_idx] = 0.0

                if len(completed_returns) >= args.episodes:
                    break

    completed_returns = completed_returns[: args.episodes]
    if completed_returns:
        avg_return = sum(completed_returns) / len(completed_returns)
        print(f"Evaluated {len(completed_returns)} episodes. Average return: {avg_return:.3f}")
        print(f"Returns: {completed_returns}")
    else:
        print("No completed episodes recorded during evaluation.")


if __name__ == "__main__":
    main()
