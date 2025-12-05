import argparse
import math
import pickle
from importlib import metadata
from pathlib import Path

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


def get_train_cfg(exp_name: str, max_iterations: int):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 0.8,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 32,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }


def get_task_cfgs():
    env_cfg = {
        "num_actions": 6,
        "mjcf_file": "xml/ufactory_xarm6_pendulum/xarm6_pendulum.xml",
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "pendulum_joint_name": "pendulum_hinge",
        "pendulum_link_name": "pendulum",
        "default_joint_pos": [0.0, 0.2, -0.2, 0.0, -1.57, 0.0],
        "default_pendulum_angle": 0.0,
        "ctrl_dt": 0.03,
        "episode_length_s": 20.0,
        "action_scale": 0.6,
        "max_action": 1.0,
        "kp": 100.0,
        "kd": 10.0,
        "init_joint_noise": 0.02,
        "init_pendulum_noise": 0.1,
        "termination_cos_threshold": 0.3,
        "pendulum_axis_local": (1.0, 0.0, 0.0),
        "pendulum_hinge_axis_local": (0.0, 0.0, 1.0),
        "gravity_axis": (0.0, 0.0, 1.0),
        "sim_substeps": 2,
        "max_joint_velocity": math.radians(180.0),
    }
    obs_cfg = {
        "num_obs": 24,
        "obs_scales": {
            "joint_pos": 2.0,
            "joint_vel": 0.05,
            "pendulum_vel": 0.2,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "upright": 6.0,
            "tilt": -3.0,
            "joint_vel": -0.01,
            "action_rate": -0.01,
            "pendulum_velocity": -0.5,
        },
    }
    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="xarm-pendulum")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")

    env_cfg, obs_cfg, reward_cfg = get_task_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    log_dir = Path("logs") / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, train_cfg], f)

    env = XArmPendulumEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
To start training the inverted pendulum policy run:
python examples/inverted_pendulum/xarm_pendulum_train.py --max_iterations 500 --num_envs 2048
"""
