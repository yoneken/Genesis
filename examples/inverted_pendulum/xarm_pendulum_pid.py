"""
Simple PD controller for the xArm inverted pendulum setup.

This script drives joint 2 proportionally to the pendulum tilt so that we can
quickly judge whether the current mechanics (joint speed, range, damping, etc.)
can physically recover from perturbations without training an RL policy.
"""

import argparse
import math
import time

import torch

import genesis as gs
from xarm_pendulum_env import XArmPendulumEnv
from xarm_pendulum_train import get_task_cfgs


def run_pd(args):
    env_cfg, obs_cfg, reward_cfg = get_task_cfgs()

    if args.disable_randomization:
        env_cfg = env_cfg.copy()
        env_cfg["pendulum_damping_range"] = None
        env_cfg["pendulum_mass_range"] = None
        env_cfg["pendulum_length_range"] = None

    gs.init(backend=getattr(gs, args.backend), logging_level=args.logging)
    env = XArmPendulumEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )

    target_angle = torch.full(
        (env.num_envs,),
        math.radians(args.target_angle_deg),
        dtype=gs.tc_float,
        device=gs.device,
    )
    max_steps = math.inf if args.max_time_s <= 0.0 else int(args.max_time_s / env.dt)
    max_joint_delta = None
    if args.max_joint_delta_deg is not None:
        max_joint_delta = math.radians(args.max_joint_delta_deg)

    print(
        f"Starting PD control with kp={args.kp:.2f}, kd={args.kd:.2f}, "
        f"target={args.target_angle_deg:.1f} deg, max_time={args.max_time_s}s, "
        f"max_retries={args.max_retries}"
    )

    attempt = 0
    while attempt <= args.max_retries:
        obs, _ = env.reset()
        step_count = 0
        sim_start = time.time()
        try:
            while step_count < max_steps:
                angle_error = target_angle - env.pendulum_joint_pos
                # Classic PD: derivative term opposes velocity.
                control = args.kp * angle_error - args.kd * env.pendulum_joint_vel
                if max_joint_delta is not None:
                    control = torch.clamp(control, -max_joint_delta, max_joint_delta)
                actions = torch.clamp(control / env.action_scale, -env.max_action, env.max_action).unsqueeze(-1)

                obs, reward, done, extras = env.step(actions)

                if args.log_interval > 0 and step_count % args.log_interval == 0:
                    avg_tip = env.tip_alignment.mean().item()
                    avg_angle = env.pendulum_joint_pos.mean().item()
                    avg_joint = env.dof_pos[:, env.action_dof_idx[0]].mean().item()
                    print(
                        f"[step {step_count:06d}] "
                        f"pendulum={avg_angle:+.4f} rad ({math.degrees(avg_angle):+.2f} deg), "
                        f"tip_align={avg_tip:+.3f}, joint2={avg_joint:+.3f} rad, "
                        f"reward={reward.mean().item():+.3f}"
                    )

                if torch.any(done):
                    timeout_mask = extras.get("time_outs")
                    timeouts = int((timeout_mask > 0.5).sum().item()) if timeout_mask is not None else 0
                    total_done = int(done.sum().item())
                    falls = total_done - timeouts
                    reason_parts = []
                    if timeouts:
                        reason_parts.append(f"timeout {timeouts}")
                    if falls:
                        reason_parts.append(f"fell {falls}")
                    reason = "; ".join(reason_parts) or "unknown"
                    print(f"Termination detected at step {step_count}: {reason}. Exiting.")
                    break

                step_count += 1
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down PD test.")
            break
        finally:
            elapsed = time.time() - sim_start
            print(
                f"Simulated {step_count} steps in {elapsed:.2f} s "
                f"(avg {step_count * env.dt:.2f} s of simulation time) on attempt {attempt + 1}."
            )

        # Stop on termination or max steps; only retry if explicitly requested and not terminated.
        if torch.any(done):
            break
        if step_count >= max_steps:
            print("Reached max steps; stopping without automatic rerun.")
            break
        attempt += 1
        if attempt > args.max_retries:
            print("Max retries exhausted; stopping.")
            break
        print(f"Retrying PD control (attempt {attempt + 1}/{args.max_retries + 1}) after timeout.")


def parse_args():
    parser = argparse.ArgumentParser(description="PD controller experiment for the xArm inverted pendulum.")
    parser.add_argument("--kp", type=float, default=6.0, help="P gain applied to pendulum angle error (rad).")
    parser.add_argument("--kd", type=float, default=0.5, help="D gain applied to pendulum angular velocity.")
    parser.add_argument("--target_angle_deg", type=float, default=0.0, help="Pendulum target angle in degrees.")
    parser.add_argument("--max_joint_delta_deg", type=float, default=30.0, help="Clamp per-step joint delta (deg).")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--max_time_s", type=float, default=0.0, help="Maximum wall-clock simulation time in seconds.")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"], help="Genesis backend to use.")
    parser.add_argument("--logging", type=str, default="warning", help="Genesis logging level.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer.")
    parser.add_argument("--log_interval", type=int, default=200, help="Print status every N steps.")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="Number of additional runs to attempt after a failure or timeout.",
    )
    parser.add_argument(
        "--disable_randomization",
        action="store_true",
        help="Disable pendulum parameter randomization for deterministic runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pd(parse_args())
