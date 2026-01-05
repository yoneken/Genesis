"""
Simple PID-style velocity controller for the xArm inverted pendulum setup.

Updates:
- Adds optional integral term with clamping anti-windup.
- Adds acceleration and velocity saturation on the velocity command.
- Adds optional gain sweep to iterate Kp/Kd/Ki combinations.
"""

import argparse
import math
import time
import csv
from pathlib import Path

import torch

import genesis as gs
from xarm_pendulum_env import XArmPendulumEnv
from xarm_pendulum_train import get_task_cfgs


def _parse_list(arg_val):
    if arg_val is None:
        return None
    if isinstance(arg_val, (int, float)):
        return [float(arg_val)]
    s = str(arg_val).strip()
    if not s:
        return None
    return [float(x) for x in s.split(",")]


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
    max_joint_vel = None
    if args.max_joint_vel_deg_s is not None:
        max_joint_vel = math.radians(args.max_joint_vel_deg_s)
    max_joint_accel = None
    if args.max_joint_accel_deg_s2 is not None:
        max_joint_accel = math.radians(args.max_joint_accel_deg_s2)

    print(
        f"Starting control: kp={args.kp:.2f}, kd={args.kd:.2f}, ki={args.ki:.2f}, "
        f"target={args.target_angle_deg:.1f} deg, max_time={args.max_time_s}s, "
        f"vel_limit={args.max_joint_vel_deg_s} deg/s, accel_limit={args.max_joint_accel_deg_s2} deg/s^2, "
        f"success={args.success_duration_s}s @ cos>{args.success_cos_threshold}"
    )

    # Prepare CSV
    csv_path = Path(args.csv_path) if args.csv_path else Path("logs/pid_sweep.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "kp", "kd", "ki", "steps", "sim_time_s", "success",
                "avg_tip_align", "min_tip_align", "max_tip_align",
                "avg_pendulum_angle_rad", "min_pendulum_angle_rad", "max_pendulum_angle_rad"
            ])

    def run_once(kp, kd, ki):
        obs, _ = env.reset()
        step_count = 0
        sim_start = time.time()
        # Integrator and previous velocity command per env
        i_err = torch.zeros((env.num_envs,), dtype=gs.tc_float, device=gs.device)
        v_prev = torch.zeros((env.num_envs,), dtype=gs.tc_float, device=gs.device)
        done = torch.zeros((env.num_envs,), dtype=torch.bool, device=gs.device)
        upright_time = torch.zeros((env.num_envs,), dtype=gs.tc_float, device=gs.device)
        # Metrics accumulators
        tip_sum = 0.0
        angle_sum = 0.0
        min_tip = 1.0
        max_tip = -1.0
        min_angle = 1e9
        max_angle = -1e9
        try:
            while step_count < max_steps:
                angle_error = target_angle - env.pendulum_joint_pos
                # Update integrator with clamping anti-windup
                if ki > 0.0:
                    v_unsat_preview = kp * angle_error - kd * env.pendulum_joint_vel + ki * i_err
                    if max_joint_vel is not None:
                        will_sat = torch.abs(v_unsat_preview) > max_joint_vel
                        # sign of integrator contribution
                        integ_contrib = ki * angle_error
                        same_dir = torch.sign(integ_contrib) == torch.sign(v_unsat_preview)
                        # integrate only if not saturating in the same direction
                        integrate_mask = ~will_sat | ~same_dir
                    else:
                        integrate_mask = torch.ones_like(i_err, dtype=torch.bool)
                    i_err = i_err + (angle_error * env.dt) * integrate_mask.to(gs.tc_float)

                # Compute unsaturated desired velocity (rad/s)
                v_unsat = kp * angle_error - kd * env.pendulum_joint_vel + (ki * i_err if ki > 0.0 else 0.0)

                # Acceleration (jerk) limiting
                if max_joint_accel is not None:
                    dv = v_unsat - v_prev
                    dv = torch.clamp(dv, -max_joint_accel * env.dt, max_joint_accel * env.dt)
                    v_cmd = v_prev + dv
                else:
                    v_cmd = v_unsat

                # Velocity saturation
                if max_joint_vel is not None:
                    v_cmd = torch.clamp(v_cmd, -max_joint_vel, max_joint_vel)

                actions = torch.clamp(v_cmd / env.action_scale, -env.max_action, env.max_action).unsqueeze(-1)
                v_prev = v_cmd

                obs, reward, done, extras = env.step(actions)

                # Success-duration tracking: cosine alignment must stay above threshold
                upright_mask = env.tip_alignment >= args.success_cos_threshold
                # Reset timer where not upright
                upright_time = upright_time + (upright_mask.to(gs.tc_float) * env.dt)
                upright_time = upright_time * upright_mask.to(gs.tc_float)
                if torch.any(upright_time >= args.success_duration_s):
                    print("Success condition reached: upright for target duration.")
                    break

                # Metrics
                avg_tip_batch = env.tip_alignment.mean().item()
                avg_angle_batch = env.pendulum_joint_pos.mean().item()
                tip_sum += avg_tip_batch
                angle_sum += avg_angle_batch
                min_tip = min(min_tip, env.tip_alignment.min().item())
                max_tip = max(max_tip, env.tip_alignment.max().item())
                min_angle = min(min_angle, env.pendulum_joint_pos.min().item())
                max_angle = max(max_angle, env.pendulum_joint_pos.max().item())

                if args.log_interval > 0 and step_count % args.log_interval == 0:
                    tip0 = env.tip_alignment[0].item()
                    angle0 = env.pendulum_joint_pos[0].item()
                    j2_vel0 = env.dof_vel[0, env.action_dof_idx[0]].item()
                    j2_pos0 = env.dof_pos[0, env.action_dof_idx[0]].item()
                    print(
                        f"[step {step_count:06d}] "
                        f"pendulum={angle0:+.4f} rad ({math.degrees(angle0):+.2f} deg), "
                        f"tip_align={tip0:+.3f}, joint2_vel={j2_vel0:+.3f} rad/s, "
                        f"joint2_pos={j2_pos0:+.3f} rad, "
                        f"kp={kp:.2f} kd={kd:.2f} ki={ki:.2f}"
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
                    print(f"Termination at step {step_count}: {reason}.")
                    break

                step_count += 1
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down test.")
        finally:
            elapsed = time.time() - sim_start
            print(
                f"Simulated {step_count} steps in {elapsed:.2f} s "
                f"(avg {step_count * env.dt:.2f} s of simulation time)."
            )
        # success achieved if any env kept upright for the full duration
        success = bool(torch.any(upright_time >= args.success_duration_s).item())
        # Write CSV row
        avg_tip_align = tip_sum / max(step_count, 1)
        avg_angle = angle_sum / max(step_count, 1)
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{kp:.6f}", f"{kd:.6f}", f"{ki:.6f}", step_count, f"{step_count * env.dt:.3f}", int(success),
                f"{avg_tip_align:.6f}", f"{min_tip:.6f}", f"{max_tip:.6f}",
                f"{avg_angle:.6f}", f"{min_angle:.6f}", f"{max_angle:.6f}"
            ])
        print(f"Logged metrics to {csv_path}")
        return step_count, done, success

    kp_list = _parse_list(args.kp_list) or [args.kp]
    kd_list = _parse_list(args.kd_list) or [args.kd]
    ki_list = _parse_list(args.ki_list) or [args.ki]

    best = None
    for kp in kp_list:
        for kd in kd_list:
            for ki in ki_list:
                print(f"\n=== Running with kp={kp:.2f}, kd={kd:.2f}, ki={ki:.2f} ===")
                steps, done, success = run_once(kp, kd, ki)
                if best is None or steps > best[0]:
                    best = (steps, kp, kd, ki)
                if success:
                    print("Gain set reached success duration; stopping sweep.")
                    return
    if best is not None:
        print(
            f"Finished sweep. Best steps={best[0]} with kp={best[1]:.2f}, kd={best[2]:.2f}, ki={best[3]:.2f}."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="PD controller experiment for the xArm inverted pendulum.")
    parser.add_argument("--kp", type=float, default=8.0, help="P gain applied to pendulum angle error (rad).")
    parser.add_argument("--kd", type=float, default=0.8, help="D gain applied to pendulum angular velocity.")
    parser.add_argument("--ki", type=float, default=0.0, help="I gain for pendulum angle error (rad/s). 0 to disable.")
    parser.add_argument("--target_angle_deg", type=float, default=0.0, help="Pendulum target angle in degrees.")
    parser.add_argument("--max_joint_vel_deg_s", type=float, default=180.0, help="Clamp desired joint velocity (deg/s).")
    parser.add_argument(
        "--max_joint_accel_deg_s2",
        type=float,
        default=500.0,
        help="Clamp change of desired joint velocity per second (deg/s^2).",
    )
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--max_time_s", type=float, default=0.0, help="Maximum wall-clock simulation time in seconds.")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"], help="Genesis backend to use.")
    parser.add_argument("--logging", type=str, default="warning", help="Genesis logging level.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer.")
    parser.add_argument("--log_interval", type=int, default=5, help="Print status every N steps.")
    parser.add_argument(
        "--success_duration_s",
        type=float,
        default=10.0,
        help="Auto-stop when pendulum stays upright for this continuous duration.",
    )
    parser.add_argument(
        "--success_cos_threshold",
        type=float,
        default=0.985,
        help="Cosine of tip alignment threshold to consider 'upright' (in [0,1]).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV file to append sweep results (default logs/pid_sweep.csv).",
    )
    # Optional gain lists for sweep (comma-separated floats)
    parser.add_argument("--kp_list", type=str, default=None, help="Comma-separated list of Kp values to sweep.")
    parser.add_argument("--kd_list", type=str, default=None, help="Comma-separated list of Kd values to sweep.")
    parser.add_argument("--ki_list", type=str, default=None, help="Comma-separated list of Ki values to sweep.")
    parser.add_argument(
        "--disable_randomization",
        action="store_true",
        help="Disable pendulum parameter randomization for deterministic runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pd(parse_args())
