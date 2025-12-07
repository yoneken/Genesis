"""
Keyboard teleop for the xArm inverted pendulum setup.

Controls
--------
Arrow Left/Right : decrease / increase joint 1
Arrow Up/Down    : increase / decrease joint 2
r                : reset to default joint configuration
q                : quit
"""

import argparse
import math
import threading
import time

import numpy as np
from pynput import keyboard

import genesis as gs
from xarm_pendulum_train import get_task_cfgs


class KeyboardDevice:
    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def start(self):
        self._listener.start()

    def stop(self):
        try:
            self._listener.stop()
        finally:
            self._listener.join()

    def snapshot(self):
        with self._lock:
            return set(self._pressed)

    def _on_press(self, key):
        with self._lock:
            self._pressed.add(key)

    def _on_release(self, key):
        with self._lock:
            self._pressed.discard(key)


def apply_dependent_rules(joint_vec: np.ndarray):
    joint_vec[2] = -2.0 * joint_vec[1]
    joint_vec[3] = joint_vec[0] + math.pi / 2.0
    joint_vec[4] = - math.pi / 2.0
    joint_vec[5] = - joint_vec[1]
    return joint_vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_deg", type=float, default=2.0, help="Increment per key press in degrees")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    env_cfg, _, _ = get_task_cfgs()
    gs.init(backend=getattr(gs, args.backend), logging_level="warning")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -1.7, 1.2),
            camera_lookat=(0.3, 0.0, 0.4),
            camera_fov=50,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=env_cfg["ctrl_dt"]),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=env_cfg["mjcf_file"]))
    scene.build()

    joint_names = env_cfg["joint_names"]
    motors_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joint_names]
    dof_lower, dof_upper = robot.get_dofs_limit(motors_idx)
    joint_limit_overrides = env_cfg.get("joint_limit_overrides", {})
    if joint_limit_overrides:
        for name, (lower, upper) in joint_limit_overrides.items():
            idx = robot.get_joint(name).dofs_idx_local[0]
            dof_lower[idx] = lower
            dof_upper[idx] = upper
    dof_lower_np = dof_lower.cpu().numpy()
    dof_upper_np = dof_upper.cpu().numpy()

    robot.set_dofs_kp([env_cfg["kp"]] * len(joint_names), motors_idx)
    robot.set_dofs_kv([env_cfg["kd"]] * len(joint_names), motors_idx)

    action_joint_idx = np.array([robot.get_joint(name).dofs_idx_local[0] for name in env_cfg["action_joint_names"]])
    q_target = np.array(env_cfg["default_joint_pos"], dtype=np.float64)
    apply_dependent_rules(q_target)

    keyboard_device = KeyboardDevice()
    keyboard_device.start()

    print("Keyboard teleop started.")
    print("Arrow keys control joints 1 and 2. 'r' resets pose. 'q' quits.")

    running = True
    step = math.radians(args.step_deg)
    reset_needed = True

    try:
        while running:
            pressed = keyboard_device.snapshot()

            if keyboard.KeyCode.from_char("q") in pressed:
                running = False

            if keyboard.KeyCode.from_char("r") in pressed:
                q_target[:] = env_cfg["default_joint_pos"]
                apply_dependent_rules(q_target)
                reset_needed = True

            delta = np.zeros_like(action_joint_idx, dtype=np.float64)
            if keyboard.Key.left in pressed:
                delta[0] -= step
            if keyboard.Key.right in pressed:
                delta[0] += step
            if keyboard.Key.up in pressed:
                delta[1] += step
            if keyboard.Key.down in pressed:
                delta[1] -= step

            if np.any(delta):
                q_target[action_joint_idx] += delta

            q_target = np.clip(q_target, dof_lower_np, dof_upper_np)
            apply_dependent_rules(q_target)

            if reset_needed or np.any(delta):
                print(
                    f"joint1={q_target[0]: .3f} rad, joint2={q_target[1]: .3f} rad "
                    f"(joint3={q_target[2]: .3f}, joint4={q_target[3]: .3f}, joint5={q_target[4]: .3f})"
                )
                reset_needed = False

            robot.control_dofs_position(q_target, motors_idx)
            scene.step()
            time.sleep(env_cfg["ctrl_dt"])
    finally:
        keyboard_device.stop()
        print("Teleop terminated.")


if __name__ == "__main__":
    main()
