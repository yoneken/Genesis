import math
from typing import Dict

import torch

import genesis as gs
from genesis.utils.geom import transform_by_quat


class XArmPendulumEnv:
    """Parallel RL environment for balancing the pendulum with the UFactory xArm6."""

    def __init__(
        self, num_envs: int, env_cfg: Dict, obs_cfg: Dict, reward_cfg: Dict, show_viewer: bool = False
    ) -> None:
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"].copy()

        self.dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.action_scale = env_cfg["action_scale"]
        self.max_joint_velocity = env_cfg.get("max_joint_velocity", math.pi)
        self.max_action = env_cfg.get("max_action", 1.0)
        self.init_joint_noise = env_cfg.get("init_joint_noise", 0.05)
        self.init_pendulum_noise = env_cfg.get("init_pendulum_noise", 0.1)
        self.default_pendulum_angle = env_cfg.get("default_pendulum_angle", 0.0)
        self.termination_cos_threshold = env_cfg.get("termination_cos_threshold", 0.4)
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)

        # === scene ===
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=env_cfg.get("sim_substeps", 2),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                gravity=(0, 0, -9.8),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-0.5, -1.0, 0.4),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=38,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=env_cfg["mjcf_file"]),
        )

        self.scene.build(n_envs=num_envs)

        # === robot handles ===
        self.joint_names = env_cfg["joint_names"]
        self.arm_num_dofs = len(self.joint_names)
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.action_joint_names = env_cfg.get("action_joint_names", self.joint_names[: self.num_actions])
        assert (
            len(self.action_joint_names) == self.num_actions
        ), "Number of action joints must match num_actions in config."
        self.action_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.action_joint_names],
            dtype=torch.long,
            device=gs.device,
        )
        self.wrist_joint_names = env_cfg.get("wrist_joint_names", [])
        self.wrist_dof_idx = torch.tensor([], dtype=torch.long, device=gs.device)
        self.wrist_dof_idx_list = []
        self.dependent_joint_mappings = []
        for mapping in env_cfg.get("dependent_joints", []):
            target_joint = self.robot.get_joint(mapping["target"])
            source_joint = self.robot.get_joint(mapping["source"])
            self.dependent_joint_mappings.append(
                (
                    int(target_joint.dofs_idx_local[0]),
                    int(source_joint.dofs_idx_local[0]),
                    float(mapping.get("scale", 1.0)),
                    float(mapping.get("offset", 0.0)),
                )
            )
        self.pendulum_joint = self.robot.get_joint(env_cfg["pendulum_joint_name"])
        self.pendulum_dof_idx = torch.tensor([self.pendulum_joint.dofs_idx_local[0]], dtype=gs.tc_int, device=gs.device)
        self.pendulum_link = self.robot.get_link(env_cfg["pendulum_link_name"])
        self.pendulum_link_idx = self.pendulum_link.idx_local

        self.default_joint_pos = torch.tensor(env_cfg["default_joint_pos"], dtype=gs.tc_float, device=gs.device)
        self._default_joint_pos_batched = self.default_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
        dof_lower, dof_upper = self.robot.get_dofs_limit(self.motors_dof_idx)
        self.dof_lower = dof_lower.to(gs.device)
        self.dof_upper = dof_upper.to(gs.device)
        joint_limit_overrides = env_cfg.get("joint_limit_overrides", {})
        if joint_limit_overrides:
            joint_name_to_idx = {name: self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names}
            for joint_name, (lower, upper) in joint_limit_overrides.items():
                idx = joint_name_to_idx[joint_name]
                self.dof_lower[idx] = lower
                self.dof_upper[idx] = upper

        kp = env_cfg.get("kp", 600.0)
        kd = env_cfg.get("kd", 60.0)
        if isinstance(kp, (int, float)):
            kp = [kp] * self.arm_num_dofs
        else:
            assert len(kp) == self.arm_num_dofs, "Length of kp gains must match number of controllable joints."
        if isinstance(kd, (int, float)):
            kd = [kd] * self.arm_num_dofs
        else:
            assert len(kd) == self.arm_num_dofs, "Length of kd gains must match number of controllable joints."
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd, self.motors_dof_idx)

        self.pendulum_axis_local = torch.tensor(
            env_cfg.get("pendulum_axis_local", (1.0, 0.0, 0.0)),
            dtype=gs.tc_float,
            device=gs.device,
        ).view(1, 3)
        self.hinge_axis_local = torch.tensor(
            env_cfg.get("pendulum_hinge_axis_local", (1.0, 0.0, 0.0)),
            dtype=gs.tc_float,
            device=gs.device,
        ).view(1, 3)
        self.gravity_axis = torch.tensor(
            env_cfg.get("gravity_axis", (0.0, 0.0, 1.0)), dtype=gs.tc_float, device=gs.device
        )
        self.pendulum_damping_range = self._parse_range(env_cfg.get("pendulum_damping_range"))
        self.pendulum_mass_range = self._parse_range(env_cfg.get("pendulum_mass_range"))
        self.pendulum_length_range = self._parse_range(env_cfg.get("pendulum_length_range"))
        self.action_center_ranges = {}
        ranges_cfg = env_cfg.get("action_center_ranges")
        if isinstance(ranges_cfg, dict):
            for joint_name, rng in ranges_cfg.items():
                parsed = self._parse_range(rng)
                joint_idx = self.robot.get_joint(joint_name).dofs_idx_local[0]
                self.action_center_ranges[joint_idx] = parsed
        elif ranges_cfg is not None:
            parsed = self._parse_range(ranges_cfg)
            for idx in self.action_dof_idx.tolist():
                self.action_center_ranges[int(idx)] = parsed
        self.action_center_penalty = env_cfg.get("action_center_penalty", 1.0)
        self.action_center_sigma = max(env_cfg.get("action_center_sigma", 0.1), 1e-6)

        # reward helpers
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # === buffers ===
        self.dof_pos = torch.zeros((self.num_envs, self.arm_num_dofs), dtype=gs.tc_float, device=gs.device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.pendulum_joint_pos = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.pendulum_joint_vel = torch.zeros_like(self.pendulum_joint_pos)
        self.pendulum_dir = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.tip_alignment = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.tilt_squared = torch.zeros_like(self.tip_alignment)
        self.pendulum_axis_speed = torch.zeros_like(self.tip_alignment)
        self.current_pendulum_length = torch.full(
            (self.num_envs,), env_cfg.get("pendulum_nominal_length", 1.0), dtype=gs.tc_float, device=gs.device
        )
        self.current_pendulum_mass = torch.full(
            (self.num_envs,),
            self.robot.get_links_inertial_mass(links_idx_local=[self.pendulum_link_idx]).squeeze().item(),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.current_pendulum_damping = torch.full(
            (self.num_envs,),
            self.robot.get_dofs_damping(self.pendulum_dof_idx).squeeze().item(),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.bool, device=gs.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.extras = {"observations": {}}

        self.reset()

    # --------------------------------------------------------------------- #
    # RL API
    # --------------------------------------------------------------------- #
    def step(self, actions: torch.Tensor):
        # Interpret incoming actions as joint velocity commands on action joints.
        actions = torch.clamp(actions, -self.max_action, self.max_action)
        self.actions = actions

        commanded_actions = self.last_actions if self.simulate_action_latency else self.actions
        # Scale to physical velocity and clamp
        desired_vel = torch.zeros_like(self.dof_vel)
        vel_cmd = torch.clamp(commanded_actions * self.action_scale, -self.max_joint_velocity, self.max_joint_velocity)
        desired_vel[:, self.action_dof_idx] = vel_cmd
        # Apply dependent joint constraints for velocities (scale only, ignore offsets)
        for target_idx, source_idx, scale, _ in self.dependent_joint_mappings:
            desired_vel[..., target_idx] = desired_vel[..., source_idx] * scale

        self.robot.control_dofs_velocity(desired_vel, self.motors_dof_idx)

        self.scene.step()
        self.episode_length_buf += 1

        self._refresh_kinematics()
        self._compute_observations()

        self.rew_buf.zero_()
        for name, func in self.reward_functions.items():
            rew = func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        time_out = self.episode_length_buf >= self.max_episode_length
        fell_over = self.tip_alignment < self.termination_cos_threshold
        done = time_out | fell_over
        done_out = done.clone()
        self.reset_buf = done

        self.extras["time_outs"] = torch.zeros_like(self.rew_buf, dtype=gs.tc_float, device=gs.device)
        self.extras["time_outs"][time_out & ~fell_over] = 1.0
        self.extras["done_mask"] = done_out

        reset_envs = torch.nonzero(self.reset_buf, as_tuple=False).squeeze(-1)
        self.reset_idx(reset_envs)

        self.last_actions.copy_(self.actions)
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, done_out, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _parse_range(self, value):
        if value is None:
            return None
        low, high = float(value[0]), float(value[1])
        if low > high:
            low, high = high, low
        return low, high

    def _sample_range(self, rng, shape):
        if rng is None:
            return None
        low, high = rng
        return torch.rand(shape, device=gs.device, dtype=gs.tc_float) * (high - low) + low

    def _apply_dependent_joint_constraints(self, joint_tensor: torch.Tensor):
        if not self.dependent_joint_mappings:
            return joint_tensor
        for target_idx, source_idx, scale, offset in self.dependent_joint_mappings:
            joint_tensor[..., target_idx] = joint_tensor[..., source_idx] * scale + offset
        return joint_tensor

    def _randomize_pendulum(self, envs_idx: torch.Tensor):
        if envs_idx.numel() == 0:
            return
        num_reset = envs_idx.shape[0]
        if self.pendulum_damping_range is not None:
            damping = self._sample_range(self.pendulum_damping_range, (num_reset,))
            for i in range(num_reset):
                env_slice = envs_idx[i : i + 1]
                self.robot.set_dofs_damping(
                    damping[i : i + 1], dofs_idx_local=self.pendulum_dof_idx, envs_idx=env_slice
                )
            self.current_pendulum_damping[envs_idx] = damping
        if self.pendulum_mass_range is not None:
            mass = self._sample_range(self.pendulum_mass_range, (num_reset,))
            for i in range(num_reset):
                env_slice = envs_idx[i : i + 1]
                self.robot.set_links_inertial_mass(
                    mass[i : i + 1], links_idx_local=[self.pendulum_link_idx], envs_idx=env_slice
                )
            self.current_pendulum_mass[envs_idx] = mass
        if self.pendulum_length_range is not None:
            length = self._sample_range(self.pendulum_length_range, (num_reset,))
            com_shift = torch.zeros((num_reset, 3), dtype=gs.tc_float, device=gs.device)
            com_shift[:, 0] = length / 2.0
            self.robot.set_COM_shift(com_shift, links_idx_local=[self.pendulum_link_idx], envs_idx=envs_idx)
            self.current_pendulum_length[envs_idx] = length

    def reset_idx(self, envs_idx: torch.Tensor):
        if envs_idx.numel() == 0:
            return
        envs_idx = envs_idx.to(dtype=torch.long)
        num_reset = envs_idx.shape[0]

        self._randomize_pendulum(envs_idx)

        # reset robot joints
        joint_noise = 2.0 * torch.rand((num_reset, self.arm_num_dofs), dtype=gs.tc_float, device=gs.device) - 1.0
        joint_target = torch.clamp(
            self._default_joint_pos_batched[envs_idx] + joint_noise * self.init_joint_noise,
            self.dof_lower,
            self.dof_upper,
        )
        self._apply_dependent_joint_constraints(joint_target)
        self.robot.set_dofs_position(joint_target, dofs_idx_local=self.motors_dof_idx, envs_idx=envs_idx)
        self.dof_pos[envs_idx] = joint_target
        self.dof_vel[envs_idx] = 0.0

        # reset pendulum
        pendulum_target = self.default_pendulum_angle + (
            (2.0 * torch.rand((num_reset, 1), dtype=gs.tc_float, device=gs.device) - 1.0) * self.init_pendulum_noise
        )
        self.robot.set_dofs_position(pendulum_target, dofs_idx_local=self.pendulum_dof_idx, envs_idx=envs_idx)
        self.pendulum_joint_pos[envs_idx] = pendulum_target[:, 0]
        self.pendulum_joint_vel[envs_idx] = 0.0

        pendulum_quat = self.pendulum_link.get_quat(envs_idx=envs_idx)
        axis = self.pendulum_axis_local.expand(num_reset, -1)
        self.pendulum_dir[envs_idx] = transform_by_quat(axis, pendulum_quat)
        self.tip_alignment[envs_idx] = torch.sum(self.pendulum_dir[envs_idx] * self.gravity_axis.view(1, 3), dim=1)
        self.tilt_squared[envs_idx] = torch.clamp(1.0 - torch.square(self.tip_alignment[envs_idx]), min=0.0, max=1.0)

        self.last_actions[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = False

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if num_reset > 0:
                self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
                )
            self.episode_sums[key][envs_idx] = 0.0

        self._compute_observations()

    def _refresh_kinematics(self):
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        pendulum_quat = self.pendulum_link.get_quat()
        axis = self.pendulum_axis_local.expand(self.num_envs, -1)
        hinge_axis = self.hinge_axis_local.expand(self.num_envs, -1)
        self.pendulum_dir = transform_by_quat(axis, pendulum_quat)
        self.pendulum_joint_pos = self.robot.get_dofs_position(self.pendulum_dof_idx).squeeze(-1)
        self.pendulum_joint_vel = self.robot.get_dofs_velocity(self.pendulum_dof_idx).squeeze(-1)
        hinge_world = transform_by_quat(hinge_axis, pendulum_quat)
        pendulum_ang = self.pendulum_link.get_ang()
        self.tip_alignment = torch.sum(self.pendulum_dir * self.gravity_axis.view(1, 3), dim=1)
        self.tilt_squared = torch.clamp(1.0 - torch.square(self.tip_alignment), min=0.0, max=1.0)
        self.pendulum_axis_speed = torch.sum(hinge_world * pendulum_ang, dim=1)

    def _compute_observations(self):
        joint_pos_err = (self.dof_pos - self._default_joint_pos_batched) * self.obs_scales["joint_pos"]
        joint_vel = self.dof_vel * self.obs_scales["joint_vel"]
        pendulum_sin = torch.sin(self.pendulum_joint_pos).unsqueeze(-1)
        pendulum_cos = torch.cos(self.pendulum_joint_pos).unsqueeze(-1)
        pendulum_vel = self.pendulum_joint_vel.unsqueeze(-1) * self.obs_scales["pendulum_vel"]
        self.obs_buf = torch.cat(
            (
                joint_pos_err,
                joint_vel,
                self.pendulum_dir,
                pendulum_sin,
                pendulum_cos,
                pendulum_vel,
                self.last_actions,
            ),
            dim=-1,
        )

    # --------------------------------------------------------------------- #
    # Rewards
    # --------------------------------------------------------------------- #
    def _reward_upright(self):
        return torch.clip(self.tip_alignment, min=-1.0, max=1.0)

    def _reward_tilt(self):
        return self.tilt_squared

    def _reward_joint_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_pendulum_velocity(self):
        return torch.square(self.pendulum_axis_speed)

    def _reward_action_center(self):
        if not self.action_center_ranges:
            return torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        rewards = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        for idx, rng in self.action_center_ranges.items():
            joint_pos = self.dof_pos[:, idx]
            target = self.default_joint_pos[idx]
            deviation = torch.abs(joint_pos - target)
            rewards += torch.exp(-deviation / self.action_center_sigma)
            if rng is not None:
                low, high = rng
                below = torch.clamp(low - joint_pos, min=0.0)
                above = torch.clamp(joint_pos - high, min=0.0)
                rewards -= self.action_center_penalty * (below + above)
        return rewards
