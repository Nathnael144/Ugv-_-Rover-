"""Direct Isaac Lab environment for the UGV rover in rectangular delivery arenas."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, sample_uniform, wrap_to_pi

if TYPE_CHECKING:
    from .ugv_rover_env_cfg import UGVRoverEmptyEnvCfg


class UGVRoverEmptyEnv(DirectRLEnv):
    """Goal-reaching rover task with differential-drive actions and planar LiDAR ranges."""

    cfg: UGVRoverEmptyEnvCfg

    def __init__(self, cfg: UGVRoverEmptyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        left_names = ["front_left", "rear_left"]
        right_names = ["front_right", "rear_right"]
        self._left_joint_ids, _ = self.robot.find_joints(left_names, preserve_order=True)
        self._right_joint_ids, _ = self.robot.find_joints(right_names, preserve_order=True)
        self._drive_joint_ids = self._left_joint_ids + self._right_joint_ids

        self._actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._goal_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)

        ray_angles = torch.linspace(-math.pi, math.pi, self.cfg.lidar_num_rays + 1, device=self.device)[:-1]
        self._lidar_ray_angles = ray_angles
        if len(self.cfg.obstacle_rects) > 0:
            self._obstacle_rects = torch.tensor(self.cfg.obstacle_rects, dtype=torch.float32, device=self.device)
        else:
            self._obstacle_rects = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        self._goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self._spawn_arena("/World/envs/env_0")

        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, positions, global_paths=("/World/ground",))
        cloner.replicate(plan)

        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.85, 0.9, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_arena(self, env_path: str) -> None:
        length = self.cfg.arena_length_m
        width = self.cfg.arena_width_m
        height = self.cfg.wall_height_m
        thickness = self.cfg.wall_thickness_m
        z = height * 0.5
        wall_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.74, 0.62))
        collision = sim_utils.CollisionPropertiesCfg()

        wall_specs = {
            "north_wall": ((length + 2.0 * thickness, thickness, height), (0.0, width * 0.5, z)),
            "south_wall": ((length + 2.0 * thickness, thickness, height), (0.0, -width * 0.5, z)),
            "east_wall": ((thickness, width, height), (length * 0.5, 0.0, z)),
            "west_wall": ((thickness, width, height), (-length * 0.5, 0.0, z)),
        }
        for name, (size, translation) in wall_specs.items():
            cfg = sim_utils.CuboidCfg(size=size, collision_props=collision, visual_material=wall_material)
            cfg.func(f"{env_path}/{name}", cfg, translation=translation)

        if len(self.cfg.obstacle_rects) > 0:
            self._spawn_city_details(env_path)

    def _spawn_city_details(self, env_path: str) -> None:
        """Spawn a static city-block scene that matches the obstacle rectangles."""
        length = self.cfg.arena_length_m
        width = self.cfg.arena_width_m
        collision = sim_utils.CollisionPropertiesCfg()

        asphalt_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.014, 0.013), roughness=0.82)
        sidewalk_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.43, 0.41), roughness=0.9)
        line_yellow = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.72, 0.05), roughness=0.55)
        line_white = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.86), roughness=0.6)

        asphalt = sim_utils.CuboidCfg(size=(length - 0.18, width - 0.18, 0.008), visual_material=asphalt_mat)
        asphalt.func(f"{env_path}/asphalt_delivery_block", asphalt, translation=(0.0, 0.0, 0.004))

        sidewalk_specs = {
            "sidewalk_north": ((length - 0.28, 0.30, 0.018), (0.0, width * 0.5 - 0.24, 0.018)),
            "sidewalk_south": ((length - 0.28, 0.30, 0.018), (0.0, -width * 0.5 + 0.24, 0.018)),
            "sidewalk_east": ((0.30, width - 0.60, 0.018), (length * 0.5 - 0.24, 0.0, 0.018)),
            "sidewalk_west": ((0.30, width - 0.60, 0.018), (-length * 0.5 + 0.24, 0.0, 0.018)),
        }
        for name, (size, translation) in sidewalk_specs.items():
            cfg = sim_utils.CuboidCfg(size=size, visual_material=sidewalk_mat)
            cfg.func(f"{env_path}/{name}", cfg, translation=translation)

        road_markings = {
            "yellow_center_x": ((length - 0.70, 0.035, 0.006), (0.0, 0.0, 0.014), line_yellow),
            "yellow_center_y": ((0.035, width - 0.70, 0.006), (0.0, 0.0, 0.015), line_yellow),
            "white_left_lane": ((length - 0.90, 0.025, 0.006), (0.0, 0.72, 0.015), line_white),
            "white_right_lane": ((length - 0.90, 0.025, 0.006), (0.0, -0.72, 0.015), line_white),
        }
        for name, (size, translation, material) in road_markings.items():
            cfg = sim_utils.CuboidCfg(size=size, visual_material=material)
            cfg.func(f"{env_path}/{name}", cfg, translation=translation)

        for i in range(7):
            x = -0.42 + i * 0.14
            stripe = sim_utils.CuboidCfg(size=(0.08, 0.78, 0.006), visual_material=line_white)
            stripe.func(f"{env_path}/crosswalk_stripe_{i}", stripe, translation=(x, -0.02, 0.017))

        building_mats = (
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.56, 0.52), roughness=0.75),
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.50, 0.42), roughness=0.78),
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.43, 0.48), roughness=0.72),
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.66, 0.55), roughness=0.8),
        )
        car_mats = (
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.05, 0.04), roughness=0.55),
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.18, 0.55), roughness=0.55),
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.55),
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.48, 0.26), roughness=0.55),
        )
        glass_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.18, 0.22), opacity=0.82, roughness=0.2)
        tire_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 0.01, 0.01), roughness=0.95)

        for index, (cx, cy, sx, sy) in enumerate(self.cfg.obstacle_rects):
            if index < 8:
                height = 0.55 + 0.10 * (index % 4)
                body = sim_utils.CuboidCfg(
                    size=(sx, sy, height),
                    collision_props=collision,
                    visual_material=building_mats[index % len(building_mats)],
                )
                body.func(f"{env_path}/building_{index:02d}", body, translation=(cx, cy, height * 0.5))

                roof = sim_utils.CuboidCfg(
                    size=(sx * 1.04, sy * 1.04, 0.035),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.17), roughness=0.9),
                )
                roof.func(f"{env_path}/building_{index:02d}_roof", roof, translation=(cx, cy, height + 0.025))

                for floor in range(2):
                    window = sim_utils.CuboidCfg(size=(sx * 0.55, 0.012, 0.055), visual_material=glass_mat)
                    window.func(
                        f"{env_path}/building_{index:02d}_window_{floor}",
                        window,
                        translation=(cx, cy - sy * 0.5 - 0.007, 0.20 + floor * 0.16),
                    )
            else:
                car_id = index - 8
                body = sim_utils.CuboidCfg(
                    size=(sx, sy, 0.20),
                    collision_props=collision,
                    visual_material=car_mats[car_id % len(car_mats)],
                )
                body.func(f"{env_path}/parked_car_{car_id:02d}", body, translation=(cx, cy, 0.10))

                cabin = sim_utils.CuboidCfg(size=(sx * 0.46, sy * 0.76, 0.13), visual_material=glass_mat)
                cabin.func(f"{env_path}/parked_car_{car_id:02d}_cabin", cabin, translation=(cx, cy, 0.245))

                for tire_x_index, wx in enumerate((-sx * 0.34, sx * 0.34)):
                    for tire_y_index, wy in enumerate((-sy * 0.56, sy * 0.56)):
                        tire = sim_utils.CuboidCfg(size=(0.065, 0.035, 0.10), visual_material=tire_mat)
                        tire.func(
                            f"{env_path}/parked_car_{car_id:02d}_tire_{tire_x_index}_{tire_y_index}",
                            tire,
                            translation=(cx + wx, cy + wy, 0.055),
                        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions[:] = self._actions
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        linear_m_s = self._actions[:, 0] * self.cfg.max_linear_m_s
        angular_rad_s = self._actions[:, 1] * self.cfg.max_angular_rad_s
        cad_linear_m_s = linear_m_s * self.cfg.linear_drive_sign
        left_m_s = cad_linear_m_s - angular_rad_s * self.cfg.track_width_m * 0.5
        right_m_s = cad_linear_m_s + angular_rad_s * self.cfg.track_width_m * 0.5

        # Isaac Lab velocity targets are joint rates in rad/s.
        left_rad_s = -left_m_s / self.cfg.wheel_radius_m
        right_rad_s = -right_m_s / self.cfg.wheel_radius_m
        targets = torch.stack((left_rad_s, left_rad_s, right_rad_s, right_rad_s), dim=-1)
        self.robot.set_joint_velocity_target_index(target=targets, joint_ids=self._drive_joint_ids)
        self._update_follow_camera()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos = self.robot.data.root_pos_w.torch
        root_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        root_quat = self.robot.data.root_quat_w.torch
        root_lin_vel_b = self.robot.data.root_lin_vel_b.torch
        root_ang_vel_b = self.robot.data.root_ang_vel_b.torch
        _, _, yaw = euler_xyz_from_quat(root_quat)
        front_yaw = self._front_yaw(yaw)
        front_lin_vel_b = self._cad_to_front_body_xy(root_lin_vel_b[:, :2])

        goal_delta_w = self._goal_xy - root_xy
        distance = torch.linalg.norm(goal_delta_w, dim=-1).clamp_min(1e-6)
        goal_direction = goal_delta_w / distance.unsqueeze(-1)
        goal_heading = torch.atan2(goal_delta_w[:, 1], goal_delta_w[:, 0])
        heading_error = wrap_to_pi(goal_heading - front_yaw)
        lidar = self._compute_lidar_ranges(root_xy, front_yaw)

        obs = torch.cat(
            (
                goal_delta_w / torch.tensor(
                    [self.cfg.arena_length_m * 0.5, self.cfg.arena_width_m * 0.5], device=self.device
                ),
                distance.unsqueeze(-1) / self.cfg.lidar_max_range_m,
                torch.sin(heading_error).unsqueeze(-1),
                torch.cos(heading_error).unsqueeze(-1),
                goal_direction,
                front_lin_vel_b / self.cfg.max_linear_m_s,
                root_ang_vel_b[:, 2:3] / self.cfg.max_angular_rad_s,
                self._actions,
                self._previous_actions,
                lidar / self.cfg.lidar_max_range_m,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w.torch
        root_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        root_quat = self.robot.data.root_quat_w.torch
        root_lin_vel_b = self.robot.data.root_lin_vel_b.torch
        root_ang_vel_b = self.robot.data.root_ang_vel_b.torch
        _, _, yaw = euler_xyz_from_quat(root_quat)
        front_yaw = self._front_yaw(yaw)
        front_lin_vel_b = self._cad_to_front_body_xy(root_lin_vel_b[:, :2])

        goal_delta = self._goal_xy - root_xy
        distance = torch.linalg.norm(goal_delta, dim=-1)
        progress = self._previous_goal_distance - distance
        self._previous_goal_distance = distance

        goal_heading = torch.atan2(goal_delta[:, 1], goal_delta[:, 0])
        heading_error = torch.abs(wrap_to_pi(goal_heading - front_yaw))
        reached = distance < self.cfg.goal_reached_radius_m
        wall_margin = self._wall_margin(root_xy)
        near_wall = (self.cfg.safety_margin_m - wall_margin).clamp_min(0.0)
        obstacle_margin = self._obstacle_margin(root_xy)
        near_obstacle = (self.cfg.obstacle_safety_margin_m - obstacle_margin).clamp_min(0.0)

        forward_speed = (front_lin_vel_b[:, 0] / self.cfg.max_linear_m_s).clamp(-1.0, 1.0)
        forward_motion = forward_speed.clamp_min(0.0)
        reverse_action = (-self._actions[:, 0]).clamp_min(0.0)
        wall_pressure = (near_wall / self.cfg.safety_margin_m).clamp(0.0, 1.0)
        obstacle_pressure = (near_obstacle / self.cfg.obstacle_safety_margin_m).clamp(0.0, 1.0)
        clearance_pressure = torch.maximum(wall_pressure, obstacle_pressure)
        progressing = (progress > 0.0).float()
        reverse_penalty_scale = (1.0 - 0.70 * clearance_pressure - 0.40 * progressing).clamp(0.15, 1.0)
        reverse_progress = reverse_action * progress.clamp_min(0.0)
        yaw_rate = torch.abs(root_ang_vel_b[:, 2]) / self.cfg.max_angular_rad_s
        heading_turn_need = (heading_error / math.pi).clamp(0.0, 1.0)
        yaw_penalty_scale = ((1.0 - 0.75 * clearance_pressure) * (1.0 - 0.50 * heading_turn_need)).clamp(0.10, 1.0)
        turn_action = torch.abs(self._actions[:, 1])
        unneeded_turn = turn_action * torch.cos(heading_error).clamp_min(0.0) * (1.0 - clearance_pressure)
        clearance_turn = turn_action * clearance_pressure * (0.35 + 0.65 * heading_turn_need)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=-1)
        action_mag = torch.sum(torch.square(self._actions), dim=-1)
        reward = (
            self.cfg.rew_alive
            + self.cfg.rew_progress * progress
            + self.cfg.rew_heading * torch.cos(heading_error)
            + self.cfg.rew_goal * reached.float()
            + self.cfg.rew_forward_velocity * forward_motion
            + self.cfg.rew_reverse_action * reverse_action * reverse_penalty_scale
            + self.cfg.rew_reverse_progress * reverse_progress
            + self.cfg.rew_yaw_rate * yaw_rate * yaw_penalty_scale
            + self.cfg.rew_unneeded_turn * unneeded_turn
            + self.cfg.rew_clearance_turn * clearance_turn
            + self.cfg.rew_action_rate * action_rate
            + self.cfg.rew_action_mag * action_mag
            + self.cfg.rew_wall_margin * near_wall
            + self.cfg.rew_obstacle_margin * near_obstacle
            + self.cfg.rew_terminated * self.reset_terminated.float()
        )
        if torch.any(reached):
            reached_ids = reached.nonzero(as_tuple=False).squeeze(-1)
            self._sample_goals(reached_ids, root_xy[reached_ids])
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.robot.data.root_pos_w.torch
        root_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        root_quat = self.robot.data.root_quat_w.torch
        roll, pitch, _ = euler_xyz_from_quat(root_quat)

        out_of_arena = self._wall_margin(root_xy) < 0.0
        hit_obstacle = self._obstacle_margin(root_xy) < 0.12
        tipped = (torch.abs(roll) > 0.75) | (torch.abs(pitch) > 0.75)
        time_out = self.episode_length_buf >= self.max_episode_length
        return out_of_arena | hit_obstacle | tipped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_resets = len(env_ids_t)
        spawn_xy = self._sample_spawn_points(num_resets)
        yaw = sample_uniform(-math.pi, math.pi, (num_resets,), self.device)

        self._sample_goals(env_ids_t, spawn_xy)

        root_pose = self.robot.data.default_root_pose.torch[env_ids_t].clone()
        root_pose[:, :2] = spawn_xy + self.scene.env_origins[env_ids_t, :2]
        root_pose[:, 2] = self.robot.data.default_root_pose.torch[env_ids_t, 2]
        root_pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        root_vel = torch.zeros_like(self.robot.data.default_root_vel.torch[env_ids_t])

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids_t].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel.torch[env_ids_t])

        self._actions[env_ids_t] = 0.0
        self._previous_actions[env_ids_t] = 0.0

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids_t)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids_t)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids_t)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids_t)

    def _sample_goals(self, env_ids_t: torch.Tensor, avoid_xy: torch.Tensor) -> None:
        """Sample reachable goal points and refresh the visual target markers."""
        if len(env_ids_t) == 0:
            return

        half_length = self.cfg.arena_length_m * 0.5 - self.cfg.safety_margin_m
        half_width = self.cfg.arena_width_m * 0.5 - self.cfg.safety_margin_m
        goal_xy = torch.zeros((len(env_ids_t), 2), device=self.device)
        needs_resample = torch.ones(len(env_ids_t), dtype=torch.bool, device=self.device)

        for _ in range(16):
            count = int(torch.sum(needs_resample).item())
            if count == 0:
                break
            candidates = torch.zeros((count, 2), device=self.device)
            candidates[:, 0] = sample_uniform(-half_length, half_length, (count,), self.device)
            candidates[:, 1] = sample_uniform(-half_width, half_width, (count,), self.device)
            candidate_distance = torch.linalg.norm(candidates - avoid_xy[needs_resample], dim=-1)
            accepted = (candidate_distance > self.cfg.min_goal_distance_m) & self._is_free_xy(
                candidates, self.cfg.obstacle_safety_margin_m
            )
            pending_indices = needs_resample.nonzero(as_tuple=False).squeeze(-1)
            goal_xy[pending_indices[accepted]] = candidates[accepted]
            needs_resample[pending_indices[accepted]] = False

        if torch.any(needs_resample):
            fallback = needs_resample.nonzero(as_tuple=False).squeeze(-1)
            goal_xy[fallback, 0] = torch.sign(avoid_xy[fallback, 0]).where(
                avoid_xy[fallback, 0] != 0.0,
                torch.ones_like(avoid_xy[fallback, 0]),
            ) * -half_length
            goal_xy[fallback, 1] = -avoid_xy[fallback, 1].clamp(-half_width, half_width)

        self._goal_xy[env_ids_t] = goal_xy
        self._previous_goal_distance[env_ids_t] = torch.linalg.norm(goal_xy - avoid_xy, dim=-1)
        self._update_goal_markers()

    def _update_goal_markers(self) -> None:
        if not hasattr(self, "_goal_markers"):
            return
        translations = torch.zeros((self.num_envs, 3), device=self.device)
        translations[:, :2] = self._goal_xy + self.scene.env_origins[:, :2]
        translations[:, 2] = 0.012
        self._goal_markers.visualize(translations=translations, environment_ids=self.scene._ALL_INDICES)

    def _wall_margin(self, xy: torch.Tensor) -> torch.Tensor:
        half_length = self.cfg.arena_length_m * 0.5
        half_width = self.cfg.arena_width_m * 0.5
        return torch.minimum(half_length - torch.abs(xy[:, 0]), half_width - torch.abs(xy[:, 1]))

    def _obstacle_margin(self, xy: torch.Tensor) -> torch.Tensor:
        """Signed distance from each point to the nearest obstacle rectangle."""
        if self._obstacle_rects.numel() == 0:
            return torch.full((xy.shape[0],), self.cfg.lidar_max_range_m, device=self.device)

        centers = self._obstacle_rects[:, :2].unsqueeze(0)
        half_sizes = (self._obstacle_rects[:, 2:4] * 0.5).unsqueeze(0)
        delta = torch.abs(xy.unsqueeze(1) - centers) - half_sizes
        outside = torch.linalg.norm(delta.clamp_min(0.0), dim=-1)
        inside = torch.minimum(torch.maximum(delta[..., 0], delta[..., 1]), torch.zeros_like(outside))
        signed_distance = outside + inside
        return torch.min(signed_distance, dim=1).values

    def _is_free_xy(self, xy: torch.Tensor, margin: float) -> torch.Tensor:
        wall_ok = self._wall_margin(xy) > self.cfg.safety_margin_m
        obstacle_ok = self._obstacle_margin(xy) > margin
        return wall_ok & obstacle_ok

    def _sample_spawn_points(self, count: int) -> torch.Tensor:
        spawn_xy = torch.zeros((count, 2), device=self.device)
        needs_resample = torch.ones(count, dtype=torch.bool, device=self.device)
        for _ in range(32):
            pending = int(torch.sum(needs_resample).item())
            if pending == 0:
                break
            candidates = torch.zeros((pending, 2), device=self.device)
            candidates[:, 0] = sample_uniform(-1.35, 1.35, (pending,), self.device)
            candidates[:, 1] = sample_uniform(-0.65, 0.65, (pending,), self.device)
            accepted = self._is_free_xy(candidates, self.cfg.obstacle_safety_margin_m)
            pending_indices = needs_resample.nonzero(as_tuple=False).squeeze(-1)
            spawn_xy[pending_indices[accepted]] = candidates[accepted]
            needs_resample[pending_indices[accepted]] = False

        if torch.any(needs_resample):
            spawn_xy[needs_resample] = torch.tensor((0.0, 0.0), device=self.device)
        return spawn_xy

    def _update_follow_camera(self) -> None:
        """Orbit the recording camera around one env while tracking robot progress."""
        if not self.cfg.enable_follow_camera:
            return
        if self.common_step_counter % self.cfg.follow_camera_update_interval != 0:
            return

        env_id = min(self.cfg.follow_camera_env_id, self.num_envs - 1)
        robot_xy = self.robot.data.root_pos_w.torch[env_id, :2]
        goal_xy = self._goal_xy[env_id] + self.scene.env_origins[env_id, :2]
        blend = self.cfg.follow_camera_goal_blend
        path_center = robot_xy * (1.0 - blend) + goal_xy * blend

        center_x = float(path_center[0].item())
        center_y = float(path_center[1].item())
        sim_time = self.common_step_counter * self.step_dt
        orbit_angle = sim_time * self.cfg.follow_camera_orbit_rad_per_s
        zoom_phase = math.sin(sim_time * self.cfg.follow_camera_zoom_rad_per_s)
        radius = self.cfg.follow_camera_orbit_radius_m + self.cfg.follow_camera_zoom_amplitude_m * zoom_phase
        height = self.cfg.follow_camera_height_m + (
            self.cfg.follow_camera_zoom_amplitude_m * self.cfg.follow_camera_height_zoom_scale * zoom_phase
        )
        eye = (
            center_x + radius * math.cos(orbit_angle),
            center_y + radius * math.sin(orbit_angle),
            height,
        )
        target = (center_x, center_y, 0.08)
        self.sim.set_camera_view(eye=eye, target=target)

    def _front_yaw(self, cad_yaw: torch.Tensor) -> torch.Tensor:
        return wrap_to_pi(cad_yaw + self.cfg.front_yaw_offset_rad)

    def _cad_to_front_body_xy(self, cad_xy: torch.Tensor) -> torch.Tensor:
        offset = self.cfg.front_yaw_offset_rad
        cos_offset = math.cos(offset)
        sin_offset = math.sin(offset)
        return torch.stack(
            (
                cos_offset * cad_xy[:, 0] + sin_offset * cad_xy[:, 1],
                -sin_offset * cad_xy[:, 0] + cos_offset * cad_xy[:, 1],
            ),
            dim=-1,
        )

    def _compute_lidar_ranges(self, xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        half_length = self.cfg.arena_length_m * 0.5
        half_width = self.cfg.arena_width_m * 0.5
        angles = yaw.unsqueeze(-1) + self._lidar_ray_angles.unsqueeze(0)
        dx = torch.cos(angles)
        dy = torch.sin(angles)
        eps = 1e-6

        tx = torch.where(dx > 0.0, (half_length - xy[:, 0:1]) / dx.clamp_min(eps), (-half_length - xy[:, 0:1]) / dx.clamp_max(-eps))
        ty = torch.where(dy > 0.0, (half_width - xy[:, 1:2]) / dy.clamp_min(eps), (-half_width - xy[:, 1:2]) / dy.clamp_max(-eps))
        ranges = torch.minimum(tx, ty)
        ranges = ranges.clamp(0.0, self.cfg.lidar_max_range_m)

        if self._obstacle_rects.numel() == 0:
            return ranges

        for cx, cy, sx, sy in self._obstacle_rects:
            obs_min_x = cx - sx * 0.5
            obs_max_x = cx + sx * 0.5
            obs_min_y = cy - sy * 0.5
            obs_max_y = cy + sy * 0.5
            ranges = torch.minimum(
                ranges,
                self._ray_aabb_intersections(xy, dx, dy, obs_min_x, obs_max_x, obs_min_y, obs_max_y),
            )
        return ranges

    def _ray_aabb_intersections(
        self,
        xy: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
        min_x: torch.Tensor,
        max_x: torch.Tensor,
        min_y: torch.Tensor,
        max_y: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-6
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        dx_valid = torch.abs(dx) > eps
        dy_valid = torch.abs(dy) > eps
        dx_safe = torch.where(dx_valid, dx, torch.ones_like(dx))
        dy_safe = torch.where(dy_valid, dy, torch.ones_like(dy))

        tx1 = (min_x - x) / dx_safe
        tx2 = (max_x - x) / dx_safe
        ty1 = (min_y - y) / dy_safe
        ty2 = (max_y - y) / dy_safe

        tx_near = torch.minimum(tx1, tx2)
        tx_far = torch.maximum(tx1, tx2)
        ty_near = torch.minimum(ty1, ty2)
        ty_far = torch.maximum(ty1, ty2)

        x_inside = (x >= min_x) & (x <= max_x)
        y_inside = (y >= min_y) & (y <= max_y)
        neg_inf = torch.full_like(dx, -float("inf"))
        pos_inf = torch.full_like(dx, float("inf"))

        tx_near = torch.where(dx_valid, tx_near, torch.where(x_inside, neg_inf, pos_inf))
        tx_far = torch.where(dx_valid, tx_far, torch.where(x_inside, pos_inf, neg_inf))
        ty_near = torch.where(dy_valid, ty_near, torch.where(y_inside, neg_inf, pos_inf))
        ty_far = torch.where(dy_valid, ty_far, torch.where(y_inside, pos_inf, neg_inf))

        t_near = torch.maximum(tx_near, ty_near)
        t_far = torch.minimum(tx_far, ty_far)
        hit = (t_far >= torch.maximum(t_near, torch.zeros_like(t_near))) & (t_far >= 0.0)
        hit_distance = t_near.clamp_min(0.0).clamp_max(self.cfg.lidar_max_range_m)
        no_hit = torch.full_like(hit_distance, self.cfg.lidar_max_range_m)
        return torch.where(hit, hit_distance, no_hit)
