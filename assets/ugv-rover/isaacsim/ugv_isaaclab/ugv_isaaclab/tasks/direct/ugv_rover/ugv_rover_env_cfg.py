"""Environment configurations for UGV rover goal-navigation RL."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg


ISAACSIM_ASSET_DIR = Path(__file__).resolve().parents[5]
UGV_ROVER_USD = ISAACSIM_ASSET_DIR / "ugv_rover_physics_rl.usda"


@configclass
class UGVRoverEmptyEnvCfg(DirectRLEnvCfg):
    """MDP for goal reaching in a 7 x 5 m rectangular arena."""

    # MDP timing.
    decimation = 4
    episode_length_s = 20.0
    action_space = 2
    observation_space = 30
    state_space = 0

    # Simulation.
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    viewer: ViewerCfg = ViewerCfg(eye=(4.0, -4.0, 3.0), lookat=(0.0, 0.0, 0.0))

    # Scene. Start modestly; increase num_envs after the first smoke test passes.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=8.0, replicate_physics=True)

    # Robot.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(UGV_ROVER_USD), fix_root_link=False),
        articulation_root_prim_path="/UGV_Rover",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.04), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={
            "drive_wheels": ImplicitActuatorCfg(
                joint_names_expr=["front_left", "rear_left", "front_right", "rear_right"],
                stiffness=0.0,
                damping=4.0,
                joint_effort_limit=1.2,
                joint_velocity_limit=80.0,
            )
        },
    )

    # Arena.
    arena_length_m = 7.0
    arena_width_m = 5.0
    wall_height_m = 0.35
    wall_thickness_m = 0.08
    safety_margin_m = 0.25

    # Differential-drive geometry.
    wheel_radius_m = 0.040
    track_width_m = 0.17462
    max_linear_m_s = 1.0
    max_angular_rad_s = 2.0
    front_yaw_offset_rad = 3.141592653589793
    linear_drive_sign = -1.0

    # Perception. This is a planar LiDAR abstraction to the rectangular walls.
    lidar_num_rays = 16
    lidar_max_range_m = 4.0

    # Randomized goal sampling.
    goal_reached_radius_m = 0.25
    min_goal_distance_m = 1.0
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/UGV_Rover/goal_marker",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=0.25,
                height=0.012,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.25), opacity=0.75),
            ),
        },
    )

    # Rewards.
    rew_alive = 0.02
    rew_progress = 4.0
    rew_goal = 5.0
    rew_heading = 0.15
    rew_forward_velocity = 0.20
    rew_reverse_action = -0.25
    rew_yaw_rate = -0.015
    rew_unneeded_turn = -0.04
    rew_action_rate = -0.03
    rew_action_mag = -0.01
    rew_wall_margin = -0.25
    rew_obstacle_margin = -0.35
    rew_terminated = -3.0

    # Static obstacles are axis-aligned local-environment boxes:
    # (center_x, center_y, size_x, size_y). The empty task leaves this disabled.
    obstacle_rects: tuple[tuple[float, float, float, float], ...] = ()
    obstacle_safety_margin_m = 0.28

    # Recording camera. Disabled for the empty task; city training enables it.
    enable_follow_camera = False
    follow_camera_env_id = 0
    follow_camera_update_interval = 2
    follow_camera_orbit_radius_m = 3.6
    follow_camera_zoom_amplitude_m = 0.25
    follow_camera_height_m = 2.55
    follow_camera_height_zoom_scale = 0.25
    follow_camera_orbit_rad_per_s = 0.10
    follow_camera_zoom_rad_per_s = 0.06
    follow_camera_goal_blend = 0.30


@configclass
class UGVRoverCityEnvCfg(UGVRoverEmptyEnvCfg):
    """Second-stage task: static city-like delivery arena with roads and obstacles."""

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        visualizer_cfgs=KitVisualizerCfg(
            eye=(5.8, -6.2, 5.2),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            origin_env_index=0,
            visible_env_indices=[0],
            randomly_sample_visible_envs=False,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=8.0, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(5.8, -6.2, 5.2), lookat=(0.0, 0.0, 0.0))

    # Keep the observation/action sizes identical to the empty-space task so we
    # can warm-start from the preserved empty-space sim-to-real checkpoint.
    min_goal_distance_m = 1.25

    # Building blocks and parked cars. Leave street corridors open for delivery navigation.
    obstacle_rects = (
        (-2.75, 1.65, 0.90, 1.10),
        (-1.35, 1.70, 0.85, 1.00),
        (1.45, 1.60, 0.95, 1.15),
        (2.75, 1.55, 0.80, 1.00),
        (-2.60, -1.55, 0.95, 1.20),
        (-0.85, -1.65, 0.90, 1.00),
        (1.00, -1.55, 0.85, 1.10),
        (2.55, -1.45, 0.95, 1.15),
        (-2.15, 0.30, 0.62, 0.30),
        (2.00, -0.25, 0.62, 0.30),
        (0.25, 1.05, 0.32, 0.68),
        (-0.15, -1.00, 0.32, 0.68),
    )

    # City stage rewards: stronger clearance and termination pressure.
    rew_progress = 4.5
    rew_goal = 6.0
    rew_forward_velocity = 0.28
    rew_reverse_action = -0.35
    rew_yaw_rate = -0.02
    rew_unneeded_turn = -0.06
    rew_obstacle_margin = -0.60
    rew_terminated = -4.0

    enable_follow_camera = True
