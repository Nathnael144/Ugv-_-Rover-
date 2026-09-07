# UGV Rover Isaac Sim RL Project

This repository adapts the Waveshare UGV Rover platform for Isaac Sim and Isaac Lab reinforcement learning. The goal is to train a delivery-style ground robot in simulation, starting from a simple rectangular arena and gradually moving toward realistic city navigation before transferring the learned policy to real hardware.

The project keeps the real CAD model as the visual source of truth and adds physics, wheel joints, motor control, dataset collection, behavior-cloning pretraining, and PPO reinforcement learning around it.

## Training Concept

The training pipeline follows a staged curriculum:

1. Build a physically correct rover model from the CAD/USD asset.
2. Collect teleoperation data in a simple environment.
3. Pretrain a behavior-cloning policy from the dataset.
4. Warm-start PPO from the pretrained policy instead of learning from scratch.
5. Train first in an empty 7 x 5 m rectangular arena.
6. Preserve the empty-space checkpoint for sim-to-real testing.
7. Continue training in a static delivery city environment with asphalt roads, lane markings, sidewalks, buildings, parked cars, and boundary walls.
8. Later add higher-fidelity camera and LiDAR observations for perception-heavy training.

This staged approach is intentional. The robot first learns stable motion, goal seeking, and collision avoidance from compact state observations. Full RTX cameras and real LiDAR can then be added after the control policy is already reliable.

## Robot Model

The Isaac Sim assets are in:

```bash
assets/ugv-rover/isaacsim
```

Key files:

```bash
assets/ugv-rover/isaacsim/UGV_Rover_PT_AI_Kit.usd
assets/ugv-rover/isaacsim/ugv_rover_physics.usda
assets/ugv-rover/isaacsim/ugv_rover_physics_rl.usda
assets/ugv-rover/isaacsim/real_world_physics.yaml
```

The rover uses real-world-inspired mass, inertia, friction, and drive limits so the policy has a better chance of transferring to hardware. Four wheels are motor-driven: front-left, rear-left, front-right, and rear-right. The middle wheels are passive and rotate through contact/friction, matching the intended physical behavior.

## Environments

Two Isaac Lab tasks are registered:

```text
Isaac-UGV-Rover-Empty-v0
Isaac-UGV-Rover-City-v0
```

The empty environment is a 7 x 5 m rectangular arena with a circular goal marker. When the rover reaches the goal, a new goal is sampled so training continues without restarting the whole run.

The city environment keeps the same policy shape but adds static obstacles and delivery-road visuals: black asphalt, lane markings, sidewalks, building-like blocks, parked cars, and boundary walls. The current city stage is 10 x 7 m with the same obstacle layout and a longer LiDAR-like range. This lets the policy continue from the empty-space checkpoint while learning obstacle clearance in a wider delivery space.

## RL Formulation

Actions are continuous differential-drive commands:

```text
[linear_velocity, angular_velocity]
```

The normalized action range is `[-1, 1]`. In the current task configuration this maps to:

```text
linear velocity:  up to 1.0 m/s
angular velocity: up to 2.0 rad/s
```

Observations are a compact 30-dimensional vector:

```text
goal delta x/y
goal distance
sin/cos heading error
goal direction x/y
body linear velocity x/y
body yaw angular velocity
current action
previous action
16 planar LiDAR-like range readings
```

The reward encourages:

```text
staying alive
making progress toward the goal
facing the goal
reaching the goal
smooth action changes
reasonable action size
staying away from walls
staying away from obstacles
avoiding termination
```

The city task increases the goal reward and obstacle penalty so the rover learns delivery navigation without cutting through static objects.

## Reward And Penalty Values

The empty task and city task use the same reward structure. The city task overrides several values to make delivery navigation stricter around obstacles.

| Term | Empty value | City value | Purpose |
| --- | ---: | ---: | --- |
| `rew_alive` | `0.02` | `0.02` | Small reward for staying active each step. |
| `rew_progress` | `4.0` | `4.5` | Reward for reducing distance to the goal. |
| `rew_goal` | `5.0` | `6.0` | Bonus when the rover reaches the circular goal marker. |
| `rew_heading` | `0.15` | `0.15` | Reward for pointing the robot front toward the goal. |
| `rew_forward_velocity` | `0.20` | `0.28` | Encourages forward motion as the normal driving mode. |
| `rew_reverse_action` | `-0.25` | `-0.35` | Penalizes reverse so it is not the default strategy. |
| `rew_yaw_rate` | `-0.015` | `-0.02` | Penalizes excessive spinning or sharp yaw motion. |
| `rew_unneeded_turn` | `-0.04` | `-0.06` | Penalizes turning when already facing the goal. |
| `rew_action_rate` | `-0.03` | `-0.03` | Penalizes sudden action changes for smoother driving. |
| `rew_action_mag` | `-0.01` | `-0.01` | Penalizes unnecessarily large commands. |
| `rew_wall_margin` | `-0.25` | `-0.25` | Penalizes getting too close to boundary walls. |
| `rew_obstacle_margin` | `-0.35` | `-0.60` | Penalizes getting too close to city obstacles. |
| `rew_terminated` | `-3.0` | `-4.0` | Penalty for collision, leaving the arena, or tipping. |

Reverse is still allowed. The policy action can command negative linear velocity, but reverse receives a penalty. Near an obstacle, the reverse penalty is reduced so the robot can back out when that is the safest maneuver. This should teach: drive forward and turn smoothly when possible, reverse only when useful for obstacle avoidance.

Current city geometry and perception values:

| Setting | Value |
| --- | ---: |
| City arena length | `10.0 m` |
| City arena width | `7.0 m` |
| Empty arena length | `7.0 m` |
| Empty arena width | `5.0 m` |
| LiDAR-like rays | `16` |
| Empty LiDAR-like range | `4.0 m` |
| City LiDAR-like range | `6.0 m` |
| Goal reached radius | `0.25 m` |
| Obstacle safety margin | `0.28 m` |
| Max linear speed | `1.0 m/s` |
| Max angular speed | `2.0 rad/s` |

## Pretraining

Teleoperation datasets are used for behavior cloning before PPO training. The pretrained policy is stored here:

```bash
assets/ugv-rover/isaacsim/pretrained/ugv_isaaclab_bc_rsl_rl.pt
```

The preserved empty-space sim-to-real checkpoint is stored here:

```bash
assets/ugv-rover/isaacsim/sim_to_real/empty_space/ugv_empty_space_final_model_1399.pt
```

That checkpoint should stay preserved as the baseline policy for real-hardware tests in empty space. Later city training can continue from it, but should not replace it unless a new sim-to-real baseline is intentionally selected.

## Training Commands

Install or refresh the local Isaac Lab task package:

```bash
cd /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/IsaacLab
source ../env_isaaclab_isaacpy/bin/activate
python -m pip install --no-build-isolation -e ../ugv_isaaclab
```

Train the empty arena task:

```bash
/home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/train_ugv_empty_rl.sh
```

Train the city delivery task:

```bash
/home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/train_ugv_city_rl.sh
```

The city launcher uses 100 parallel environments, opens the Isaac/Kit viewer, resumes from the latest city checkpoint if one exists, otherwise starts from the preserved empty-space checkpoint, and records training video every 10000 frames.

## Video And Visualization

Training video is configured to focus on one visible environment. The camera follows the rover and goal area with a slow orbit plus gentle zoom so the robot motion and path are easier to inspect.

Current city video settings:

```text
video length:    7200 frames
video interval:  10000 frames
visible env:     environment 0
camera behavior: follow, orbit, and zoom
```

## Perception Roadmap

The current RL task does not yet train directly from raw camera images or full RTX LiDAR. It uses a lightweight planar LiDAR-style range vector so the first policy can learn fast and remain closer to what can be transferred to real hardware.

Next perception stages:

```text
add real LiDAR-style scan observations
add front camera features
add top camera features
add domain randomization for lighting, friction, mass, and obstacle placement
compare low-dimensional policy against vision/LiDAR policy
test the preserved empty-space checkpoint on real hardware
```

## Training Decision Log

These notes record the important project decisions so future training changes have context.

| Stage | Decision | Reason |
| --- | --- | --- |
| CAD import | Keep the original CAD/USD visual model intact. | The visible robot should match the real hardware; physics is added around the CAD rather than replacing it. |
| Wheel physics | Use four motor-driven wheels and passive middle wheels. | The real rover drives front/rear wheels; middle wheels should rotate through contact and friction. |
| Front direction | Keep the corrected robot front/camera direction from the working setup. | Policy observations and heading reward must match the real robot front for sim-to-real transfer. |
| Teleoperation dataset | Collect simple-environment driving data before RL. | Behavior cloning gives PPO a warmer start than random exploration. |
| BC pretraining | Use the same observation/action format as the Isaac Lab task. | Matching shapes lets the pretrained policy load into PPO cleanly. |
| Empty arena | Train first in a 7 x 5 m rectangular arena. | This isolates basic drive, turning, goal seeking, and wall avoidance before adding city complexity. |
| Empty checkpoint | Preserve the empty-space checkpoint separately. | It remains a sim-to-real baseline for hardware tests in a simple open space. |
| City arena | Add static city-like obstacles, asphalt, sidewalks, lane markings, buildings, and parked cars. | Delivery robots need obstacle-aware navigation in structured outdoor spaces. |
| Reward shaping | Add forward-motion and smooth-turn preference while keeping reverse available. | The robot was using reverse too often; reverse should remain an escape behavior, not the main strategy. |
| Wider city | Expand the city training area to 10 x 7 m. | A wider arena gives more realistic delivery paths and avoids overfitting to a small box. |
| LiDAR range | Increase city LiDAR-like range from 4 m to 6 m without changing ray count. | Longer range helps in the bigger arena while preserving the 30-D observation size for checkpoint compatibility. |
| Camera sensors | Delay raw RTX camera training. | First stabilize locomotion/control; then add high-dimensional perception after the policy is reliable. |
| Recording | Record focused training videos every 10000 frames. | Videos make it easier to diagnose path quality, reverse behavior, collisions, and goal reaching. |

## Original Raspberry Pi Robot Code

This repository also contains the Waveshare Raspberry Pi control stack for the physical UGV Rover, including Flask control UI, camera streaming, pan-tilt control, OpenCV examples, and tutorial notebooks. That code is useful for the final hardware bridge after simulation policies are ready.

## License

The original Waveshare Raspberry Pi code is licensed under the GNU General Public License v3.0. See `LICENSE` for details.
