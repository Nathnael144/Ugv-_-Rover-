# UGV Isaac Lab RL

This package registers rover RL tasks:

- `Isaac-UGV-Rover-Empty-v0`
- Environment: 7 x 5 m rectangular arena
- Action: `[linear_m_s, yaw_rate_rad_s]` normalized to `[-1, 1]`
- Observation: goal vector, heading error, base velocity, previous actions, and 16 planar LiDAR-like wall ranges
- Algorithm: PPO via `rsl_rl`
- `Isaac-UGV-Rover-City-v0`
- Environment: 7 x 5 m static delivery city block with asphalt, lane markings, buildings, parked cars, sidewalks, and boundary walls
- Observation: same policy shape as the empty task, but LiDAR-like ranges also include static city obstacles
- Warm start: use `sim_to_real/empty_space/ugv_empty_space_final_model_1399.pt`

The first stage intentionally avoids raw image observations. The fixed front cameras, top camera, and real LiDAR should be added after the low-dimensional policy learns stable navigation, otherwise the first RL run spends most of its effort solving perception instead of control.

## Install

```bash
source /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/env_isaaclab_isaacpy/bin/activate
python -m pip install --no-build-isolation -e /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/ugv_isaaclab
```

## Smoke Test

```bash
cd /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/IsaacLab
source ../env_isaaclab_isaacpy/bin/activate
python -c "import ugv_isaaclab, gymnasium as gym; print(gym.spec('Isaac-UGV-Rover-Empty-v0'))"
```

## Train

```bash
cd /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/IsaacLab
source ../env_isaaclab_isaacpy/bin/activate
./isaaclab.sh -p -m isaaclab_rl.entrypoints.backends.train_rsl_rl --task Isaac-UGV-Rover-Empty-v0 --num_envs 16 --headless
```

For debugging, remove `--headless` and reduce `--num_envs 1`.

## City Training

```bash
/home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/train_ugv_city_rl.sh
```

The city launcher uses 100 environments, opens the Kit viewer, starts from the preserved empty-space policy unless a city checkpoint already exists, and records a 4-minute video every 10000 frames.
