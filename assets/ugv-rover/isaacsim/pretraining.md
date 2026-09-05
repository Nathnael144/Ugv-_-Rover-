# UGV Behavior-Cloning Pretraining

This folder now supports a first warm-start policy for later RL training.
The policy is trained from teleoperation CSV files under `datasets/teleop`.

## Train

Use Isaac Sim's Python because it already includes NumPy:

```bash
/home/nathan/isaacsim/python.sh /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/pretrain_bc_policy.py
```

Outputs are written to:

```text
assets/ugv-rover/isaacsim/pretrained/
```

Important files:

- `ugv_bc_policy.npz`: NumPy MLP checkpoint
- `ugv_bc_policy_metadata.json`: observation/action schema
- `ugv_bc_policy_metrics.json`: train/validation errors
- `rl_warm_start_config.json`: handoff config for RL actor initialization

## Observation And Action Schema

Observations:

- `x_m`
- `y_m`
- `sin_yaw`
- `cos_yaw`
- `vx_m_s`
- `vy_m_s`
- `yaw_rate_rad_s`

Actions:

- `target_linear_m_s`
- `target_angular_rad_s`

This is a behavior-cloning warm start, not the final RL policy. It gives RL a
non-random initial driving behavior. Next we should collect more diverse demos
and then either convert this MLP into a PyTorch actor or use it as an auxiliary
imitation loss during Isaac Lab training.
