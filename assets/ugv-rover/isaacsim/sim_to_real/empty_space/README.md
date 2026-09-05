# UGV Empty-Space Sim-to-Real Checkpoint

This folder preserves the final policy from the empty rectangular arena training run.
Use this checkpoint for empty-space sim-to-real tests before continuing with obstacle training.

## Preserved Policy

- Checkpoint: `ugv_empty_space_final_model_1399.pt`
- Source run: `IsaacLab/logs/rsl_rl/ugv_rover_empty/2026-09-05_16-47-49`
- Source checkpoint: `model_1399.pt`
- Training task: `Isaac-UGV-Rover-Empty-v0`
- Environments: `100`
- Arena: `7 m x 5 m` rectangular boundary
- Goal behavior: circular goal marker is resampled after the robot reaches it
- Perception used for this policy: proprioception, goal-relative state, wall-distance lidar-style rays
- Camera images: not used by this policy yet

## Integrity

- SHA256:
  `5b1cd80d84fd668e4642a1acb0f806d1bb1bbd9602773558c5ebc26d7963063d`

## Files

- `ugv_empty_space_final_model_1399.pt`: preserved empty-space policy checkpoint
- `agent_empty_space.yaml`: PPO/RSL-RL agent config used for this run
- `env_empty_space.yaml`: Isaac Lab environment config used for this run
- `run_empty_space.json`: run metadata

## Important

Do not overwrite this folder with obstacle-training checkpoints.
Obstacle training should start from this checkpoint but save into a different run/folder.
