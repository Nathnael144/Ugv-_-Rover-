#!/usr/bin/env bash
set -euo pipefail

ISAACSIM_DIR="/home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim"
ISAACLAB_DIR="${ISAACSIM_DIR}/IsaacLab"
VENV_DIR="${ISAACSIM_DIR}/env_isaaclab_isaacpy"
EMPTY_SIM_TO_REAL_CHECKPOINT="${ISAACSIM_DIR}/sim_to_real/empty_space/ugv_empty_space_final_model_1399.pt"
RUN_DIR="${ISAACLAB_DIR}/logs/rsl_rl/ugv_rover_city_static"

source "${VENV_DIR}/bin/activate"
export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1"
export WARP_CACHE_PATH="${ISAACSIM_DIR}/.warp-cache"
export MPLCONFIGDIR="${ISAACSIM_DIR}/.matplotlib-cache"
python -m pip install --no-build-isolation -e "${ISAACSIM_DIR}/ugv_isaaclab"

TRAIN_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [[ -z "${TRAIN_CHECKPOINT}" && -d "${RUN_DIR}" ]]; then
  TRAIN_CHECKPOINT="$(
    find "${RUN_DIR}" -maxdepth 2 -type f -name 'model_*.pt' -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' ' -f2-
  )"
fi
if [[ -z "${TRAIN_CHECKPOINT}" ]]; then
  TRAIN_CHECKPOINT="${EMPTY_SIM_TO_REAL_CHECKPOINT}"
fi

cd "${ISAACLAB_DIR}"
exec ./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-UGV-Rover-City-v0 \
  --external_callback ugv_isaaclab.register_tasks \
  --num_envs 100 \
  --viz kit \
  --video \
  --video_length 7200 \
  --video_interval 10000 \
  --checkpoint "${TRAIN_CHECKPOINT}"
