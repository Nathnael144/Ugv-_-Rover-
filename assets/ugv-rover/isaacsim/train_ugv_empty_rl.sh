#!/usr/bin/env bash
set -euo pipefail

ISAACSIM_DIR="/home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim"
ISAACLAB_DIR="${ISAACSIM_DIR}/IsaacLab"
VENV_DIR="${ISAACSIM_DIR}/env_isaaclab_isaacpy"
BC_CHECKPOINT="${ISAACSIM_DIR}/pretrained/ugv_isaaclab_bc_rsl_rl.pt"
RUN_DIR="${ISAACLAB_DIR}/logs/rsl_rl/ugv_rover_empty"

source "${VENV_DIR}/bin/activate"
export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1"
export WARP_CACHE_PATH="${ISAACSIM_DIR}/.warp-cache"
export MPLCONFIGDIR="${ISAACSIM_DIR}/.matplotlib-cache"
python -m pip install --no-build-isolation -e "${ISAACSIM_DIR}/ugv_isaaclab"

if [[ ! -f "${BC_CHECKPOINT}" ]]; then
  python "${ISAACSIM_DIR}/pretrain_isaaclab_bc_policy.py"
fi

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
  TRAIN_CHECKPOINT="${BC_CHECKPOINT}"
fi

cd "${ISAACLAB_DIR}"
exec ./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-UGV-Rover-Empty-v0 \
  --external_callback ugv_isaaclab.register_tasks \
  --num_envs 100 \
  --viz kit \
  --max_visible_envs 4 \
  --video \
  --video_length 3600 \
  --video_interval 0 \
  --checkpoint "${TRAIN_CHECKPOINT}"
