"""Inference helper for the NumPy UGV behavior-cloning policy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "pretrained" / "ugv_bc_policy.npz"


class UGVBCPolicy:
    """Small tanh MLP exported by `pretrain_bc_policy.py`."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> None:
        data = np.load(Path(checkpoint_path))
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]
        self.obs_mean = data["obs_mean"]
        self.obs_std = data["obs_std"]
        self.action_scale = data["action_scale"]

    def predict_from_array(self, observation: np.ndarray) -> np.ndarray:
        obs = (observation.astype(np.float32) - self.obs_mean) / self.obs_std
        h1 = np.tanh(obs @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        action_norm = np.tanh(h2 @ self.w3 + self.b3)
        return action_norm * self.action_scale

    def predict_from_state(self, state: Mapping[str, float]) -> tuple[float, float]:
        yaw = float(state["yaw_rad"])
        observation = np.array(
            [
                float(state["x_m"]),
                float(state["y_m"]),
                math.sin(yaw),
                math.cos(yaw),
                float(state["vx_m_s"]),
                float(state["vy_m_s"]),
                float(state["yaw_rate_rad_s"]),
            ],
            dtype=np.float32,
        )
        action = self.predict_from_array(observation)
        return float(action[0]), float(action[1])
