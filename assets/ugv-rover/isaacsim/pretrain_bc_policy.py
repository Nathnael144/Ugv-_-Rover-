"""Behavior-cloning pretraining for the UGV teleop dataset.

This script intentionally uses only NumPy so it can run in Isaac Sim's bundled
Python even before PyTorch/Isaac Lab is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "datasets" / "teleop"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pretrained"

OBSERVATION_KEYS = (
    "x_m",
    "y_m",
    "sin_yaw",
    "cos_yaw",
    "vx_m_s",
    "vy_m_s",
    "yaw_rate_rad_s",
)
ACTION_KEYS = ("target_linear_m_s", "target_angular_rad_s")
ACTION_SCALE = np.array([1.30, 4.00], dtype=np.float32)


def _iter_csv_paths(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.glob("*/teleop.csv")):
        if path.stat().st_size > 200:
            yield path


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    observations: list[list[float]] = []
    actions: list[list[float]] = []
    sources: list[str] = []

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            yaw = float(row["yaw_rad"])
            obs = [
                float(row["x_m"]),
                float(row["y_m"]),
                math.sin(yaw),
                math.cos(yaw),
                float(row["vx_m_s"]),
                float(row["vy_m_s"]),
                float(row["yaw_rate_rad_s"]),
            ]
            action = [
                float(row["target_linear_m_s"]),
                float(row["target_angular_rad_s"]),
            ]
            observations.append(obs)
            actions.append(action)
            sources.append(str(path))

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        sources,
    )


def load_dataset(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    obs_parts = []
    action_parts = []
    source_parts = []
    for csv_path in _iter_csv_paths(dataset_root):
        obs, actions, sources = _load_csv(csv_path)
        if len(obs) == 0:
            continue
        obs_parts.append(obs)
        action_parts.append(actions)
        source_parts.extend(sources)

    if not obs_parts:
        raise RuntimeError(f"No non-empty teleop CSV files found under {dataset_root}")

    observations = np.concatenate(obs_parts, axis=0)
    actions = np.concatenate(action_parts, axis=0)
    return observations, actions, source_parts


def _init_params(rng: np.random.Generator, in_dim: int, hidden_dim: int, out_dim: int) -> dict[str, np.ndarray]:
    def weight(shape: tuple[int, int]) -> np.ndarray:
        fan_in = shape[0]
        return (rng.standard_normal(shape).astype(np.float32) * math.sqrt(2.0 / fan_in))

    return {
        "w1": weight((in_dim, hidden_dim)),
        "b1": np.zeros((hidden_dim,), dtype=np.float32),
        "w2": weight((hidden_dim, hidden_dim)),
        "b2": np.zeros((hidden_dim,), dtype=np.float32),
        "w3": weight((hidden_dim, out_dim)),
        "b3": np.zeros((out_dim,), dtype=np.float32),
    }


def _forward(params: dict[str, np.ndarray], obs: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    z1 = obs @ params["w1"] + params["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ params["w2"] + params["b2"]
    h2 = np.tanh(z2)
    z3 = h2 @ params["w3"] + params["b3"]
    pred = np.tanh(z3)
    return pred, (obs, z1, h1, z2, h2, z3, pred)


def _backward(
    params: dict[str, np.ndarray],
    cache: tuple[np.ndarray, ...],
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    obs, z1, h1, z2, h2, z3, pred = cache
    batch = max(1, obs.shape[0])
    grad_pred = (2.0 / batch) * (pred - target)
    grad_z3 = grad_pred * (1.0 - np.tanh(z3) ** 2)

    grad_w3 = h2.T @ grad_z3
    grad_b3 = grad_z3.sum(axis=0)
    grad_h2 = grad_z3 @ params["w3"].T
    grad_z2 = grad_h2 * (1.0 - np.tanh(z2) ** 2)

    grad_w2 = h1.T @ grad_z2
    grad_b2 = grad_z2.sum(axis=0)
    grad_h1 = grad_z2 @ params["w2"].T
    grad_z1 = grad_h1 * (1.0 - np.tanh(z1) ** 2)

    grad_w1 = obs.T @ grad_z1
    grad_b1 = grad_z1.sum(axis=0)

    return {
        "w1": grad_w1,
        "b1": grad_b1,
        "w2": grad_w2,
        "b2": grad_b2,
        "w3": grad_w3,
        "b3": grad_b3,
    }


def _adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    moments: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for key, grad in grads.items():
        moments[key] = beta1 * moments[key] + (1.0 - beta1) * grad
        velocities[key] = beta2 * velocities[key] + (1.0 - beta2) * (grad * grad)
        moment_hat = moments[key] / (1.0 - beta1**step)
        velocity_hat = velocities[key] / (1.0 - beta2**step)
        params[key] -= learning_rate * moment_hat / (np.sqrt(velocity_hat) + eps)


def train(
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(observations))
    rng.shuffle(indices)
    split = max(1, int(0.8 * len(indices)))
    train_idx = indices[:split]
    val_idx = indices[split:]

    obs_mean = observations[train_idx].mean(axis=0)
    obs_std = observations[train_idx].std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std).astype(np.float32)

    obs_norm = ((observations - obs_mean) / obs_std).astype(np.float32)
    actions_norm = np.clip(actions / ACTION_SCALE, -1.0, 1.0).astype(np.float32)

    params = _init_params(rng, obs_norm.shape[1], hidden_dim, actions_norm.shape[1])
    moments = {key: np.zeros_like(value) for key, value in params.items()}
    velocities = {key: np.zeros_like(value) for key, value in params.items()}

    step = 0
    for epoch in range(1, epochs + 1):
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start : start + batch_size]
            pred, cache = _forward(params, obs_norm[batch_idx])
            grads = _backward(params, cache, actions_norm[batch_idx])
            step += 1
            _adam_update(params, grads, moments, velocities, step, learning_rate)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            train_loss = mse(params, obs_norm[train_idx], actions_norm[train_idx])
            val_loss = mse(params, obs_norm[val_idx], actions_norm[val_idx]) if len(val_idx) else train_loss
            print(f"epoch={epoch:04d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    train_pred = predict(params, obs_norm[train_idx])
    val_pred = predict(params, obs_norm[val_idx]) if len(val_idx) else train_pred
    train_rmse = rmse_actions(train_pred, actions_norm[train_idx])
    val_rmse = rmse_actions(val_pred, actions_norm[val_idx]) if len(val_idx) else train_rmse

    metrics = {
        "samples": float(len(observations)),
        "train_samples": float(len(train_idx)),
        "validation_samples": float(len(val_idx)),
        "train_rmse_linear_m_s": float(train_rmse[0]),
        "train_rmse_angular_rad_s": float(train_rmse[1]),
        "validation_rmse_linear_m_s": float(val_rmse[0]),
        "validation_rmse_angular_rad_s": float(val_rmse[1]),
    }
    norm = {"obs_mean": obs_mean.astype(np.float32), "obs_std": obs_std}
    return params, metrics, norm


def mse(params: dict[str, np.ndarray], obs_norm: np.ndarray, actions_norm: np.ndarray) -> float:
    pred, _ = _forward(params, obs_norm)
    return float(np.mean((pred - actions_norm) ** 2))


def predict(params: dict[str, np.ndarray], obs_norm: np.ndarray) -> np.ndarray:
    pred, _ = _forward(params, obs_norm)
    return pred


def rmse_actions(pred_norm: np.ndarray, target_norm: np.ndarray) -> np.ndarray:
    pred = pred_norm * ACTION_SCALE
    target = target_norm * ACTION_SCALE
    return np.sqrt(np.mean((pred - target) ** 2, axis=0))


def save_outputs(
    output_dir: Path,
    params: dict[str, np.ndarray],
    metrics: dict[str, float],
    norm: dict[str, np.ndarray],
    dataset_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ugv_bc_policy.npz"
    np.savez(
        checkpoint_path,
        **params,
        obs_mean=norm["obs_mean"],
        obs_std=norm["obs_std"],
        action_scale=ACTION_SCALE,
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "observation_keys": list(OBSERVATION_KEYS),
        "action_keys": list(ACTION_KEYS),
        "action_scale": ACTION_SCALE.tolist(),
        "policy_type": "numpy_mlp_tanh",
        "hidden_layers": [int(params["b1"].shape[0]), int(params["b2"].shape[0])],
        "notes": (
            "Behavior cloning warm-start from teleoperation. Use this checkpoint "
            "to initialize or supervise the first RL policy before reward-only training."
        ),
    }
    (output_dir / "ugv_bc_policy_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ugv_bc_policy_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    rl_config = {
        "robot_stage": str(SCRIPT_DIR / "ugv_dataset_empty_env.usda"),
        "warm_start_checkpoint": str(checkpoint_path),
        "observation_keys": list(OBSERVATION_KEYS),
        "action_keys": list(ACTION_KEYS),
        "action_limits": {
            "linear_m_s": [-1.30, 1.30],
            "angular_rad_s": [-4.00, 4.00],
        },
        "recommended_next_step": "Collect more varied demos, then initialize the RL actor from this policy.",
    }
    (output_dir / "rl_warm_start_config.json").write_text(
        json.dumps(rl_config, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations, actions, sources = load_dataset(args.dataset_root)
    unique_sources = sorted(set(sources))
    print(f"loaded_samples={len(observations)} files={len(unique_sources)}")
    for source in unique_sources:
        print(f"source={source}")

    params, metrics, norm = train(
        observations=observations,
        actions=actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    save_outputs(args.output_dir, params, metrics, norm, args.dataset_root)
    print(f"saved_checkpoint={args.output_dir / 'ugv_bc_policy.npz'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
