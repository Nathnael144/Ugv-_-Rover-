"""Export a behavior-cloning warm-start checkpoint for the Isaac Lab UGV task.

This trains an RSL-RL-compatible actor on the teleop CSV data using the same
30-value observation layout as ``Isaac-UGV-Rover-Empty-v0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "datasets" / "teleop"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pretrained"

ARENA_LENGTH_M = 7.0
ARENA_WIDTH_M = 5.0
LIDAR_NUM_RAYS = 16
LIDAR_MAX_RANGE_M = 4.0
MAX_LINEAR_M_S = 1.0
MAX_ANGULAR_RAD_S = 2.0
HIDDEN_DIMS = (128, 128)


def _iter_csv_paths(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.glob("*/teleop.csv")):
        if path.stat().st_size > 200:
            yield path


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _world_to_body_velocity(vx_w: torch.Tensor, vy_w: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    vx_b = cos_yaw * vx_w + sin_yaw * vy_w
    vy_b = -sin_yaw * vx_w + cos_yaw * vy_w
    return vx_b, vy_b


def _compute_lidar_ranges(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    half_length = ARENA_LENGTH_M * 0.5
    half_width = ARENA_WIDTH_M * 0.5
    ray_angles = torch.linspace(-math.pi, math.pi, LIDAR_NUM_RAYS + 1)[:-1]
    angles = yaw.unsqueeze(-1) + ray_angles.unsqueeze(0)
    dx = torch.cos(angles)
    dy = torch.sin(angles)
    eps = 1e-6

    tx_pos = (half_length - xy[:, 0:1]) / dx.clamp_min(eps)
    tx_neg = (-half_length - xy[:, 0:1]) / dx.clamp_max(-eps)
    ty_pos = (half_width - xy[:, 1:2]) / dy.clamp_min(eps)
    ty_neg = (-half_width - xy[:, 1:2]) / dy.clamp_max(-eps)
    tx = torch.where(dx > 0.0, tx_pos, tx_neg)
    ty = torch.where(dy > 0.0, ty_pos, ty_neg)
    return torch.minimum(tx, ty).clamp(0.0, LIDAR_MAX_RANGE_M)


def _load_one_csv(path: Path, goal_horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows.extend(reader)
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    x = torch.tensor([float(row["x_m"]) for row in rows], dtype=torch.float32)
    y = torch.tensor([float(row["y_m"]) for row in rows], dtype=torch.float32)
    yaw = torch.tensor([float(row["yaw_rad"]) for row in rows], dtype=torch.float32)
    vx_w = torch.tensor([float(row["vx_m_s"]) for row in rows], dtype=torch.float32)
    vy_w = torch.tensor([float(row["vy_m_s"]) for row in rows], dtype=torch.float32)
    yaw_rate = torch.tensor([float(row["yaw_rate_rad_s"]) for row in rows], dtype=torch.float32)
    linear = torch.tensor([float(row["target_linear_m_s"]) for row in rows], dtype=torch.float32)
    angular = torch.tensor([float(row["target_angular_rad_s"]) for row in rows], dtype=torch.float32)

    xy = torch.stack((x, y), dim=-1)
    future_indices = (torch.arange(len(rows)) + goal_horizon).clamp(max=len(rows) - 1)
    goal_xy = xy[future_indices]
    goal_delta = goal_xy - xy
    distance = torch.linalg.norm(goal_delta, dim=-1).clamp_min(1e-6)
    goal_direction = goal_delta / distance.unsqueeze(-1)
    goal_heading = torch.atan2(goal_delta[:, 1], goal_delta[:, 0])
    heading_error = _wrap_to_pi(goal_heading - yaw)
    vx_b, vy_b = _world_to_body_velocity(vx_w, vy_w, yaw)
    actions = torch.stack(
        (
            (linear / MAX_LINEAR_M_S).clamp(-1.0, 1.0),
            (angular / MAX_ANGULAR_RAD_S).clamp(-1.0, 1.0),
        ),
        dim=-1,
    )
    previous_actions = torch.zeros_like(actions)
    previous_actions[1:] = actions[:-1]
    lidar = _compute_lidar_ranges(xy, yaw)

    observations = torch.cat(
        (
            goal_delta / torch.tensor([ARENA_LENGTH_M * 0.5, ARENA_WIDTH_M * 0.5]),
            distance.unsqueeze(-1) / LIDAR_MAX_RANGE_M,
            torch.sin(heading_error).unsqueeze(-1),
            torch.cos(heading_error).unsqueeze(-1),
            goal_direction,
            torch.stack((vx_b / MAX_LINEAR_M_S, vy_b / MAX_LINEAR_M_S), dim=-1),
            (yaw_rate / MAX_ANGULAR_RAD_S).unsqueeze(-1),
            actions,
            previous_actions,
            lidar / LIDAR_MAX_RANGE_M,
        ),
        dim=-1,
    )
    return observations, actions


def load_dataset(dataset_root: Path, goal_horizon: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    obs_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    sources: list[str] = []
    for csv_path in _iter_csv_paths(dataset_root):
        obs, actions = _load_one_csv(csv_path, goal_horizon)
        obs_parts.append(obs)
        action_parts.append(actions)
        sources.append(str(csv_path))
    if not obs_parts:
        raise RuntimeError(f"No non-empty teleop CSV files found under {dataset_root}")
    return torch.cat(obs_parts, dim=0), torch.cat(action_parts, dim=0), sources


def _make_models(obs_mean: torch.Tensor, obs_std: torch.Tensor, sample_count: int) -> tuple[MLPModel, MLPModel]:
    obs = TensorDict({"policy": torch.zeros(1, 30)}, batch_size=[1])
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        2,
        hidden_dims=list(HIDDEN_DIMS),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"},
    )
    critic = MLPModel(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=list(HIDDEN_DIMS),
        activation="elu",
        obs_normalization=True,
    )
    for model in (actor, critic):
        model.obs_normalizer._mean[:] = obs_mean.unsqueeze(0)
        model.obs_normalizer._var[:] = obs_std.square().unsqueeze(0)
        model.obs_normalizer._std[:] = obs_std.unsqueeze(0)
        model.obs_normalizer.count.fill_(sample_count)
    return actor, critic


def train_bc(
    observations: torch.Tensor,
    actions: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[MLPModel, MLPModel, torch.optim.Optimizer, dict[str, float]]:
    torch.manual_seed(seed)
    indices = torch.randperm(len(observations))
    split = max(1, int(0.8 * len(indices)))
    train_idx = indices[:split]
    val_idx = indices[split:]

    obs_mean = observations[train_idx].mean(dim=0)
    obs_std = observations[train_idx].std(dim=0, unbiased=False).clamp_min(1e-4)
    actor, critic = _make_models(obs_mean, obs_std, len(train_idx))
    optimizer = torch.optim.Adam((*actor.parameters(), *critic.parameters()), lr=learning_rate)

    value_targets = -observations[:, 2:3].clamp(0.0, 1.0)
    td = TensorDict({"policy": observations}, batch_size=[len(observations)])

    for epoch in range(1, epochs + 1):
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start : start + batch_size]
            batch = TensorDict({"policy": observations[batch_idx]}, batch_size=[len(batch_idx)])
            pred_actions = actor(batch)
            pred_values = critic(batch)
            actor_loss = torch.nn.functional.mse_loss(pred_actions, actions[batch_idx])
            critic_loss = torch.nn.functional.mse_loss(pred_values, value_targets[batch_idx])
            loss = actor_loss + 0.2 * critic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((*actor.parameters(), *critic.parameters()), 1.0)
            optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            with torch.no_grad():
                train_loss = torch.nn.functional.mse_loss(actor(td[train_idx]), actions[train_idx]).item()
                val_loss = torch.nn.functional.mse_loss(actor(td[val_idx]), actions[val_idx]).item() if len(val_idx) else train_loss
            print(f"epoch={epoch:04d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    with torch.no_grad():
        train_pred = actor(td[train_idx])
        val_pred = actor(td[val_idx]) if len(val_idx) else train_pred
        action_scale = torch.tensor([MAX_LINEAR_M_S, MAX_ANGULAR_RAD_S])
        train_rmse = torch.sqrt(torch.mean(((train_pred - actions[train_idx]) * action_scale) ** 2, dim=0))
        val_rmse = torch.sqrt(torch.mean(((val_pred - actions[val_idx]) * action_scale) ** 2, dim=0)) if len(val_idx) else train_rmse

    metrics = {
        "samples": float(len(observations)),
        "train_samples": float(len(train_idx)),
        "validation_samples": float(len(val_idx)),
        "train_rmse_linear_m_s": float(train_rmse[0]),
        "train_rmse_angular_rad_s": float(train_rmse[1]),
        "validation_rmse_linear_m_s": float(val_rmse[0]),
        "validation_rmse_angular_rad_s": float(val_rmse[1]),
    }
    return actor, critic, optimizer, metrics


def save_outputs(
    output_dir: Path,
    actor: MLPModel,
    critic: MLPModel,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    dataset_root: Path,
    sources: list[str],
    goal_horizon: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ugv_isaaclab_bc_rsl_rl.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": 0,
            "infos": {
                "type": "behavior_cloning_warm_start",
                "dataset_root": str(dataset_root),
                "sources": sources,
                "observation_space": 30,
                "action_space": 2,
            },
        },
        checkpoint_path,
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "sources": sources,
        "goal_horizon_samples": goal_horizon,
        "observation_space": 30,
        "action_space": 2,
        "action_limits": {
            "linear_m_s": [-MAX_LINEAR_M_S, MAX_LINEAR_M_S],
            "angular_rad_s": [-MAX_ANGULAR_RAD_S, MAX_ANGULAR_RAD_S],
        },
        "policy_architecture": {
            "type": "rsl_rl.MLPModel",
            "hidden_dims": list(HIDDEN_DIMS),
            "activation": "elu",
            "actor_distribution": "GaussianDistribution(init_std=0.6)",
        },
        "notes": "RSL-RL checkpoint compatible with Isaac-UGV-Rover-Empty-v0 PPO --checkpoint.",
    }
    (output_dir / "ugv_isaaclab_bc_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "ugv_isaaclab_bc_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved_checkpoint={checkpoint_path}")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--goal-horizon", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations, actions, sources = load_dataset(args.dataset_root, args.goal_horizon)
    print(f"loaded_samples={len(observations)} files={len(sources)} obs_dim={observations.shape[1]}")
    for source in sources:
        print(f"source={source}")
    actor, critic, optimizer, metrics = train_bc(
        observations=observations,
        actions=actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    save_outputs(args.output_dir, actor, critic, optimizer, metrics, args.dataset_root, sources, args.goal_horizon)


if __name__ == "__main__":
    main()
