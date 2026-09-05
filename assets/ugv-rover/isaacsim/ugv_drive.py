"""Smooth differential-drive wheel controller for the open UGV physics stage.

Run this file from Isaac Sim's Script Editor after opening
ugv_rover_physics.usda. Change LINEAR_M_S and ANGULAR_RAD_S as needed.
"""

import math
from pathlib import Path
import sys
import types

import omni.kit.app
import omni.timeline
import omni.usd


STAGE_PATH = Path(__file__).with_name("ugv_rover_physics.usda")
LINEAR_M_S = 1.00
ANGULAR_RAD_S = 10.00
SHOW_COLLISION_SHAPES = True

WHEEL_RADIUS_M = 0.040
TRACK_WIDTH_M = 0.17462
MAX_LINEAR_M_S = 1.30
MAX_ANGULAR_RAD_S = 10.0
LINEAR_ACCEL_M_S2 = 1.20
ANGULAR_ACCEL_RAD_S2 = 3.50

JOINTS = {
    "left": (
        "/World/UGV_Rover/Joints/front_left",
        "/World/UGV_Rover/Joints/rear_left",
    ),
    "right": (
        "/World/UGV_Rover/Joints/front_right",
        "/World/UGV_Rover/Joints/rear_right",
    ),
}
WHEEL_COLLISIONS = (
    "/World/UGV_Rover/wheel_front_left/Collision",
    "/World/UGV_Rover/wheel_middle_left/Collision",
    "/World/UGV_Rover/wheel_rear_left/Collision",
    "/World/UGV_Rover/wheel_front_right/Collision",
    "/World/UGV_Rover/wheel_middle_right/Collision",
    "/World/UGV_Rover/wheel_rear_right/Collision",
)

_registry = sys.modules.setdefault("_ugv_rover_drive_state", types.SimpleNamespace())
if not hasattr(_registry, "subscription"):
    _registry.subscription = None
_state = {
    "linear_m_s": 0.0,
    "angular_rad_s": 0.0,
    "ready": False,
    "elapsed_s": 0.0,
    "last_report_s": 0.0,
    "stage_open_requested": False,
}


def _ensure_stage_open() -> None:
    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage and stage.GetPrimAtPath("/World/UGV_Rover/Joints/front_left").IsValid():
        return
    if _state["stage_open_requested"]:
        return
    _state["stage_open_requested"] = True
    context.open_stage(str(STAGE_PATH))


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _step_toward(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


def _apply_velocity(linear_m_s: float, angular_rad_s: float) -> None:
    """Apply body-frame forward and yaw velocity commands to wheel drives."""
    linear_m_s = _clamp(linear_m_s, MAX_LINEAR_M_S)
    angular_rad_s = _clamp(angular_rad_s, MAX_ANGULAR_RAD_S)
    left_m_s = linear_m_s - angular_rad_s * TRACK_WIDTH_M / 2.0
    right_m_s = linear_m_s + angular_rad_s * TRACK_WIDTH_M / 2.0

    # USD angular drive targets are authored in degrees per second.
    targets = {
        "left": -math.degrees(left_m_s / WHEEL_RADIUS_M),
        "right": -math.degrees(right_m_s / WHEEL_RADIUS_M),
    }

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    connected = 0
    commanded = []
    for side, joint_paths in JOINTS.items():
        for joint_path in joint_paths:
            joint = stage.GetPrimAtPath(joint_path)
            if not joint.IsValid():
                print(f"Waiting for wheel joint: {joint_path}")
                continue
            joint.GetAttribute("drive:angular:physics:targetVelocity").Set(targets[side])
            connected += 1
            commanded.append(f"{joint_path.rsplit('/', 1)[-1]}={targets[side]:.1f} deg/s")

    if connected != sum(len(paths) for paths in JOINTS.values()):
        return

    if not _state["ready"]:
        _state["ready"] = True
        print("UGV drive connected to wheel joints.")
    if _state["elapsed_s"] - _state["last_report_s"] >= 1.0:
        _state["last_report_s"] = _state["elapsed_s"]
        print("UGV wheel targets: " + ", ".join(commanded))


def _set_collision_visibility() -> None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    visibility = "inherited" if SHOW_COLLISION_SHAPES else "invisible"
    for collision_path in WHEEL_COLLISIONS:
        collision = stage.GetPrimAtPath(collision_path)
        if collision.IsValid():
            collision.GetAttribute("visibility").Set(visibility)


def set_velocity(linear_m_s: float, angular_rad_s: float) -> None:
    """Set a new smooth target command while the update loop is running."""
    global LINEAR_M_S, ANGULAR_RAD_S
    LINEAR_M_S = _clamp(linear_m_s, MAX_LINEAR_M_S)
    ANGULAR_RAD_S = _clamp(angular_rad_s, MAX_ANGULAR_RAD_S)


def _on_update(event) -> None:
    dt = event.payload.get("dt", 1.0 / 60.0) if hasattr(event, "payload") else 1.0 / 60.0
    _state["elapsed_s"] += dt
    if _state["elapsed_s"] > 1.0:
        _ensure_stage_open()
    _state["linear_m_s"] = _step_toward(
        _state["linear_m_s"], LINEAR_M_S, LINEAR_ACCEL_M_S2 * dt
    )
    _state["angular_rad_s"] = _step_toward(
        _state["angular_rad_s"], ANGULAR_RAD_S, ANGULAR_ACCEL_RAD_S2 * dt
    )
    _set_collision_visibility()
    _apply_velocity(_state["linear_m_s"], _state["angular_rad_s"])


if _registry.subscription is not None:
    # Re-running this script in Isaac Sim replaces the old controller loop.
    _registry.subscription.unsubscribe()

_registry.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _on_update,
    name="ugv_rover_smooth_drive",
)

_set_collision_visibility()
_apply_velocity(0.0, 0.0)
omni.timeline.get_timeline_interface().play()
print(
    "UGV smooth command: "
    f"target_linear={LINEAR_M_S:.3f} m/s, target_angular={ANGULAR_RAD_S:.3f} rad/s, "
    f"linear_accel={LINEAR_ACCEL_M_S2:.2f} m/s^2"
)
