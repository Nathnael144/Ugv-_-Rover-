"""Keyboard teleoperation and dataset recorder for the empty UGV Isaac Sim env.

Run from Isaac Sim with:
    /home/nathan/isaacsim/isaac-sim.sh --exec /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/ugv_teleop_dataset.py

Controls:
    Hold W/S: forward/back
    Hold A/D: turn left/right
    Space: stop
    R: start/stop recording

The recorder writes CSV rows that are useful for behavior cloning pretraining:
timestamp, dt, command action, robot pose, and finite-difference body velocity.
"""

import csv
import atexit
import json
import math
from pathlib import Path
import sys
import time
import types

import carb
import carb.input
import omni.appwindow
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import UsdGeom


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_PATH = SCRIPT_DIR / "ugv_dataset_empty_env.usda"
DATASET_ROOT = SCRIPT_DIR / "datasets" / "teleop"

TELEOP_LINEAR_M_S = 1.00
TELEOP_ANGULAR_RAD_S = 2.00
MAX_LINEAR_M_S = 1.30
MAX_ANGULAR_RAD_S = 4.0
LINEAR_ACCEL_M_S2 = 1.20
ANGULAR_ACCEL_RAD_S2 = 3.50

WHEEL_RADIUS_M = 0.040
TRACK_WIDTH_M = 0.17462
CHASSIS_PATH = "/World/UGV_Rover/Chassis"
FRONT_LEFT_JOINT = "/World/UGV_Rover/Joints/front_left"

DRIVEN_JOINTS = {
    "left": (
        "/World/UGV_Rover/Joints/front_left",
        "/World/UGV_Rover/Joints/rear_left",
    ),
    "right": (
        "/World/UGV_Rover/Joints/front_right",
        "/World/UGV_Rover/Joints/rear_right",
    ),
}

_registry = sys.modules.setdefault("_ugv_teleop_dataset_state", types.SimpleNamespace())
if not hasattr(_registry, "update_subscription"):
    _registry.update_subscription = None
if not hasattr(_registry, "keyboard_subscription"):
    _registry.keyboard_subscription = None
if not hasattr(_registry, "keyboard"):
    _registry.keyboard = None
if not hasattr(_registry, "input_interface"):
    _registry.input_interface = None
if not hasattr(_registry, "control_window"):
    _registry.control_window = None

_state = {
    "target_linear_m_s": 0.0,
    "target_angular_rad_s": 0.0,
    "linear_m_s": 0.0,
    "angular_rad_s": 0.0,
    "elapsed_s": 0.0,
    "stage_open_requested": False,
    "ready": False,
    "recording": False,
    "record_dir": None,
    "csv_file": None,
    "csv_writer": None,
    "rows": 0,
    "last_pose": None,
    "last_time_s": None,
    "keyboard_drive_active": False,
    "last_keyboard_retry_s": -10.0,
    "last_keyboard_error": "",
    "last_target_report": None,
    "last_record_report_s": 0.0,
    "last_pose_wait_report_s": 0.0,
    "recording_started_s": None,
}


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _step_toward(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


def _ensure_stage_open() -> None:
    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage and stage.GetPrimAtPath(FRONT_LEFT_JOINT).IsValid():
        return
    if _state["stage_open_requested"]:
        return
    _state["stage_open_requested"] = True
    context.open_stage(str(STAGE_PATH))


def _get_stage():
    return omni.usd.get_context().get_stage()


def _set_target(
    linear_m_s: float | None = None,
    angular_rad_s: float | None = None,
    report: bool = True,
) -> None:
    if linear_m_s is not None:
        _state["target_linear_m_s"] = _clamp(linear_m_s, MAX_LINEAR_M_S)
    if angular_rad_s is not None:
        _state["target_angular_rad_s"] = _clamp(angular_rad_s, MAX_ANGULAR_RAD_S)

    target_report = (
        round(_state["target_linear_m_s"], 3),
        round(_state["target_angular_rad_s"], 3),
    )
    if report and target_report != _state["last_target_report"]:
        _state["last_target_report"] = target_report
        print(
            "Teleop target: "
            f"linear={_state['target_linear_m_s']:.2f} m/s, "
            f"angular={_state['target_angular_rad_s']:.2f} rad/s"
        )


def _stop_robot() -> None:
    _state["keyboard_drive_active"] = False
    _set_target(0.0, 0.0)


def _wheel_targets_deg_s(linear_m_s: float, angular_rad_s: float) -> dict[str, float]:
    left_m_s = linear_m_s - angular_rad_s * TRACK_WIDTH_M / 2.0
    right_m_s = linear_m_s + angular_rad_s * TRACK_WIDTH_M / 2.0
    return {
        "left": -math.degrees(left_m_s / WHEEL_RADIUS_M),
        "right": -math.degrees(right_m_s / WHEEL_RADIUS_M),
    }


def _apply_drive(linear_m_s: float, angular_rad_s: float) -> bool:
    stage = _get_stage()
    if stage is None:
        return False

    targets = _wheel_targets_deg_s(linear_m_s, angular_rad_s)
    connected = 0
    for side, joint_paths in DRIVEN_JOINTS.items():
        for joint_path in joint_paths:
            joint = stage.GetPrimAtPath(joint_path)
            if not joint.IsValid():
                return False
            joint.GetAttribute("drive:angular:physics:targetVelocity").Set(targets[side])
            connected += 1
    return connected == 4


def _pose_from_stage():
    stage = _get_stage()
    if stage is None:
        return None

    chassis = stage.GetPrimAtPath(CHASSIS_PATH)
    if not chassis.IsValid():
        return None

    transform = UsdGeom.Xformable(chassis).ComputeLocalToWorldTransform(0.0)
    translation = transform.ExtractTranslation()
    quat = transform.ExtractRotationQuat()
    return {
        "x": float(translation[0]),
        "y": float(translation[1]),
        "z": float(translation[2]),
        "qw": float(quat.GetReal()),
        "qx": float(quat.GetImaginary()[0]),
        "qy": float(quat.GetImaginary()[1]),
        "qz": float(quat.GetImaginary()[2]),
    }


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_delta(current: float, previous: float) -> float:
    delta = current - previous
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _begin_recording() -> None:
    if _state["recording"]:
        return

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    record_dir = DATASET_ROOT / time.strftime("%Y%m%d_%H%M%S")
    record_dir.mkdir(parents=True, exist_ok=False)

    csv_path = record_dir / "teleop.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "time_s",
            "dt_s",
            "target_linear_m_s",
            "target_angular_rad_s",
            "applied_linear_m_s",
            "applied_angular_rad_s",
            "x_m",
            "y_m",
            "z_m",
            "qw",
            "qx",
            "qy",
            "qz",
            "vx_m_s",
            "vy_m_s",
            "vz_m_s",
            "yaw_rad",
            "yaw_rate_rad_s",
        ],
    )
    writer.writeheader()
    csv_file.flush()

    metadata = {
        "stage_path": str(STAGE_PATH),
        "environment": "empty_ground_plane",
        "created_unix_time_s": time.time(),
        "robot": "Waveshare UGV Rover PT AI Kit",
        "control": "keyboard_teleoperation",
        "driven_wheels": ["front_left", "rear_left", "front_right", "rear_right"],
        "passive_wheels": ["middle_left", "middle_right"],
        "units": "SI",
    }
    (record_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    _state.update(
        {
            "recording": True,
            "record_dir": record_dir,
            "csv_file": csv_file,
            "csv_writer": writer,
            "rows": 0,
            "last_pose": None,
            "last_time_s": None,
            "recording_started_s": _state["elapsed_s"],
            "last_record_report_s": _state["elapsed_s"],
            "last_pose_wait_report_s": _state["elapsed_s"],
        }
    )
    print(f"Recording started: {csv_path}")


def _end_recording() -> None:
    if not _state["recording"]:
        return
    _state["csv_file"].flush()
    _state["csv_file"].close()
    print(f"Recording stopped: {_state['rows']} rows in {_state['record_dir']}")
    if _state["rows"] == 0:
        print("Warning: recording has no samples. Wait until the robot is visible before pressing R.")
    _state.update(
        {
            "recording": False,
            "record_dir": None,
            "csv_file": None,
            "csv_writer": None,
            "last_pose": None,
            "last_time_s": None,
            "recording_started_s": None,
        }
    )


def _toggle_recording() -> None:
    if _state["recording"]:
        _end_recording()
    else:
        _begin_recording()


def _record_sample(dt: float) -> None:
    if not _state["recording"]:
        return

    pose = _pose_from_stage()
    if pose is None:
        if _state["elapsed_s"] - _state["last_pose_wait_report_s"] >= 1.0:
            _state["last_pose_wait_report_s"] = _state["elapsed_s"]
            print("Recording waiting for robot pose: /World/UGV_Rover/Chassis")
        return

    time_s = _state["elapsed_s"]
    last_pose = _state["last_pose"]
    last_time_s = _state["last_time_s"]
    vx = vy = vz = yaw_rate = 0.0

    yaw = _yaw_from_quat(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
    if last_pose is not None and last_time_s is not None:
        sample_dt = max(1e-6, time_s - last_time_s)
        vx = (pose["x"] - last_pose["x"]) / sample_dt
        vy = (pose["y"] - last_pose["y"]) / sample_dt
        vz = (pose["z"] - last_pose["z"]) / sample_dt
        last_yaw = _yaw_from_quat(
            last_pose["qw"],
            last_pose["qx"],
            last_pose["qy"],
            last_pose["qz"],
        )
        yaw_rate = _angle_delta(yaw, last_yaw) / sample_dt

    _state["csv_writer"].writerow(
        {
            "time_s": f"{time_s:.6f}",
            "dt_s": f"{dt:.6f}",
            "target_linear_m_s": f"{_state['target_linear_m_s']:.6f}",
            "target_angular_rad_s": f"{_state['target_angular_rad_s']:.6f}",
            "applied_linear_m_s": f"{_state['linear_m_s']:.6f}",
            "applied_angular_rad_s": f"{_state['angular_rad_s']:.6f}",
            "x_m": f"{pose['x']:.6f}",
            "y_m": f"{pose['y']:.6f}",
            "z_m": f"{pose['z']:.6f}",
            "qw": f"{pose['qw']:.9f}",
            "qx": f"{pose['qx']:.9f}",
            "qy": f"{pose['qy']:.9f}",
            "qz": f"{pose['qz']:.9f}",
            "vx_m_s": f"{vx:.6f}",
            "vy_m_s": f"{vy:.6f}",
            "vz_m_s": f"{vz:.6f}",
            "yaw_rad": f"{yaw:.6f}",
            "yaw_rate_rad_s": f"{yaw_rate:.6f}",
        }
    )
    _state["rows"] += 1
    if _state["rows"] % 30 == 0:
        _state["csv_file"].flush()
        if time_s - _state["last_record_report_s"] >= 1.0:
            _state["last_record_report_s"] = time_s
            print(f"Recording rows: {_state['rows']}")
    _state["last_pose"] = pose
    _state["last_time_s"] = time_s


def _keyboard_input_name(key) -> str:
    return getattr(key, "name", str(key)).upper()


def _on_keyboard_event(event) -> bool:
    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return False

    key_name = _keyboard_input_name(event.input)
    if event.input == carb.input.KeyboardInput.SPACE or key_name == "SPACE":
        _stop_robot()
        return True
    if event.input == carb.input.KeyboardInput.R or key_name == "R":
        _toggle_recording()
        return True
    return False


def _remove_keyboard_handler() -> None:
    if _registry.keyboard_subscription is None:
        return
    try:
        _registry.input_interface.unsubscribe_to_keyboard_events(
            _registry.keyboard,
            _registry.keyboard_subscription,
        )
    except Exception as exc:
        print(f"Keyboard unsubscribe warning: {exc}")
    _registry.keyboard_subscription = None


def _install_keyboard_handler() -> bool:
    if _registry.keyboard_subscription is not None:
        return True

    try:
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            return False
        app_window.focus()
        keyboard = app_window.get_keyboard()
        if keyboard is None:
            return False
        input_interface = carb.input.acquire_input_interface()
        _registry.keyboard = keyboard
        _registry.input_interface = input_interface
        _registry.keyboard_subscription = input_interface.subscribe_to_keyboard_events(
            keyboard,
            _on_keyboard_event,
        )
        print("Keyboard teleop connected. Click the viewport, then hold W/S/A/D.")
        return True
    except Exception as exc:
        message = str(exc)
        if message != _state["last_keyboard_error"]:
            _state["last_keyboard_error"] = message
            print(f"Waiting for keyboard teleop: {message}")
        return False


def _keyboard_is_pressed(key: carb.input.KeyboardInput) -> bool:
    if _registry.input_interface is None or _registry.keyboard is None:
        return False
    flags = _registry.input_interface.get_keyboard_button_flags(_registry.keyboard, key)
    value = _registry.input_interface.get_keyboard_value(_registry.keyboard, key)
    return bool(flags & carb.input.BUTTON_FLAG_DOWN) or value > 0.0


def _poll_keyboard_drive() -> None:
    if _registry.keyboard_subscription is None:
        if _state["elapsed_s"] - _state["last_keyboard_retry_s"] >= 1.0:
            _state["last_keyboard_retry_s"] = _state["elapsed_s"]
            _install_keyboard_handler()
        return

    forward = float(_keyboard_is_pressed(carb.input.KeyboardInput.W)) - float(
        _keyboard_is_pressed(carb.input.KeyboardInput.S)
    )
    turn = float(_keyboard_is_pressed(carb.input.KeyboardInput.A)) - float(
        _keyboard_is_pressed(carb.input.KeyboardInput.D)
    )
    has_drive_input = bool(forward or turn)

    if has_drive_input:
        _state["keyboard_drive_active"] = True
        _set_target(
            linear_m_s=forward * TELEOP_LINEAR_M_S,
            angular_rad_s=turn * TELEOP_ANGULAR_RAD_S,
        )
    elif _state["keyboard_drive_active"]:
        _state["keyboard_drive_active"] = False
        _set_target(0.0, 0.0)


def _build_control_window() -> None:
    if _registry.control_window is not None:
        try:
            _registry.control_window.visible = False
        except Exception:
            pass

    window = ui.Window("UGV Teleop Dataset", width=320, height=265)
    _registry.control_window = window
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label("Keyboard: click viewport, hold W/S/A/D")
            ui.Label("Buttons below are a fallback if keyboard focus fails.")
            with ui.HStack(spacing=6):
                ui.Button(
                    "Forward",
                    clicked_fn=lambda: _set_target(TELEOP_LINEAR_M_S, 0.0),
                )
                ui.Button(
                    "Reverse",
                    clicked_fn=lambda: _set_target(-TELEOP_LINEAR_M_S, 0.0),
                )
            with ui.HStack(spacing=6):
                ui.Button(
                    "Turn Left",
                    clicked_fn=lambda: _set_target(0.0, TELEOP_ANGULAR_RAD_S),
                )
                ui.Button(
                    "Turn Right",
                    clicked_fn=lambda: _set_target(0.0, -TELEOP_ANGULAR_RAD_S),
                )
            with ui.HStack(spacing=6):
                ui.Button("Stop", clicked_fn=_stop_robot)
                ui.Button("Record R", clicked_fn=_toggle_recording)
            ui.Label("Recording saves to assets/ugv-rover/isaacsim/datasets/teleop")


def _on_update(event) -> None:
    dt = event.payload.get("dt", 1.0 / 60.0) if hasattr(event, "payload") else 1.0 / 60.0
    _state["elapsed_s"] += dt
    if _state["elapsed_s"] > 1.0:
        _ensure_stage_open()
    _poll_keyboard_drive()

    _state["linear_m_s"] = _step_toward(
        _state["linear_m_s"],
        _state["target_linear_m_s"],
        LINEAR_ACCEL_M_S2 * dt,
    )
    _state["angular_rad_s"] = _step_toward(
        _state["angular_rad_s"],
        _state["target_angular_rad_s"],
        ANGULAR_ACCEL_RAD_S2 * dt,
    )

    if _apply_drive(_state["linear_m_s"], _state["angular_rad_s"]):
        if not _state["ready"]:
            _state["ready"] = True
            print("UGV teleop connected to four driven wheel joints.")
    elif _state["recording"] and _state["elapsed_s"] - _state["last_pose_wait_report_s"] >= 1.0:
        _state["last_pose_wait_report_s"] = _state["elapsed_s"]
        print("Recording while drive joints are still connecting.")

    _record_sample(dt)


if _registry.update_subscription is not None:
    _registry.update_subscription.unsubscribe()
if _registry.keyboard_subscription is not None:
    _remove_keyboard_handler()

atexit.register(_end_recording)
_install_keyboard_handler()
_build_control_window()
_registry.update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _on_update,
    name="ugv_teleop_dataset_recorder",
)

omni.timeline.get_timeline_interface().play()
print("UGV teleop dataset recorder loaded.")
print("Hold W/S forward/back, A/D turn, Space stop, R start/stop recording.")
