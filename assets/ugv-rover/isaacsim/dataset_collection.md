# UGV Teleoperation Dataset Collection

Use `ugv_teleop_dataset.py` to collect demonstration data before RL training.
The script opens `ugv_dataset_empty_env.usda`, drives the four motorized
wheels, keeps the middle wheels passive, and records state/action samples.

`ugv_dataset_empty_env.usda` is the clean collection scene: a flat ground plane,
a simple rectangular boundary, the UGV robot, and lighting. It sublayers
`ugv_rover_physics.usda`, so robot physics stays in one place while dataset
environments can vary later.

The first arena is about `7.0 m x 5.0 m` with low `0.2 m` walls. This keeps
early demonstrations simple while giving enough space for forward driving,
turning, and stop/recover examples.

## Launch

```bash
/home/nathan/isaacsim/isaac-sim.sh --exec /home/nathan/Downloads/ugv_rpi-main/assets/ugv-rover/isaacsim/ugv_teleop_dataset.py
```

## Controls

- hold `W`: drive forward
- hold `S`: drive backward
- hold `A`: turn left
- hold `D`: turn right
- `Space`: stop
- `R`: start or stop recording

After the script loads, click the Isaac viewport once before using the
keyboard. If the viewport still does not receive keyboard focus, use the
`UGV Teleop Dataset` button window that the script opens.

## Output

Recordings are written under:

```text
assets/ugv-rover/isaacsim/datasets/teleop/YYYYMMDD_HHMMSS/
```

Each run contains:

- `teleop.csv`: synchronized action and pose samples
- `metadata.json`: robot, stage, control, and unit metadata

The CSV stores command actions, applied smoothed actions, chassis pose,
finite-difference linear velocity, yaw, and yaw rate. This is suitable as a
first behavior-cloning dataset before Isaac Sim RL.

## Why Pretrain

Pretraining gives the RL policy a useful starting behavior before reward-based
exploration. Instead of beginning with random wheel commands, the model first
learns from teleoperated examples such as driving forward, turning, stopping,
and recovering from small heading errors.

This usually reduces RL training time, avoids many unstable early rollouts, and
makes sim-to-real safer because the first learned policy already looks somewhat
like human driving. RL can then fine-tune for the actual reward objective.
