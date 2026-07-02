# Bimanual YAM — MolmoAct2 closed-loop eval

This directory holds the **robot/client side** of MolmoAct2 on a bimanual YAM
setup. The inference **server** (`host_server_yam.py`) loads the
[`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM)
checkpoint and serves actions over HTTP; the eval launcher here drives the two
YAM arms, captures the 3-camera observation, queries a policy, executes the
returned action chunk, records each rollout, and (optionally) converts a
labeled session into a LeRobot v3.0 dataset.

It is vendored and trimmed from the reference YAM implementation at
<https://github.com/williamtsai726/YAM> — only the eval-relevant pieces are
kept (teleop, data collection, and the Gello leader-arm code are omitted).

> Hardware-coupled example: it talks to real YAM arms over CAN (via `i2rt`) and
> Intel RealSense cameras. It is meant to run on the workstation wired to the
> robot, not in the dependency-light server environment.

## Layout

```
examples/yam/
├── host_server_yam.py            # inference server (separate; see top-level README §5)
├── launch_yaml_eval_molmoact.py  # eval launcher — main entry point
├── molmoact_client.py            # MolmoAct (HTTP) + MolmoActLocal (in-process) policies
├── camera_server.py              # long-lived ZMQ server owning the 3 RealSense cams
├── camera_client.py              # ZMQ client + standalone live viewer
├── eval_utils.py                 # per-rollout saver, cv2 viewer, labeling, conversion
├── lerobot_convert.py            # raw rollouts -> LeRobot v3.0 dataset
├── start_camera_server.sh        # convenience launcher for camera_server.py
├── requirements.txt
├── configs/
│   ├── yam_left.yaml             # cameras, storage, eval, lerobot + left arm
│   └── yam_right.yaml            # right arm only
└── gello_min/                    # trimmed YAM runtime (robot/env/camera drivers)
```

## Install

```bash
pip install -r examples/yam/requirements.txt
# Plus the two non-PyPI deps (see requirements.txt):
#   i2rt    — YAM CAN/motor driver (required)
#   lerobot — only for the optional dataset conversion
```

Run every command below **from the molmoact2 repo root** (the scripts add
`examples/yam/` to `sys.path`, so `gello_min` and the sibling modules resolve).

## Inference modes

Set `eval.mode` in `configs/yam_left.yaml`:

- **`server`** (default) — POST observations to a running `host_server_yam.py`
  using the `json_numpy` wire protocol. Point `eval.molmoact_server` at the
  server (`host:port`, full URL, or ngrok hostname; `/act` is appended).
  Start the server per the top-level README §5, e.g.
  `uv run python examples/yam/host_server_yam.py --port 8202`.
- **`local`** — load the checkpoint in-process via `transformers` (no server).
  Configure under `eval.local`. bf16 needs ~10–14 GB VRAM, fp32 ~26 GB.

The two policies are interchangeable; the launcher picks one from the config.

## Hardware setup

1. Both YAM arms powered, e-stop released.
2. The 3 RealSense cameras plugged into USB 3; put their serials in
   `configs/yam_left.yaml` under `sensors.cameras`. **Order matters** — the
   model was trained on `[top, left, right]`; here `front_camera` plays the
   `top` role.
3. Bring up CAN and set the camera/CAN interface names in the configs
   (`channel:` — find them with `ip link show`). Disable the motor watchdog so
   the arms don't collapse during long sessions (see the `i2rt` docs / your
   YAM bring-up scripts).

## Run a session

Two terminals when the camera server is enabled (the default).

**Terminal A — camera server (long-lived):**

```bash
bash examples/yam/start_camera_server.sh
# or: python examples/yam/camera_server.py --config examples/yam/configs/yam_left.yaml
```

Wait for `REP bound on tcp://127.0.0.1:5555` / `PUB bound on tcp://127.0.0.1:5556`.
It holds the cameras warm across sessions and feeds the live viewer's PUB
stream so the cv2 window keeps repainting during inference.

**Terminal B — eval:**

```bash
python examples/yam/launch_yaml_eval_molmoact.py \
    --left_config_path  examples/yam/configs/yam_left.yaml \
    --right_config_path examples/yam/configs/yam_right.yaml \
    -n 10
```

`-n 10` runs 10 rollouts. Set `eval.camera_server.enabled: false` to open the
cameras in-process instead (one fewer terminal, but the viewer freezes during
inference).

## What happens per rollout

1. Arms interpolate to `agent.start_joints` — your cue to reset the workspace.
2. Stdin prompts for the task instruction (Enter reuses the previous one).
3. The rollout runs; a 3-pane cv2 window (`YAM Eval`) shows LEFT / FRONT / RIGHT.
4. End it by pressing a key **in the cv2 window**:
   - `y` → success, `n` → failure, `q` → quit (kept unlabeled under `eval/`)
   - or let it hit `max_steps` → you're prompted on stdin afterwards.

`Ctrl-C` is handled: the in-progress rollout is flushed with an `err.md`
marker and any rollouts already labeled this session are still converted.

## Where files land

Under `{storage.base_dir}/data/{storage.task_directory}/`:

```
eval/<ts>/                       # quit / unlabeled rollouts
success/<YYYY-MM-DD>/<ts>/
failure/<YYYY-MM-DD>/<ts>/
eval_lerobot_v30/<session_ts>/   # LeRobot v3.0 dataset (labeled rollouts, end of session)
```

Each rollout has `episode.h5` (joint trajectory, command targets, effort
telemetry, instruction) and one PNG per camera per frame under `left_rgb/`,
`front_rgb/`, `right_rgb/`.

## Key config knobs (`configs/yam_left.yaml`)

| Key | Meaning |
|---|---|
| `eval.mode` | `server` (HTTP) or `local` (in-process). |
| `eval.molmoact_server` | Server address for `mode: server`. |
| `eval.local.*` | Checkpoint / device / dtype for `mode: local`. |
| `eval.camera_server.enabled` | `true` uses the ZMQ camera server; `false` opens cameras in-process. |
| `eval.live_view_enabled` | `false` disables the cv2 window (headless runs). |
| `force_safety.*` | Optional synchronized command limiting and raw-effort hold/abort thresholds. |
| `max_steps` | Per-rollout timeout in control steps. |
| `storage.*` | Output location, instruction, PNG save settings. |
| `lerobot.*` | End-of-session dataset conversion knobs. |

## Force safety guard

The eval launcher wires `force_safety` from `configs/yam_left.yaml` into
`RobotEnv.step_command_only()`, so policy actions, reset moves, and interpolated
sub-steps all pass through the same command path.

Current defaults:

- YAM arms are created with `zero_gravity_mode: false` for inference.
- Gripper force is capped through i2rt's `limit_gripper_force` argument.
- Command limiting is off by default so supervised tele-op and training remain
  responsive while a human is watching.
- If command limiting is enabled for unattended inference, it uses synchronized
  joint-space scaling (`command_limit_mode: scale`) instead of independently
  clipping each joint. That preserves the requested path direction while the
  arm catches up over multiple control ticks.
- Raw effort thresholds are present but left unset until free-space effort logs
  are collected for the active arms, grippers, payload, and speed.
- NEXT-lite residual thresholds are also present but left unset. They require a
  trained `next_lite.checkpoint_path` and should be validated on a soft
  surrogate before glass handling.

Rollout HDF5 files include command/modeling fields when available:

- `state`, `next_state`
- `policy_action`
- `requested_joint_positions`
- `commanded_joint_positions`
- `command_delta`
- `joint_velocities`, `next_joint_velocities`
- `joint_efforts`, `next_joint_efforts`
- `force_residual`
- `force_contact_score`
- `force_contact_state_code`
- `interpolation_steps`

Use those logs to tune first raw thresholds before enabling hard effort aborts.

## Free-space force baseline

Collect contact-free effort data without policy inference:

```bash
python examples/yam/collect_force_baseline.py \
  --left_config_path examples/yam/configs/yam_left.yaml \
  --output_dir ./yam_force_baselines \
  --skip_move_to_start \
  --joints 0 \
  --amplitude_rad 0.02 \
  --dry_run
```

The script runs autonomous scripted motion, not tele-op. It moves small
sinusoidal joint sweeps around the current pose when `--skip_move_to_start` is
set, or around the configured start pose otherwise. It writes
`free_space_baseline.h5` with per-control-tick `q`, `qdot`, effort, target,
requested command, and sent command.

For a cluttered bench, use `--dry_run` first, start with one known-safe joint,
and keep the amplitude small. Only run without `--dry_run` after verifying the
planned swept volume clears the table, camera pole, cables, and fixtures. Use
`--joint_amplitudes 0:0.02,1:0.015` to set per-joint amplitudes. The script
refuses to move unless `--joints` is explicit; use `--joints all` only in a
fully clear workspace.

After collection, generate a first conservative raw-effort threshold block:

```bash
python examples/yam/analyze_force_baseline.py \
  ./yam_force_baselines/<run_ts>/free_space_baseline.h5 \
  --output_path ./yam_force_baselines/<run_ts>/force_safety_thresholds.yaml
```

Treat the output as a starting point. Validate warning/freeze and hard-abort
behavior on a soft surrogate before using the thresholds around glass.

## NEXT-lite expected-effort model

For FACTR2-style external torque, train NEXT-lite on contact-free logs and plot
the residual instead of raw effort:

```bash
python examples/yam/train_next_lite.py \
  --input-glob "./yam_teleop_force_logs/*/teleop_force_log.h5" \
  --output_dir ./yam_next_lite_runs \
  --history 50 \
  --epochs 50

python examples/yam/apply_next_lite.py \
  ./yam_teleop_force_logs/<run_ts>/teleop_force_log.h5 \
  ./yam_next_lite_runs/<train_ts>/model.pt \
  --output_path ./yam_teleop_force_logs/<run_ts>/teleop_force_log_with_residual.h5

python examples/yam/plot_force_timeline.py \
  ./yam_teleop_force_logs/<run_ts>/teleop_force_log_with_residual.h5 \
  --output_path ./yam_teleop_force_logs/<run_ts>/force_residual_timeline.png
```

The passive tele-op logger stores `commanded_joint_positions` as the observed
joint position proxy. That is enough to smoke-test the residual pipeline, but
it is not sufficient for FACTR2/NEXT-quality contact testing. NEXT relies on
the command tracking error `commanded_q - q` to model controller effort and
actuator response, especially during fast free-space motion. Use the
command-aware GELLO collector for training data and for live validation.

For live read-only smoke testing from a Mac while the follower portal server
runs on the robot PC:

```bash
python examples/yam/live_next_lite_force_monitor.py \
  ./yam_next_lite_runs/<train_ts>/model.pt \
  --ssh-host steven@100.99.120.65 \
  --host 127.0.0.1 \
  --port 11333 \
  --hz 30
```

This opens a local cv2 window with the rolling scalar residual and per-joint
residual bars. It does not command the robot. The robot PC runs
`examples/yam/stream_teleop_force.py` over SSH, which only reads from the
existing follower portal server. Because this path cannot see the active
leader/follower target command, it should not be used to decide whether light
contact is detected correctly.

## Tele-op force logging

For cluttered benches, prefer human-guided leader/follower data over autonomous
sweeps. The passive logger is only for read-only pipeline checks. Start the
i2rt follower server, then run the passive logger in parallel:

```bash
python examples/yam/record_teleop_force_log.py \
  --output_dir ./yam_teleop_force_logs \
  --host 127.0.0.1 \
  --port 11333 \
  --hz 50 \
  --duration_sec 120
```

The logger connects to the follower `minimum_gello.py` portal server and only
calls read methods. It does not open CAN and does not command the robot. Move
the leader/follower through contact-free, task-relevant free-space motions:
slow, medium, above-table, near-but-clear of fixtures, and around the intended
glass-handling envelope.

The current passive logger cannot see the leader's exact target command, so it
stores `commanded_joint_positions = joint_positions` as an explicit proxy. Do
not train the production NEXT-lite model from this proxy data unless there is
no command-aware alternative.

For NEXT training data, use the command-aware GELLO collector. It owns the
leader loop, runs at 100 Hz by default, and records the actual target sent to
the follower as `commanded_joint_positions`:

```bash
python examples/yam/record_gello_next_dataset.py \
  --output-dir ./yam_next_data_logs \
  --server-host 127.0.0.1 \
  --server-port 11333 \
  --leader-can-channel can_leader_l \
  --hz 100 \
  --duration-sec 600 \
  --next-lite-checkpoint ./yam_next_lite_runs/<train_ts>/model.pt \
  --force-stream-jsonl /tmp/yam_left_force.jsonl \
  --force-stream-hz 30 \
  --arm-label left
```

This script is motion-capable because it replaces the normal
`minimum_gello.py --mode leader` process. The follower server must already be
running. Press the leader top button to synchronize/start, then move through
contact-free, task-relevant free-space motions. Press the leader top button
again or `Ctrl-C` to stop and flush the HDF5 log. Keep contact, bumps, fixture
touches, and object pushes out of training logs; those runs are useful as
held-out evaluation after the free-space model is trained.

Match the FACTR2/NEXT data-collection pattern as closely as the bench permits:

- independent safe single-joint motion across each joint's clear range
- Cartesian-like multi-joint end-effector motion through the task envelope
- repeated slow and fast motions to cover velocity-dependent dynamics
- expected payload/tool/gripper configurations
- no contact with the table, camera pole, objects, cables, or glass

The ThinkPad recording dashboard can display this command-aware stream by
reading the same JSONL file:

```bash
YAM_FORCE_JSONL=/tmp/yam_left_force.jsonl \
python3 scripts/yam_button_record_dashboard.py --port 8090
```

The force panel should show `command-aware` once the recorder has synchronized
and NEXT-lite has warmed up. If it shows `proxy`, the dashboard is looking at a
read-only smoke-test stream rather than the true leader/follower command path.

The command-aware log captures future-proof numeric streams:

- follower `joint_positions`, `joint_velocities`, and `joint_efforts`
- true `commanded_joint_positions` sent to the follower
- derived `commanded_joint_velocities` and `command_error`
- leader joint positions and leader button states
- phase code for sync interpolation vs synchronized tele-op
- loop timing, loop lag, and follower read latency
- motor temperatures when exposed by the follower server
- metadata for effort source, units, command source, target rate, and actual rate

Before collecting a long NEXT run, audit the effort-unit provenance:

```bash
python examples/yam/audit_yam_effort_units.py \
  --arm yam \
  --gripper linear_4310 \
  --output-path ./yam_next_data_logs/effort_unit_audit_left_follower.json
```

This is a no-motion software audit. It verifies which DM motor types, torque
decode ranges, and direction signs i2rt uses for `joint_efforts`. For the
standard YAM + linear 4310 follower, joints 0-2 decode as DM4340 feedback
torque over +/-28 Nm and joints 3-6 decode as DM4310 feedback torque over
+/-10 Nm. That makes `joint_efforts` nominal motor torque in Nm, not raw ADC
current.

That audit does not prove absolute physical calibration. To strengthen the
signal before glass work:

- Static gravity sanity: hold several safe, contact-free poses for 10-20 s
  each, then compare measured `joint_efforts` against the MuJoCo/i2rt gravity
  torque at the same `q`. Signs and approximate scale should agree; residual
  bias captures friction, payload, and model error that NEXT should learn.
- Known-load check: apply a small known force with a force gauge or known
  hanging weight at a known end-effector offset, then compare the change in
  residual torque with `J(q)^T F`. This is the proper absolute calibration
  check for whether a displayed residual corresponds to real external load.
- NEXT training itself does not require perfect absolute Nm calibration as long
  as training and deployment use the same effort signal. Absolute calibration
  matters when choosing physical thresholds for glass handling or reporting
  force in real units.

## NEXT-lite training

Train a learned free-space effort predictor from contact-free baseline logs:

```bash
python examples/yam/train_next_lite.py \
  --input_glob "./yam_force_baselines/*/free_space_baseline.h5,./yam_next_data_logs/*/gello_next_log.h5,./yam_teleop_force_logs/*/teleop_force_log.h5" \
  --output_dir ./yam_next_lite_runs \
  --history 50 \
  --epochs 50
```

The model predicts expected free-space effort from recent
`[q, qdot, commanded_q - q]`. At runtime, the contact signal is the residual:

```text
external_load = measured_effort - predicted_free_space_effort
```

Use the validation residual stats in `metrics.json` to choose first residual
thresholds before wiring the model into live stop/retreat behavior.

To enable residual monitoring after validation, set the checkpoint and residual
thresholds in `configs/yam_left.yaml`:

```yaml
force_safety:
  next_lite:
    checkpoint_path: ./yam_next_lite_runs/<run_ts>/model.pt
    device: cpu
  warning_abs_residual: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  hard_abs_residual: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
  hard_residual_norm: 0.4
```

The monitor keeps the first `history` ticks as warmup. During warmup, raw effort
thresholds still apply if configured, but residual thresholds do not trigger
until the model has a full history window.

## Force timeline plots

The FACTR2-style operator signal is a scalar contact score, not XYZ force. We
log `force_contact_score` and show it in the live camera header. With NEXT-lite
enabled, this score comes from the learned external joint-torque residual.

Generate a timeline plot from a rollout or baseline HDF5 file:

```bash
python examples/yam/plot_force_timeline.py \
  ./yam_eval_runs/data/session/eval/<run_ts>/episode.h5 \
  --output_path ./force_timeline.png
```

The plot shades free motion, pre-contact, and contact using hysteresis on the
scalar score. Per-joint residuals remain in `force_residual` for diagnosis.

A later diagnostic layer can estimate approximate end-effector wrench from the
joint residual and the arm Jacobian:

```text
wrench_ext ~= pinv(J(q)^T) force_residual
```

That display would be useful for intuition (`Fx/Fy/Fz`), but the scalar contact
score and per-joint residual should remain the safety signals until the wrench
estimate is calibrated.

## Camera server, standalone

Sanity-check the cameras independently of the eval loop:

```bash
python examples/yam/camera_client.py --mode sub      # subscribe to the PUB stream
```
