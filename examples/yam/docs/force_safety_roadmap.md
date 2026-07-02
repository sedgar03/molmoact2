# YAM Glass Handling Force Safety Roadmap

## Goal

Build a sensing and veto layer below policy inference so the YAM arms stop or back off when measured load is larger than expected. The policy should propose actions; the safety layer decides whether each action is allowed.

The working signal is:

```text
external_load = measured_joint_effort - expected_free_space_effort
```

For glass handling, `external_load` is treated as a guarded contact signal. Unexpected load must freeze, retreat, or abort the rollout before the arm keeps pushing.

## Operating Assumptions

- Inference should create YAM robots with `zero_gravity_mode=False` so the arm starts in position-hold control instead of gravity-comp idle.
- This is not enough for glass safety. Position control can still push through fragile objects if bad targets keep arriving.
- The force safety layer must live below MolmoAct, near `RobotEnv.step_command_only()` or `YAMRobot.command_joint_state()`, so every command path is guarded.
- Learned estimators are useful after deterministic limits are in place. They should improve sensitivity, not be the first and only protection.

## Phase 0: Make State Observable

Expose and log the robot quantities needed to estimate load:

- `joint_pos`
- `joint_vel`
- `joint_eff`
- commanded target joint positions
- command deltas and interpolation step count
- gripper position, velocity, and effort
- robot config values: `zero_gravity_mode`, `kp`, `kd`, gravity compensation factor, joint limits, gripper force limit

Acceptance criteria:

- Every rollout step can be replayed with policy action, command sent to robot, measured joint state, and measured effort.
- The YAM wrapper exposes arm effort and gripper effort in observations.
- A dry-run script can collect free-space motion logs without running policy inference.

## Phase 1: Deterministic Effort Watchdog

Add a conservative watchdog before any learned observer:

```text
effort_score[i] = abs(filtered_joint_eff[i])
```

The watchdog should support:

- per-joint warning thresholds
- per-joint hard-stop thresholds
- total norm threshold
- short filtering window, initially 30-100 ms
- minimum duration before trigger, initially 2-3 control ticks
- optional synchronized command delta and velocity limits for unattended
  inference; keep these off for supervised tele-op and training

Actions on trigger:

- warning: freeze target at current pose
- stop: abort rollout and hold current pose
- severe stop: abort and optionally disable actuation if the arm is still driving into contact

Acceptance criteria:

- A single config enables or disables the watchdog.
- Threshold events are logged with timestamp, joint, measured effort, command, and action taken.
- Policy actions cannot bypass the watchdog.

## Phase 2: Model-Based Expected Effort

Replace raw effort thresholds with residual thresholds:

```text
expected_joint_effort =
    gravity_torque(q)
  + commanded_pd_effort(q_target, q, qdot)
  + friction_estimate(qdot)
  + payload_bias

residual = measured_joint_effort - expected_joint_effort
```

Initial model:

- use the i2rt gravity model already available in `MotorChainRobot`
- estimate commanded PD effort from the current target, current position, current velocity, `kp`, and `kd`
- use a simple per-joint Coulomb/viscous friction fit from free-space logs
- keep payload/tool state explicit; glass tools and grippers need separate baselines

Acceptance criteria:

- Residual is near zero in free-space motion at normal inference speeds.
- Residual spikes on deliberate gentle contact with foam or a force gauge.
- False positives are low enough to finish non-contact rollouts.

## Phase 3: NEXT-Lite Learned Observer

Reproduce the practical FACTR2/NEXT idea for this stack:

```text
expected_free_space_effort = f(history(q, qdot, commanded_q - q, gripper))
external_load = measured_joint_effort - expected_free_space_effort
```

Source-backed implementation constraints from FACTR2/NEXT:

- Train only on contact-free free-space data. Contact belongs in held-out
  evaluation or downstream FIRST-style phase labeling, not in the free-space
  inverse-dynamics target.
- Capture at 100 Hz when possible. The reference NEXT setup uses 100 Hz data
  and a 50-tick LSTM history, so the model has about 0.5 s of recent motion
  context.
- The live deployment input must include true command tracking error
  `commanded_q - q`. A read-only stream that substitutes `commanded_q = q` is
  useful only for pipeline smoke tests; it is not a valid contact test because
  it hides controller tracking error and turns fast free-space dynamics into
  apparent external load.
- The scalar operator signal should be `||external_load||_1` with hysteresis
  for free/pre-contact/contact labeling. Per-joint residuals remain available
  for diagnosis.

Start with free-space-only training data:

- human-guided leader/follower logs for the cluttered bench setup, recorded by
  the command-aware collector
- independent single-joint motions across the safe range for each joint
- Cartesian-like multi-joint end-effector motions through the task envelope
- repeated slow and fast motions to capture velocity-dependent dynamics,
  friction, hysteresis, and controller lag
- reachable regions of the glass-handling envelope that clear the table,
  camera pole, cables, and fixtures
- gripper open/close cycles
- expected payload/tool configurations
- both arms independently and bimanual motion if used in production

Candidate models:

- keep the FACTR2-like default as a 2-layer LSTM over 50 ticks of
  `[q, qdot, commanded_q - q]`
- use MLP/GRU variants only for ablations or troubleshooting

Acceptance criteria:

- External-load estimate remains quiet in free space.
- Estimate detects gentle contact earlier than raw effort thresholds.
- Runtime is fast enough for the robot command loop or a nearby monitor thread.
- Operator view shows a FACTR2-style scalar contact score over time, with
  per-joint residual available for diagnosis.

## Phase 4: Glass Mode

Add a stricter operating profile when glass is in the workspace:

- lower residual thresholds near known glass regions
- lower synchronized joint-space delta per control tick when running
  unattended inference
- lower gripper force limit
- no high-speed moves toward glass
- automatic retreat on contact while approaching glass
- require explicit phase labels: free-space, pre-contact, contact, grasp, retreat

Acceptance criteria:

- Contact with a fragile surrogate triggers freeze or retreat before visible deformation.
- Gripper cannot exceed configured glass force limit during grasp attempts.
- The monitor can explain every intervention from logged signals.

## Phase 5: Approximate Cartesian Wrench View

Add an operator-facing diagnostic that maps joint residuals into an approximate
end-effector wrench:

```text
tau_ext ~= J(q)^T wrench_ext
wrench_ext ~= pinv(J(q)^T) tau_ext
```

This should be treated as interpretability, not the primary safety signal. The
authoritative signals remain:

- per-joint residual
- scalar contact score
- thresholded contact state

Requirements:

- reliable YAM kinematics and Jacobian for the active arm
- clear convention for end-effector frame vs world/base frame
- gripper joint excluded from arm-wrench solve
- calibration against a soft surrogate or force gauge
- UI label must say "estimated EE wrench" or equivalent, not ground-truth force

Acceptance criteria:

- Estimated wrench direction is qualitatively correct for simple hand-applied
  loads on a soft surrogate.
- The scalar contact score and per-joint residual still drive freeze/abort.
- The plot/dashboard can show both `Fx/Fy/Fz` estimate and the underlying joint
  residual that produced it.

## Immediate Work Items

1. Replace the dashboard's passive residual stream with a command-aware live
   stream that receives the true target sent to the follower.
2. Record a clean 10-minute contact-free command-aware run at 100 Hz with:
   independent safe single-joint motions, Cartesian-like multi-joint motions,
   and repeated slow/fast coverage.
3. Keep contact/JBL/push tests out of training; use them as held-out
   evaluation after the free-space model is trained.
4. Derive first conservative raw effort warning/hard thresholds with `analyze_force_baseline.py`.
5. Train and evaluate NEXT-lite on free-space logs with `train_next_lite.py`.
6. Enable raw or residual thresholds in `configs/yam_left.yaml` and validate freeze/abort behavior on a soft surrogate.
7. Tune task-phase thresholds and retreat behavior for glass handling.
8. Add approximate Jacobian-based EE wrench visualization after residual
   thresholds are validated.

## Implementation Log

2026-06-30:

- Implemented first-pass guard in `sedgar03/molmoact2` commit `b2f591d`.
- YAM eval configs now instantiate arms with `zero_gravity_mode: false` and `limit_gripper_force: 20.0`.
- `YAMRobot` exposes `joint_efforts`, `joint_eff`, `gripper_effort`, and `gripper_eff` when i2rt provides them.
- `RobotEnv.step_command_only()` now applies a configurable `ForceSafetyMonitor` to policy actions, reset moves, and interpolated sub-steps.
- Command limiting is default-off in the YAM eval config. If enabled for unattended inference, it scales the whole joint-space command toward the target rather than independently clipping joints. Raw effort warning/hard thresholds are present but intentionally unset until free-space baseline logs are collected.
- Rollout `episode.h5` files now record policy targets, requested commands, sent commands, command deltas, interpolation step counts, joint velocities, and joint efforts for replay/modeling.
- Added `collect_force_baseline.py` to gather contact-free per-control-tick HDF5 logs without policy inference.
- Added `analyze_force_baseline.py` to turn those logs into first-pass raw effort threshold recommendations.
- Added `train_next_lite.py` and `gello_min.next_lite` to train/evaluate a FACTR2-style free-space effort predictor from HDF5 logs.
- Wired optional NEXT-lite checkpoints into `ForceSafetyMonitor` so live residual thresholds can freeze/abort once a trained model and validation thresholds are configured.
- Added `force_contact_score` telemetry, live camera-view force status, and `plot_force_timeline.py` for FACTR2-style free/pre-contact/contact timelines.
- Added `record_teleop_force_log.py` for passive leader/follower free-space data collection through the i2rt follower portal server.

2026-07-01:

- Re-read FACTR2/NEXT sources and local X/PDF captures. The reference method
  trains a free-space inverse-dynamics LSTM on contact-free data and deploys at
  100 Hz with a 50-tick history of `[q, qdot, commanded_q - q]`.
- Validated that the passive live dashboard stream is not sufficient for
  contact testing because it substitutes `commanded_q = q`. On a left-arm
  contact-probe log, the command-aware residual had L2 p99 `1.83`, while the
  proxy-command residual had L2 p99 `9.71`.
- Next implementation step is a command-aware live dashboard path, then a clean
  10-minute 100 Hz free-space run with deliberate slow/fast safe-envelope
  coverage.

## Open Questions

- What glass objects matter first: flat panes, beakers, bottles, slides, or general glassware?
- Do we know the expected end-effector payload and tool mass for each task?
- Should contact response be freeze-only at first, or freeze plus small retreat?
- Which process owns final authority: MolmoAct launcher, YAM wrapper, or i2rt `MotorChainRobot`?
- What is the acceptable false-stop rate during early development?
