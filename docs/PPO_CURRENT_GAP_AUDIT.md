# PPO Current Gap Audit

## Audit boundary

This audit is against source commit `7d6bfda0da593e2cace2accd8bc81d300bdd9288` in the isolated branch `ppo/phase-specific-stability-v1`. The legacy repositories listed in `artifacts/ppo_phase_v1_start/DO_NOT_USE_LEGACY_PPO_SOURCE.md` were not used. All frozen FSM, sensing, infrastructure, environment, and motion-contract hashes matched at the start of development.

The published state is accurately classified as `PPO_INTERFACE_READY / PPO_TRAINING_NOT_STARTED`.

## Existing reusable interface foundation

| Component | Present | Verified behavior | Training gap |
|---|---:|---|---|
| `PPOEnvAdapter` | yes | holds one 15 Hz residual for eight contiguous 120 Hz ticks, projects each tick, aggregates reward, stops mid-decision | backend is only a Protocol/fake; not Gym/Isaac/vectorized |
| `ActionProjector` | yes | tanh, scale, phase/runtime/safety masks, residual slew, joint/wheel limits, hard safety, bitwise zero path | v1 mask is Recording-active-channel based; scales are full-span; no smooth phase bridge |
| Observation v1 | yes | immutable canonical 85D order and finite fixed normalization | lacks lifecycle, calibrated dynamics, force/load/slip, active-leg dynamics, previous projected residual |
| Reward v1 | yes | config-driven normalized breakdown | thirteen global terms, no physical potential, no phase/lifecycle objectives, duplicate signals |
| Termination v1 | yes | ordered success/task-failure/safety-abort/timeout classifications | no live signal provider; timeout is not reset-relative |
| Episode logger | yes | immutable JSONL/Parquet transition validation | fixed 85D and one-episode baseline format; not a rollout buffer |
| Selected-trial exporter | yes | streamed 120 Hz ledger checks and zero-residual proof | offline baseline artifact only |
| Historical physical zero residual | yes | Trial 043: 12,952 ticks, 1,619 decisions, bitwise nominal equality, physical success | one historical run through `ResidualInterface`, not five fresh live `ActionProjector` episodes |

## Mandatory false flags and guards

The v1 profiles deliberately contain `training_enabled: false`; `configs/ppo_interface.yaml` also contains `residual_enabled: false`. The v1 action, reward, termination, and domain-randomization loaders reject a true training flag. The live `app_runtime` constructs `ResidualInterface(residual_enabled=False)` and always supplies twelve zeros. These safeguards remain intact; training uses separate versioned v2 profiles.

## Concrete v1 action defects

V1 active-channel counts for P01-P13 are `7, 2, 8, 5, 6, 4, 5, 1, 10, 1, 1, 6, 12`. P02 only permits the FR joints, P08/P10/P11 permit one channel, and P12 excludes its support-leg corrections. Every phase scale is `1.0` times the complete safe actuator span (266 degrees for a typical servo or 4.18879 rad/s for a wheel). The adapter also zeros prior residual state immediately at a phase transition, which can create the prohibited discontinuity.

## Missing at audit start

- live `IsaacFSMBackend` and Isaac-backed trainable environment;
- vectorized environments and independent controller/contact/latch/history state;
- reset-only snapshot capture/restore and root-write audit;
- Observation v2 and level-reference calibration;
- phase-role masks, per-channel scales, transition bridge;
- compact five-family Reward v2 and phase/substage metrics;
- RSL-RL actor, critic, rollout, GAE, PPO update and checkpoint lifecycle;
- live zero/nonzero/vector smoke gates;
- fresh paired FSM evaluation and phase snapshots;
- trained nonzero checkpoint, promotion/locked-test evaluation;
- required plots, reports and four final videos.

## Live integration hazards

The frozen runtime is intentionally single-instance: fixed `/World/WLRRobot` and `/World/Obstacle` paths, a one-row `RobotAdapter`, one-row sensing tensor checks, and exact single-robot contact sensor paths. Vectorization therefore requires a PPO-only environment-indexed scene/adapter while importing the frozen mappings, constants, contact classification, collision logic, guard logic, and FSM unchanged. Merely changing an environment count would cause state and sensor contamination.

Trial 043 was physically re-adjudicated as success after its original manifest ended one tick short of the P13 wheel-decay quality gate. Its frozen physical evidence is valid, but it cannot substitute for fresh paired evaluation.

Exact-pair sensor unavailability must fail closed during training. A missing sensor must never be interpreted as proof of no body collision.

## Selected implementation path

1. Keep every v1 profile and frozen file byte-identical.
2. Use installed, officially pinned `rsl-rl-lib==5.0.1` through the current Isaac Lab environment; do not replace the PPO optimizer.
3. Build a single-environment backend first and verify exact 120 Hz atomic dispatch against the frozen runtime.
4. Build Observation/Reward/action v2 as Isaac-free, exhaustively unit-tested layers.
5. Add a PPO-only environment-indexed live scene with one independent FSM, sensor classifier/tracker, reward state, and residual history per environment.
6. Pass single and vector zero/nonzero gates before training.
7. Train phase-balanced snapshots, then full P01-P13 episodes, selecting checkpoints only by physical validity/safety/stability priority.

## Audit test baseline

With the locked Isaac Python and `PYTHONPATH=src`, the unmodified starting suite passed: `306 passed`.
