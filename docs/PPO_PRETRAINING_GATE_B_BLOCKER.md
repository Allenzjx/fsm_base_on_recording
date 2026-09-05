# Pre-training Gate B blocker

Status: `PPO_PRETRAINING_GATE_B_BLOCKED`.

This is an incomplete delivery. Real Isaac PPO optimizer updates: **zero**.
There is no canonical trained, best-validation or improved checkpoint, no
checkpoint improvement comparison, and no final improved-policy video.
`PPO_TRAINING_COMPLETED_BUT_IMPROVEMENT_GATE_FAILED` is not applicable because
the training budget has not been executed.

## Preserved baseline and positive physical evidence

- Source: `https://github.com/Allenzjx/fsm_base_on_recording`.
- Worktree: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1`.
- Branch: `ppo/phase-specific-stability-v1`.
- Starting commit: `7d6bfda0da593e2cace2accd8bc81d300bdd9288`.
- Latest physical-runtime commit: `7624c4044f8851f4c9a9229607ef014eab2e86a4`.
- Frozen Trial 043 / v010 / RR_FIRST / P01--P13 / Full12 / 120 Hz / 15 Hz.
- The original five fresh validation-seed FSM episodes (2001--2005) all
  succeeded in 107.86666666666666 seconds on commit `36a0d57eb96a03cb8b285f04a16602fafb15d464`.
- A subsequent exact-zero diagnostic, seed 1001 and native audit ON, succeeded
  on `c910e521003a326b836624ddd60963f3c48cfdf8`. All 12944 tick audits were
  valid with equal actual/counterfactual targets, and all 12945 serialized
  observations exactly matched the original seed-2001 baseline.
- The original three canonical baseline metrics files have been preserved.
  New formal baseline publication, if resumed, must use the existing supported
  `outputs\ppo_phase_v1\metrics_runtime_v2` directory to avoid overwriting them.
- Prior phase-entry holdout/zero-rollout results remain available, but must not
  be relabeled as acceptance for a newer runtime commit.

## Latest failed physical probe

Evidence directory:
`runs/ppo_phase_v1/nonzero-residual-smoke/20260905T033652809290Z_g7624c4044f88_c5dc85cd3cb31_s1001_n1_nonzero-residual-smoke`.

The authoritative files are `acceptance.json`, `episode_000_seed_1001/task_events.jsonl`,
`episode_000_seed_1001/episode_summary.json`, and the finalized `run_manifest.json`.

| Measurement | Actual | Required |
| --- | ---: | ---: |
| P10 RR-knee entry position error | +6.934885724705289 degrees | absolute error <= 2 degrees |
| P10 signed entry velocity error | -21.3050901935639 degrees/second | absolute error <= 3.53779995797403 |
| Duration to controller-blocked finalization | 65.36666666666666 seconds | full P01--P13 success |
| Own-phase native target coverage | P01--P10 | P01--P13 |
| Nonzero handoffs | 9 | 12 |
| Valid native target audits | 7844 / 7844 | complete |
| Qualifying own-phase effect ticks | 7387 | count only genuinely changed native targets |
| Body collision / wheel-only climb / safety abort | 0 / 0 / 0 | 0 / 0 / 0 |
| In-episode root writes / Recording accesses | 0 / 0 | 0 / 0 |

The manual wheel pulse was only 0.00005--0.0001 percent of each unchanged phase
scale. It was not a trained policy and is not eligible for the separate
trained-policy activity threshold. Float32-invisible pulses were excluded from
physical effect counts; the test did not pass merely because Python values were
nonzero. The largest actual wheel target change was 1.1920928955078125e-7 rad/s.

Earlier tested servo and wheel probes also failed at P10. An instrumented
comparison identifies a genuine measured-state-to-frozen-feedback chain: the
first native difference on an unexcited RL knee is exactly the existing -8
tracking gain times the previous measured state difference. Read-only branch
checks found no fixed nonzero-path target jump. These observations support
closed-loop sensitivity, not a proof that all possible PPO policies are infeasible.

## What has not happened

The most recent serial launcher stopped at Gate B. It did not start its new
holdout, fresh five-seed baseline, reused-reset acceptance, vector benchmark
matrix, initial checkpoint publication or PPO training. The external training
driver remains protected by its explicit `PRETRAINING_HOLD`; it has not been
removed to bypass a failed prerequisite. Unit-test PPO updates on synthetic
environments do not count as real Isaac training.

The implemented budget remains 10000 smoke + 100000 phase curriculum + 100000
full-episode requested policy decisions. Cadence fixes preserve a sufficiently
long configured full-episode collection window, but no live training budget has
yet been consumed. No locked-test seeds or policy video evaluations were used
for tuning these manual probes.

## Decision needed to continue

Do not modify the frozen FSM/mapper/guards, weaken final checkpoint promotion,
or relabel a failed run as passed. Do not reduce the manual pulse below actual
target resolution simply to reproduce the zero trajectory.

One possible continuation, requiring user confirmation, is to change only the
pre-training Gate B protocol to real phase-local nonzero coverage using the
already source-proven training-reset snapshots. It would separately retain the
failed full-sequence diagnostics and require actual native actuator effects,
normal finalization, mask/bridge/reset integrity and unchanged safety checks in
every phase. This would not waive any final P01-start full-sequence validation,
locked-test, nonzero-activity, stability-improvement or video requirement.

This alternative protocol has **not** been implemented or declared passed.
Whether it will allow a successful training chain is also unverified. The user
must confirm any such material change to the pre-training acceptance boundary.
