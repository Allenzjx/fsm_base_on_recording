# PPO phase-entry calibration log

## P10 physical replay pre-roll

- Failed calibration run: `20260904T220426792632Z_g94e3eec0c786_ce3d8295ae93a_s1002_n1_phase-effective-entry-calibration`
- Changed factor: P10 reset-only physical replay anchor
- Old value: Trial043 tick 7577; replay ticks 7577--7793 (217 steps)
- New value: Trial043 tick 7552; replay ticks 7552--7793 (242 steps)
- Unchanged anchors: predecessor `P09/VERIFY_RESULT` tick 7776, controller `P10/WAIT_ENTRY` tick 7784, target entry tick 7794

The old anchor was only the first row of the three-sample contact history behind
the RR `TOP_LOADED` latch.  It was already inside the contact impulse: root
angular speed reached 1.07471 rad/s and maximum servo speed reached
114.5558 deg/s.  State write cannot restore PhysX solver/contact history, and
the real replay consequently reached tick 7794 with rear-right-knee position
-43.253145 deg and velocity -7.169843 deg/s.  The frozen P10 entry guard requires
-50.397598 +/- 2.0 deg and +23.585333 +/- 3.537800 deg/s, so it correctly stayed
in `WAIT_ENTRY`.

Tick 7552 is the frozen source ledger's all-wheel stop boundary.  It has three
supports, no body collision, root angular speed at most 0.03802 rad/s, and
maximum servo speed 11.4129 deg/s.  It supplies eight real solver ticks before
the tracked-servo waveform starts at tick 7560 and 27 ticks before the contact
latch.  The adjustment changes no guard, tolerance, frozen FSM byte, target
state, or in-episode behavior; it only extends reset-only causal physics replay.

### Result

Calibration run
`20260904T221856728216Z_g8b9ed0e197d6_c49f730ab0bcf_s1002_n1_phase-effective-entry-calibration`
remained fail-closed.  Position error improved from 7.144453 deg to
3.937853 deg, but rear-right-knee velocity was still -12.611137 deg/s instead
of the required positive rebound.  This proves that the local 25-tick pre-roll
does not reconstruct enough predecessor contact/solver history.

## P10 complete causal predecessor replay

- Changed factor: P10 reset-only physical replay anchor
- Old value: Trial043 wheel-stop tick 7552; 242 replay steps
- New value: Trial043 P09 phase-entry tick 6912; 882 replay steps
- Unchanged anchors: predecessor `P09/VERIFY_RESULT` tick 7776, controller
  `P10/WAIT_ENTRY` tick 7784, target entry tick 7794

The anchor is now derived from the frozen state-transition ledger rather than
from a calibrated numeric offset.  Reset reconstructs the complete P09 causal
trajectory, including its RR lift, crossing, top-load impact, verification,
and P10 wait window.  This remains reset-only state initialization; no guard,
tolerance, frozen controller byte, or in-episode behavior changes.

### Result

Calibration run
`20260904T222920227476Z_g66f0af803001_c5f04cb732d63_s1002_n1_phase-effective-entry-calibration`
also remained fail-closed.  The signed velocity recovered to +5.791487 deg/s,
but rear-right-knee position reached only -40.979061 deg.  Replaying a longer
open-loop history is therefore not monotonic evidence of reconstructing the
unserialized PhysX constraint warm-start state.

## P10 final P09 command-segment boundary

- Changed factor: P10 reset-only physical replay anchor
- Old value: Trial043 P09 phase-entry tick 6912; 882 replay steps
- New value: Trial043 tick 7560; replay ticks 7560--7793 (234 steps)
- Unchanged anchors: predecessor `P09/VERIFY_RESULT` tick 7776, controller
  `P10/WAIT_ENTRY` tick 7784, target entry tick 7794

Tick 7560 is the last pre-impact stable-window sample and the exact onset of
the final authored P09 tracked-servo command segment.  Its source root speed is
0.015735 m/s, root angular speed is 0.047008 rad/s, maximum servo speed is
11.59935 deg/s, and the support margin is 0.035237 m.  This is the final
single-factor natural-entry experiment before replacing open-loop solver-state
reconstruction with a versioned, source-proven post-entry curriculum reset.

### Result

Calibration run
`20260904T224818647154Z_g269c991569ae_c6fe99457bbaa_s1002_n1_phase-effective-entry-calibration`
remained fail-closed.  The final live P10 decision sample had rear-right-knee
position -43.617215 deg and velocity +0.550467 deg/s, outside both authored
entry bounds.  All 234 replay samples passed the live safety gate and retained
real PhysX contact provenance.  The failure isolates the missing state to
unserialized solver/constraint history rather than a disabled sensor or a
relaxed verifier, so no additional numeric pre-roll anchors will be promoted.

## P10 source-proven EXECUTE training reset

The replacement reset initializes Trial043 observation tick 7794, whose exact
WAIT_ENTRY-to-EXECUTE transition records all five authored entry guards passing.
The nested `wlr50_clean.phase_snapshot_source_proven_execute_restore.v1` contract
binds that complete transition row and its canonical SHA-256
`904c604cb85107f92dbcb75519cd7229ad7b3307c994fa20b7cdb27e8147278e`.
It applies source command 7794 once, including the source post-mapper RR-knee
bias, and advances one real PhysX tick before restoring the already-entered
EXECUTE controller. The first episode frame emits motion tick 1 at the live
post-prime observation. No local entry event is manufactured. The frozen
`motion_endpoint_issued` and `rl_workspace_geometry` completion guards remain
pending and authoritative. Full-sequence evaluations and videos still start P01.

All three consumers reject legacy P10 hybrid replay metadata. Fresh/reused
physical state, contact, mapper, safety, and runtime provenance checks remain
required. The canonical snapshot manifest is
`0fda5fb29e73315c2b751ee865d138566522bd40e8c3a94bb4daf34acee0557d`;
the bundle is
`39e61971ed7a8a76548f6af2e1af50ecf0248781740df9aa59751565ca3cc98e`.
The other twelve snapshot files remain byte-identical. The previous canonical
bundle is recoverable at
`C:\robotics_sim\wlr_robot\ppo_phase_snapshots_pre_source_proven_20260904`.

The five targeted test files passed in the locked Isaac Python 3.11 environment,
including its Torch tests; three platform-specific tests were skipped. The
frozen ledger remained 29/29 with zero mismatches. The broader unit suite is
currently blocked by tests loading the obsolete effective-entry contract;
that contract must be mechanically replaced from twelve real calibration runs
before checkpoint-dependent tests and training can qualify. No new physical
calibration or PPO training success is claimed by these software checks.

### Real calibration result

All P02--P13 calibrations on commit `86383b14c7db` passed on train seed 1002.
Each of the twelve independently launched runs contains one fresh-scene and
one reused-scene attempt, both successful, with unchanged frozen/runtime
identities. The P10 run is
`20260904T232822142384Z_g86383b14c7db_cb6d77cdc057d_s1002_n1_phase-effective-entry-calibration`.
The remaining phases were run under the same commit and configuration from
`20260904T232933323267Z` through `20260904T233600294073Z`.

The effective-entry builder validated all twelve runs and published the v2
contract with semantic SHA-256
`efe5bae44d67d0aa4e92735bd0718e5606b04faea35756aa3444c4de2c8a0701`
and file SHA-256
`bcd6900bd648698aa2fc44f53fe56607a23d0060d9f2dc4d5d3e2eb4f10c0393`.
It remains provisional until the independent seed-1003 holdout passes.
The old JSON and checksum are recoverable under
`C:\robotics_sim\wlr_robot\ppo_effective_entry_pre_source_proven_20260904`.

### Independent holdout aggregation repair

All twelve seed-1003 probes on `c17a61df195e` completed with both fresh and
reused attempts passing. The final aggregation nevertheless failed closed:
its call to `_attempt_passed` omitted the required snapshot-derived replay
window. No acceptance artifact or training authorization was produced.

The aggregator now reconstructs that window from the validated, pinned phase
snapshot and passes it to the same strict validator used by the live probe.
Rechecking the twenty-four recorded attempts with this call passed every
attempt, including source-proven P10. This is a software repair, not a new
physical acceptance; a new committed-runtime holdout is still required.

### Passed holdout and phase-zero rollout on 419279e

The independent seed-1003 holdout passed all twelve workers and twenty-four
fresh/reused attempts on `419279ee48d75f12aa3a09fa63579edf8311646f`. Its finalized
acceptance run is `20260905T000510668038Z_g419279ee48d7_c5dc85cd3cb31_s1003_n1_effective-entry-holdout-aggregation`.
The subsequent seed-1004 phase-zero run
`20260905T000550776168Z_g419279ee48d7_c5dc85cd3cb31_s1004_n1_phase-zero-residual-rollout`
passed all thirteen phase resets, 518 policy decisions and 4143 physics ticks,
with no failure reasons.

A pre-training delivery audit then found incompatible promotion-history roots:
training orchestration accepted only `runs/ppo_phase_v1`, whereas finalization
required the same immutable decision and paired artifacts below the delivery
output root. Placing them inside the final `metrics` directory would also
violate the input/output non-overlap guard. The new canonical history partition
is `outputs/ppo_phase_v1/validation_history/step_<global>`, separate from final
metrics. Hash, ancestry, managed export and path safety validation remain required.

The just-started seed-2001 baseline and its external sequential driver were
stopped before this repair. That unfinished baseline is not gate evidence.
All existing files were retained, including the completed holdout and phase-zero
evidence; the repaired committed runtime must obtain fresh acceptance before
training. No PPO optimizer execution or improved checkpoint is claimed yet.
