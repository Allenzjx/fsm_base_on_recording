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
