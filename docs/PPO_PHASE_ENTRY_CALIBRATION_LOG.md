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

### Float32 actuator-effect gate repair

The seed-1003 holdout was repeated successfully on `e203c78ba556c99b6a3bc1279e7f7ee3167fd932`:
`20260905T002256476224Z_ge203c78ba556_c5dc85cd3cb31_s1003_n1_effective-entry-holdout-aggregation`.
Before starting the long prerequisite sequence, a further audit found that the
old single-environment bounded-smoke pattern used normalized magnitudes of
`1e-9` (and `1e-12` in P13). A nonzero Python projected residual was therefore
not evidence of a representable change in the float32 actuator targets. Such
coverage is not accepted as a physical nonzero gate.

The smoke-only diagnostic now uses 0.5--1% of a configured phase scale on one
per-phase active servo. P13 receives a six-decision bipolar pulse and then zero
for final settling. This excitation is never added to a trained policy. Frozen
FSM commands, safety projection, completion guards, and limits are unchanged.

An opt-in, read-only audit inspects the real float32 target buffers after the
existing atomic articulation dispatch. It reconstructs the same-tick
counterfactual without the current PPO residual using the frozen mapping,
standing offsets, controller bias, and previous final drive targets. It neither
advances the stateful mapper again nor performs another target write. Default
training and evaluation do not pay for these extra GPU buffer reads.

Gate B now requires actual representable target changes attributable to each
phase's own active policy request across P01--P13. Incoming transition-bridge
ticks, quantization-zero requests, safety-zero projections, and slew-swallowed
requests do not count. Audit records bind the actual applied-minus-nominal
logical delta, dispatch tick, source phase, request mask, and changed targets.
Unit regressions cover the real projector, mapper, adapter, and evidence writer
with float32 buffers; these tests do not substitute for a new live smoke run.

This is a PPO-only prerequisite correction before any optimizer execution.
The newly committed runtime must again pass holdout, zero, nonzero, reset, and
vector gates before training. All old evidence and interrupted runs remain
preserved; no physical or training success is inferred from this code repair.

### Vector reward interface correction before first training

All twelve holdout workers and twenty-four fresh/reused attempts passed on
`a563defce132b33d0133c2baf2ee05cfefe21d7c`; the finalized acceptance run is
`20260905T004538305299Z_ga563defce132_c5dc85cd3cb31_s1003_n1_effective-entry-holdout-aggregation`.

A concurrent read-only review of the first vector-training step identified a
missing interface update: `VectorizedRslResidualEnv` did not pass the required
termination reason and controller-blocked flag to `ResidualEpisodeEnv._reward`.
The single-environment path already supplied these fields. Existing vector
fixtures replaced the reward method with its old signature, hiding the mismatch
even in official-library optimizer tests.

The vector caller now supplies its already-computed per-row termination reason
and blocked flag, without changing reward formulas or termination decisions.
Regressions exercise the real reward path and make remaining light-weight test
substitutes enforce the keyword-only production interface. This repair precedes
the first live PPO optimizer execution; a new committed-runtime prerequisite
sequence is required before training.

### Fresh physical evidence on 36a0d57 and failed genuine nonzero smoke

Commit `36a0d57eb96a03cb8b285f04a16602fafb15d464` passed all twelve seed-1003
holdout workers (twenty-four fresh/reused attempts), then the seed-1004
phase-zero rollout (13 resets, 518 decisions, 4143 physics ticks). The holdout
acceptance is in run
`20260905T010051886319Z_g36a0d57eb96a_c5dc85cd3cb31_s1003_n1_effective-entry-holdout-aggregation`.

Five independently initialized full FSM validation episodes, seeds 2001--2005,
all completed P01--P13 in 107.86666666666666 simulated seconds. Each recorded
12944 physics ticks and 1618 policy decisions; all 64720 dispatched zero-input
ticks were bitwise equivalent to frozen nominal commands. Body collision,
wheel-only climb, safety abort, in-episode root writes, and Recording runtime
access were zero. Their managed baseline metrics export run is
`20260905T015041564002Z_g36a0d57eb96a_c5dc85cd3cb31_s2001_n1_baseline-fsm-evaluation-export`.
The three original exported files under `outputs/ppo_phase_v1/metrics` remain
immutable evidence of this runtime, not evidence of future source revisions.

Genuine bounded-smoke run
`20260905T015053495405Z_g36a0d57eb96a_c5dc85cd3cb31_s1001_n1_nonzero-residual-smoke`
failed closed at P10 WAIT_ENTRY after 65.36666666666666 simulated seconds,
981 decisions and 7844 physics ticks. All ticks had valid actuator target-effect
audits and phases P01--P10 each changed real float32 targets through their own
policy requests. No safety violation occurred, but P11--P13 were not reached.
The frozen P10 right-rear-knee entry guard rejected a 9.765684946973757 degree
position error and a -29.648617057219695 degree/second velocity error. This is
an actual physical failure, not a successful gate or PPO training result.

The first smoke-only follow-up changes one design factor: P01--P12 now repeat
the same six-decision zero-mean bipolar waveform already used for P13, replacing
the previous long-lived positive offset. Its amplitude remains 0.5--1% of the
unchanged phase scale; channel selection, 15 Hz cadence, P13 terminal settling,
all safety logic, full-success requirement, all thirteen own-phase physical
effects, and all twelve nonzero transition handoffs remain required. This
diagnostic pattern is never added to a trained actor. A new live run is needed
to assess it; unit tests cannot establish physical success.

### Pre-training cadence and failed-training checkpoint retention

Before any real Isaac PPO optimizer execution, the launcher audit identified
that generic `CliArgs --policy-decisions` was rejected by the intentional
semantic argument lock. The training wrapper now has a validated, named
`-PolicyDecisions` option owned by its base arguments; the generic lock remains.

An independent horizon audit found that restarting full-episode training every
10000 global decisions gave only 85.33/42.67/25.60 seconds per environment at
N=8/16/32, shorter than the actual successful baseline. The revised cadence is
derived from the unchanged profile: full training uses respectively 4x25000,
2x50000, or 1x100000 requested decisions, each 25 PPO iterations and 3200
decisions per environment (213.33 seconds of available continuous horizon).
Full requested budget stays 100000 and actual rounded budget is 102400. Smoke
and single-environment phase curriculum retain 10000-decision validation chunks.
Synchronous peer resets can still shorten actual trajectories; a long enough
configured window is not itself success evidence.

Random exploration outcomes (missing full-episode phase visits or no stochastic
SUCCESS) become explicit checkpoint diagnostics, so real failed candidates can
be saved and deterministically evaluated. Structural telemetry validity, reward
audits, checkpoint round-trip, independent deterministic evaluation, promotion,
locked-test, and final delivery conditions are not relaxed. Initial/last or
unverified candidates must never be called improved.

Managed consumers bind every baseline to the complete current runtime, not
only frozen FSM bytes. Consequently the corrected runtime needs fresh baseline
workers. Explicit optional metrics-directory parameters preserve the previous
three canonical baseline exports unchanged and allow subsequent formal exports
and delivery under a separate directory within `outputs/ppo_phase_v1`. No old
run is relocated, rewritten, or relabeled as a new-runtime acceptance.

Pre-freeze verification: the unified unit rerun passed 1340 tests with nine
skips (JUnit: `C:\robotics_sim\wlr_robot\ppo_cadence_bipolar_final_full_unit_20260905.xml`).
The preceding run had two outdated single-environment smoke fixtures, now
updated to exercise the configured single-environment phase-curriculum reset
gate without disabling its actual cadence or soft-reset checks. The final
checkpoint telemetry wiring follow-up passed 128 tests with two skips
(`C:\robotics_sim\wlr_robot\ppo_final_training_telemetry_checkpoint_tests_20260905.xml`):
raw telemetry, derived outcomes and cadence are all stored in checkpoint infos
and its sidecar, verified after reload, and checked against training_result by
the orchestration consumer. A real PowerShell AST-driven regression also checks
the matching/missing/extra promotion gate-name branches under StrictMode;
matching empty differences must be wrapped as an array before reading Count.
None of these unit tests is counted as live Isaac PPO training evidence.

### Bipolar 1%-peak smoke result and next amplitude-only sensitivity probe

The new live run on `0d501431d5a9ed020fe09b8706bc91c1ad4239cb`,
`20260905T023142892988Z_g0d501431d5a9_c5dc85cd3cb31_s1001_n1_nonzero-residual-smoke`,
also failed at the frozen P10 WAIT_ENTRY guard, at 65.36666666666666 seconds,
981 decisions and 7844 physics ticks. Its own-phase real float32 target effects
cover P01--P10 and all tick audits are complete. Body collision, wheel-only
climb, safety abort, in-episode root writes and Recording access remain zero.
RR-knee entry position error was 9.157316582506134 degrees and signed velocity
error was -27.354560140027044 degrees/second. Removing the probe's DC offset
alone did not restore full-task success; this gate is explicitly failed.

The next experiment changes only the artificial smoke amplitude by a factor
of 1/100: the same bipolar sequence now uses 0.005--0.01% of each configured
phase residual scale, including the masked-channel probe. PPO action scales,
optimizer exploration, phase selection, cadence, P13 pulse duration, and all
success/guard/audit requirements remain unchanged. Standard float32 rounding
tests confirm representability at native target magnitudes up to a full
revolution, but those tests do not establish an actual actuator dispatch effect;
the live audit must still prove it separately in every phase. This is only a
backend sensitivity probe, not the independently thresholded trained-policy
activity requirement (which still includes 1% of the configured phase scale).

Independent stream comparison found identical initial observations and exactly
equal nominal Full12 commands through tick 7794. P09 phase/lifecycle timing was
also identical; the first nominal/lifecycle divergence at tick 7795 followed
the baseline entering P10 EXECUTE while the smoke remained WAIT_ENTRY. All
7844 smoke dispatch audits verified actual mapping and setter equality, with
the correct reset tick offset of 179. Combined actuator bias equalled frozen
controller bias plus projected residual; the controller's bias was zero in all
observed smoke ticks, and the P09 feedback trigger was false in both runs.
These checks found no dropped nominal correction or command-timing bug. The
lower-amplitude follow-up passed 188 focused tests before live execution.
