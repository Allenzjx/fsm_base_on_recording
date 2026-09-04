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
