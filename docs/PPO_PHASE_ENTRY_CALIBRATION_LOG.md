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
