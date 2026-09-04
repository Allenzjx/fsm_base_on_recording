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
