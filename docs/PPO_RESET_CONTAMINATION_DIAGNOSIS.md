# Reused-scene reset contamination diagnosis

## Observed failure

The first five-episode zero-residual diagnostic used one Isaac scene for all
episodes.  Episode 0 completed P01--P13 in 107.8666667 s.  Episodes 1--4 each
stopped at P10 `WAIT_ENTRY` after 65.3666667 s.  None of the five episodes had
a body collision, wheel-only climb, fall, numerical failure, or Recording
runtime access.  This run is retained as a failed diagnostic and is not Gate A
acceptance evidence.

Run directory:

`runs/ppo_phase_v1/zero-residual-live/20260903T223331801662Z_g7d6bfda0da59_cfd4a3c90b203_s2001_n1_zero-residual-live`

## Determinism evidence

After removing only the seed field, the policy traces for reused-scene
episodes 1 and 2 have the same canonical SHA-256:

`cc27fcb15bddd9ae5b3f2df757d72ea59116a3a41a3f45edc54e7f3db5b322ac`

The successful fresh-scene episode and the reused-scene episodes have equal
control/state/action fields through decision 973.  At decision 974 (65.0 s),
the fresh episode enters P10 `EXECUTE_MOTION`; the reused episodes remain in
`WAIT_ENTRY` and hit the configured 0.5 s progress watchdog.  Seeds are only
recorded by this deterministic backend and do not drive randomization.

P10's state-sensitive entry gate includes the rear-right knee position
(`-50.3976 deg +/- 2.0 deg`) and signed velocity
(`+23.5853 deg/s +/- 3.5378 deg/s`) with a two-physics-tick alignment delay.
The old trace did not persist the full blocker payload, so it cannot prove
whether position or velocity was the last failing predicate.  The unique
velocity-aligned entry condition is the highest-confidence cause by
elimination.

## Root cause boundary and correction

Episode 0 constructs a fresh stage.  The old reuse path called global
`SimulationContext.reset()` and then restored only articulation/contact state.
The stage, PhysX scene, sensor views, geometry caches, and session-layer
actuator-limit authorship remained from the first episode, so that reset was
not physically equivalent to a fresh process.

Two corrections are applied outside the frozen FSM:

1. Acceptance and paired evaluation launch one fresh Isaac process per seed.
2. Training reuse performs an Isaac Lab state-write soft reset, resets contact
   history, synchronizes with `SimulationContext.forward()`, and takes no
   physics step at the reset boundary.  It no longer calls global simulation
   reset.

The frozen controller, FSM graph, sensing implementation, infrastructure,
environment lock, and motion contract are unchanged.  A reused-scene soft
reset must still pass live equivalence testing before it is accepted for PPO
training.
