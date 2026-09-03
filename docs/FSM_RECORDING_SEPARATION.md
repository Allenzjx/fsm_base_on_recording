# FSM / Recording Separation

## Two different execution systems

The v010 Recording and the production FSM are evidence-compatible, but they
do not advance in the same way.

The Recording runner reads `reference/v010/accepted_steps.jsonl` offline and
dispatches the accepted events according to their recorded timing. Recording
event time and the event cursor determine progression.

The production FSM never opens `accepted_steps.jsonl`, never owns a Recording
cursor, and never runs a Recording event loop. It loads only the compact
recording-derived motion contract and the fixed P01--P13 state specification.
Within a state, time may generate a trajectory, implement a debounce, or end a
bounded timeout. Time alone cannot mark the state complete.

## Live completion evidence

Every FSM transition is evaluated from the current Isaac observation and
latched sensor history. Depending on the phase, this includes:

- actual joint position and velocity;
- applied and measured wheel velocity;
- exact contact-pair history classified as BODY, LEG, or WHEEL;
- active hip/knee motion and wheel-bottom clearance;
- AIR, obstacle-front-plane crossing, and top-contact/top-geometry latches;
- body/chassis collision status;
- final obstacle-relative wheel/body geometry and stable wheel decay.

Recovery is bounded, phase-local, and remains conditional on the failed live
guard. It is not a replay cursor.

## Acceptance layers

Final adjudication keeps three independent results:

1. `TRIAL_VALIDITY` verifies a continuous physics run, frozen environment and
   asset identity, an unstitched video, and zero forbidden control operations.
2. `TASK_SUCCESS` verifies complete physical traversal, no BODY collision, no
   wheel-only climb, and a stable final state.
3. `QUALITY_AND_REFERENCE_DIAGNOSTICS` reports Recording differences.

The 30% Recording tolerance belongs only to the third layer. Exceeding it
produces `REFERENCE_DIVERGENCE_WARNING`; it cannot revoke physical task
success, prevent video publication, or block freezing the nominal PPO
baseline.

## PPO boundary

Residual PPO receives the frozen FSM nominal action and may only add a bounded
residual through the stable adapter. Its hard projection is based on actuator
limits, joint safety margins, wheel limits, phase masks, rate limits, and the
body-collision/wheel-only-climb safety signals. Recording divergence may
suggest an initial scale, but it is not action headroom. The FSM retains phase
order and all task-safety detectors.

The final separation audit binds this design claim to a source scan and to the
selected immutable Trial's `runtime_raw_recording_access = false` evidence.
