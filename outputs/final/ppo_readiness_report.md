# PPO readiness report

- Status: `FROZEN_FSM_BASELINE_READY_FOR_PPO` / `PPO_INTERFACE_READY`
- Selected nominal FSM: `trial_043_20260902_clean_v010`
- Physics / decision cadence: `120 Hz / 15 Hz`
- Observation dimension: `85`
- Nominal / residual dimensions: `12 / 12`
- State order: `P01-P13`; rear-leg order: `RR_FIRST`
- Macro phase IDs: `P01=1` through `P13=13` (frozen)
- Phase masks: frozen for all 13 states
- Residual application: `nominal_fsm_action + projected_residual`
- Hard bounds: actuator limits, joint margins, wheel limits, phase masks, rate limits, BODY-collision safety, and wheel-only-climb safety
- Recording ±30% envelope: diagnostic / initial-scale suggestion only; not a projection bound and not a success gate
- Reset: a non-negative integer seed is required; nominal evaluation uses the frozen environment and all randomization hooks are disabled
- Episode end: SUCCESS and six physical/numerical hard failures terminate; TIMEOUT truncates at 200 s; REFERENCE_CONFORMANCE remains diagnostic-only
- Training enabled: `false` (`PPO_TRAINING_NOT_STARTED`)

The separate selected-trial exporter owns the full 120 Hz zero-residual proof and 15 Hz JSONL/Parquet dataset. This publication does not claim a trained policy.
