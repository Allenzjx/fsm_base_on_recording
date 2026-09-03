# 30% Existing-Trial Readjudication

All numeric results are recomputed from immutable observation_120hz.jsonl, full12_commands_120hz.jsonl, state transitions, physical ledgers, and the frozen v010 telemetry contract. Old manifest conformance booleans are not reused.

Wheel channels use velocity semantics: endpoint/delta and trajectory-position RMSE are reported diagnostics but are not position-style gates. Command/measured average and peak velocity plus command/actual wheel integrals are gates. Servo endpoint, delta, command/measured velocity, duration, and phase-normalized trajectory RMSE are gates.

Atomic source events require one complete full12 articulation write on one 120 Hz physics tick. Overlap channel sets are exact; onset and duration use the active numeric tolerance.

## Absolute floors

- `joint_endpoint_delta`: 2 deg; source: `existing_v010_motion_contract_joint_endpoint_delta_floor`
- `servo_velocity`: 1 deg_s; source: `existing_validated_similarity_contract`
- `wheel_velocity`: 0.05 rad_s; source: `existing_validated_similarity_contract`
- `wheel_integral`: 0.05 rad; source: `existing_validated_similarity_contract`
- `trajectory_scale`: 2 deg; source: `existing_v010_phase_normalized_rmse_scale_floor`
- `overlap_timing`: 0.00833333 s; source: `one_120hz_physics_tick`
