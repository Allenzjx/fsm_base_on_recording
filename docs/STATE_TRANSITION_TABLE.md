# State Transition Table

Every state follows WAIT_ENTRY → EXECUTE_MOTION → VERIFY_RESULT → RECOVERY/DONE. A 0.5 s no-progress watchdog applies only during EXECUTE_MOTION; reference response carry-over is not treated as a reason to settle or freeze.

| State | Live completion guards | Next | Maximum verify wait (s) | Transition rationale |
|---|---|---|---:|---|
| P01 | motion_endpoint_issued; fr_lift_entry_geometry | P02 | 0.250 | Reference-compatible load-transfer endpoint and live FR lift-entry geometry are established. |
| P02 | reference_like_active_lift; wheel_clearance_gain_or_air_history | P03 | 0.250 | FR reference-like active joint motion and increased wheel-bottom clearance make front-plane crossing observable. |
| P03 | leg_front_face_crossed_latched; leg_top_loaded_latched | P04 | 0.250 | Latched FR front-face clearance and top load make FL preparation safe. |
| P04 | motion_endpoint_issued; fl_lift_workspace_geometry | P05 | 0.250 | The FL unload workspace matches the v010 entry envelope without a force-only hard gate. |
| P05 | reference_like_active_lift; leg_front_face_crossed_latched; motion_endpoint_issued | P06 | 0.250 | FL has actively cleared the front plane and reached its placement posture; P06 continues wheel motion immediately, without a hold, to latch top load. |
| P06 | leg_top_loaded_latched; rear_pair_pre_edge_geometry | P07 | 0.250 | FL top load is latched and the rear pair has reached the measured pre-edge geometry. |
| P07 | motion_endpoint_issued; rear_entry_alignment | P08 | 0.250 | The rear-entry pose and wheel relation match the v010 first-rear entry envelope. |
| P08 | motion_endpoint_issued; rr_unload_compatible_geometry | P09 | 0.250 | The v010 FR+RL support geometry and active RR-unload preparation are present. |
| P09 | reference_like_active_lift; leg_front_face_crossed_latched; leg_top_loaded_latched | P10 | 0.250 | Latched RR active lift, front-face crossing, and top load prove RR-first traversal. |
| P10 | motion_endpoint_issued; rl_workspace_geometry | P11 | 0.250 | The FL+RR-related workspace is established for the second rear leg. |
| P11 | reference_like_active_joint_change; rl_unload_entry_geometry | P12 | 0.250 | The FR-directed transfer action has reached the v010 RL-unload entry pose. |
| P12 | reference_like_active_lift; leg_front_face_crossed_latched; leg_top_loaded_latched | P13 | 0.250 | Latched RL active lift, front-face crossing, and top load complete the rear order. |
| P13 | all_leg_front_face_crossings_latched; all_wheels_final_top_geometry; final_joint_pose_compatible; wheel_targets_zero; measured_wheel_velocity_stable_decay | TASK_COMPLETE | 1.000 | All four wheels and the body establish the v010 final top-crossing geometry and final pose; zero wheel targets are applied, then the new FSM must add a measured stable-decay debounce before SUCCESS. |
