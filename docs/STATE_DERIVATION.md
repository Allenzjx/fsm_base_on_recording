# State Derivation

All nominal motion is derived offline from the single frozen v010 RR_FIRST recording. Production execution loads only the compact relative motion projection; live sensor evidence, never an event cursor or absolute recording time, advances the graph.

| Phase | State | Physical purpose | Reference steps | Active duration (s) | Completion event |
|---|---|---|---:|---:|---|
| P01 | INITIAL_LOAD_TRANSFER_FR | Build the v010 diagonal/load structure and approach needed before active FR lift. | 1,2 | 13.267 | fr_lift_entry_ready |
| P02 | FR_ACTIVE_LIFT | Actively flex the FR leg until its wheel-bottom clearance increases. | 3 | 0.467 | fr_active_lift_observed |
| P03 | FR_CROSS_AND_PLACE | Carry FR through the front plane and establish FR obstacle-top contact. | 4 | 1.200 | fr_top_loaded |
| P04 | FL_UNLOAD_PREP | Change the v010 support geometry so FL can be actively lifted. | 5,6 | 4.800 | fl_lift_entry_ready |
| P05 | FL_ACTIVE_LIFT_AND_PLACE | Actively lift FL, clear the front plane, and commit the placement posture; top load latches during the immediately continuous P06 wheel advance. | 7,8,9 | 9.733 | fl_front_cleared_and_placement_committed |
| P06 | FRONT_PAIR_ADVANCE | Continue the v010 wheel relation without a hold, latch FL obstacle-top load, and advance until the rear pair nears the edge. | 10 | 25.533 | fl_top_loaded_and_rear_pair_pre_edge |
| P07 | REAR_ENTRY_ALIGNMENT | Align the live front/rear geometry to the v010 first-rear entry without leaving the 15 percent envelope. | 11,12,13 | 1.667 | rear_entry_aligned |
| P08 | FIRST_REAR_LOAD_TRANSFER | Load FL and prepare the FR+RL support structure that unloads RR first. | 14 | 0.400 | rr_unload_ready |
| P09 | RR_ACTIVE_LIFT_AND_PLACE | Actively lift RR, clear the front plane, and finish the v010 placement action that establishes RR top load. | 15,16,17,18 | 7.200 | rr_top_loaded |
| P10 | SECOND_REAR_TRANSFER_PREP | Establish the v010 FL+RR-related support/workspace for RL unloading. | 19 | 0.133 | rl_workspace_ready |
| P11 | TRANSFER_TOWARD_FR | Execute the v010 FR-directed load-transfer action before RL lift. | 20 | 0.467 | rl_unload_ready |
| P12 | RL_ACTIVE_LIFT_AND_PLACE | Actively lift RL, clear the front plane, and place RL on the obstacle top. | 21,22,23,24 | 4.667 | rl_top_loaded |
| P13 | FINAL_ADVANCE_AND_RECOVERY | Advance the whole body past the obstacle, recover the v010 final posture, and stop all wheels. | 25,26 | 17.800 | final_clear_and_stopped |

## Bounded nominal response shaping

P03 scales only the rear-left wheel excursion by -14.9%. This is the first strict Trial025 measured-response correction: it remains inside the v010 command envelope, preserves the source same-tick Full12 launch, and requires no runtime Recording access. Trial036 then identified the downstream support consequence: P06 restores the same 0.109068 rad (about 5.45 mm surface travel) on that already-active wheel with a +1.4238642298% scale, so the later RL support location remains reference-compatible without changing any phase timing or atomic relation.
