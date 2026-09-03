# Physical-success readjudication

Recording divergence is advisory in this report and never vetoes Layer B task success.

- Runs scanned: 44
- Complete-traversal candidates: 18
- Valid physical successes: 18
- Selected: trial_043_20260902_clean_v010

## Complete traversal candidates

| Trial | Validity | Geometry | Body collision | Wheel-only | Stable | Recoveries | Duration s | Max divergence % | New result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| trial_007_20260901_clean_v010 | VALID | True | False | False | False | 0 | 70.866667 | n/a | TASK_SUCCESS |
| trial_008_20260901_clean_v010 | VALID | True | False | False | False | 0 | 71.000000 | n/a | TASK_SUCCESS |
| trial_009_20260901_clean_v010 | VALID | True | False | False | False | 1 | 108.133333 | n/a | TASK_SUCCESS |
| trial_010_20260901_clean_v010 | VALID | True | False | False | True | 0 | 88.866667 | 52.750139 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_011_20260901_clean_v010 | VALID | True | False | False | False | 0 | 89.066667 | 52.750139 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_012_20260901_clean_v010 | VALID | True | False | False | False | 0 | 70.866667 | n/a | TASK_SUCCESS |
| trial_013_20260901_clean_v010 | VALID | True | False | False | False | 0 | 70.866667 | n/a | TASK_SUCCESS |
| trial_019_20260902_clean_v010 | VALID | True | False | False | True | 1 | 107.866667 | 52.750139 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_021_20260902_clean_v010 | VALID | True | False | False | True | 1 | 108.133333 | 52.750139 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_022_20260902_clean_v010 | VALID | True | False | False | False | 1 | 108.133333 | n/a | TASK_SUCCESS |
| trial_023_20260902_clean_v010 | VALID | True | False | False | False | 1 | 108.133333 | n/a | TASK_SUCCESS |
| trial_024_20260902_clean_v010 | VALID | True | False | False | True | 1 | 108.466667 | 107.245497 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_025_20260902_clean_v010 | VALID | True | False | False | True | 1 | 108.133333 | 52.750139 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_036_20260902_clean_v010 | VALID | True | False | False | True | 0 | 89.133333 | 94.601439 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_038_20260902_clean_v010 | VALID | True | False | False | False | 1 | 108.600000 | n/a | TASK_SUCCESS |
| trial_039_20260902_clean_v010 | VALID | True | False | False | True | 1 | 89.800000 | 75.385609 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |
| trial_042_20260902_clean_v010 | VALID | True | False | False | False | 0 | 70.933333 | n/a | TASK_SUCCESS |
| trial_043_20260902_clean_v010 | VALID | True | False | False | True | 1 | 107.933333 | 36.458637 | TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING |

## Selection rule

Section 8 explicitly accepts Trial 043 once its raw Layer A and Layer B conditions pass. If Trial 043 is ineligible, the Section 9 fallback order is: no body collision; no wheel-only climb; complete traversal; exact environment; continuous video; no forbidden control; stable final pose; fewer recoveries; shorter runtime; lower maximum Recording divergence. Trial number is used only as a deterministic final fallback.

## Selected physical success

`trial_043_20260902_clean_v010` is selected. Its Layer B result is `SUCCESS`; its combined reporting label is `TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING`.

Its maximum applicable reference divergence is `36.458636795%` at `P09/rear_left_ankle/measured_peak_velocity_error_percent`. This is diagnostic only.

## Explicit priority audits

- Trial 043: `TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING`; raw P01-P13 and endpoint geometry = `True` / `True`.
- Trial 044: `INCOMPLETE_CONTROLLER_BLOCKED`; completed states: `P01,P02,P03,P04,P05,P06,P07,P08,P09`. Reference percentages do not define this incomplete result.
