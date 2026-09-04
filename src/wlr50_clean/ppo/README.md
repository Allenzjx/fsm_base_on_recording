# Phase-specific residual PPO

The frozen-FSM artifact modules use only the project's normal Python dependencies.
Writing `ppo_baseline_transitions.parquet` additionally requires `pyarrow`.
It is imported lazily by `EpisodeLogger.write_parquet`, so Isaac/FSM execution
does not acquire a training or tabular-runtime dependency.

Install the `ppo-artifacts` optional dependency before producing the final
dataset. The focused Parquet round-trip test requires `pyarrow` and validates
fixed-size 85D and 12D columns by reading the generated file back; it does not
silently skip the serializer contract.

The versioned v2 path adds a live Isaac/Isaac Lab residual environment, a true
batched environment, the supported RSL-RL 5.0.1 PPO runner, phase-entry reset
curriculum, deterministic physical evaluation, checkpoint promotion, and video
publication.  It uses one shared actor/critic with a 125D deployable
observation and a phase-masked, phase-scaled 12D residual.  The legacy 85D
observation and artifact exporter remain byte-compatible and separate from the
v2 training path.

All live entry points are exposed through `wlr50_clean.ppo.cli` and the
fail-fast PowerShell scripts in `scripts/`.  A training checkpoint is evidence
only until matched validation metrics pass the stability gate and an
independent locked-test aggregate authorizes the final improved name.

## Residual safety boundary

The Recording ±30% envelope is an advisory divergence diagnostic and may guide
policy initialization, but it is never used in the action projection path. It
is not subtracted as nominal "headroom" and does not scale or clip the requested
residual. The tanh output scale derives from configured physical action-range
spans. Runtime hard projection then uses the canonical actuator limits,
configured servo joint margins, wheel speed limits, exact versioned
`ppo_action_mask_full12` values from the motion contract, residual slew limits,
and explicit body-collision/wheel-only stop projection. The all-zero input has
a dedicated bitwise nominal fast path only when the previous projected
residual is also zero; otherwise it safely slews back to zero.

## Confirmed-Trial dataset export

`SelectedTrialStreamingExporter` reads `full12_commands_120hz.jsonl` and
`observation_120hz.jsonl` incrementally, validates every +1 tick and +1/120 s
step, checks every logged phase mask against the frozen contract, audits
nominal/residual/applied equality on every physics tick, and emits contiguous
15 Hz transitions. Construction of `SelectedTrialMetadata` fails
unless physical task success and external Trial selection are explicitly
confirmed and Recording divergence is classified diagnostic-only.

`export_to_logger` is read-only. `export_artifacts` is the separate explicit
write boundary for the JSONL, Parquet, and manifest files; it refuses to
overwrite any existing artifact. Do not call it until a Trial has been
selected and frozen by the physical-safety review.
It also writes `zero_residual_equivalence.json`, binding every 120 Hz source
command tick and every exported 15 Hz transition to the frozen nominal action.
