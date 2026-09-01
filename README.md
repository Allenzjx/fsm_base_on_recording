# WLR 50 mm Recording-Shaped Sensor FSM (Clean v1)

This repository is a clean-room implementation of the WLR robot's 50 mm
obstacle traversal controller. Its sole motion reference is
`v010_20260806_220745_363972_manual` with rear-leg order `RR_FIRST`.

The production FSM lives only in the `wlr50_clean` namespace. It uses compact,
recording-derived state motion contracts, but state transitions are decided from
live observations. Production runtime code is prohibited from opening the
recording event stream.

Run the clean-room verifier before any Isaac Sim launch:

```powershell
& .\scripts\prepare_clean_project.ps1
```

The frozen reference is validated independently with:

```powershell
& .\scripts\validate_reference.ps1
```

The runtime contract contains ordered P01–P13 and a small phase-relative
waypoint set. Relative time shapes the continuous 120 Hz action inside a state;
it cannot complete or advance a state. The 15 Hz controller advances only after
the matching live geometry/contact/history guard is observed.

The environment is fixed at 120 Hz physics, 15 Hz decisions/rendering, the
hash-locked robot USD, and the frozen 50 mm scene. Runtime APIs deliberately do
not include root-state writes, teleportation, force/impulse application, stage
saving, or raw Recording access.

The final acceptance condition is one continuous physical Isaac Sim trial that
completes P01-P13 without a body-obstacle collision or wheel-only climb, followed
by validated Recording, FSM, and side-by-side MP4 outputs.
