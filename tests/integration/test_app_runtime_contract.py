from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.infrastructure.app_runtime import (
    MAX_CONTROL_SECONDS,
    RENDER_STRIDE,
    _physical_acceptance_failures,
    _config_from_args,
    _validate_initial_observation,
    render_due,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src" / "wlr50_clean" / "infrastructure" / "app_runtime.py"
SCRIPT = ROOT / "scripts" / "run_fsm_trial.ps1"


def _args(tmp_path: Path, **overrides):
    values = {
        "run_dir": tmp_path / "new-run",
        "fsm": ROOT / "configs" / "fsm_states.yaml",
        "motion_contract": ROOT / "configs" / "recording_motion_contract.json",
        "max_control_seconds": MAX_CONTROL_SECONDS,
        "settle_seconds": 1.5,
        "warmup_renders": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runtime_imports_without_isaac_and_caps_duration(tmp_path: Path) -> None:
    config = _config_from_args(_args(tmp_path))
    assert config.maximum_control_ticks == 24_000
    assert config.settle_ticks == 180
    with pytest.raises(ValueError, match="200"):
        _config_from_args(_args(tmp_path, max_control_seconds=200.001))
    with pytest.raises(ValueError, match="1.5"):
        _config_from_args(_args(tmp_path, settle_seconds=1.0))
    with pytest.raises(ValueError, match="locked v010"):
        _config_from_args(_args(tmp_path, motion_contract=tmp_path / "other.json"))
    config.run_dir.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        _config_from_args(_args(tmp_path))


def test_exact_render_cadence_is_one_per_eight_completed_physics_ticks() -> None:
    assert RENDER_STRIDE == 8
    assert [step for step in range(1, 33) if render_due(step)] == [8, 16, 24, 32]
    source = RUNTIME.read_text(encoding="utf-8")
    assert source.count("scene.sim.step(render=False)") == 2  # settle and continuous control
    assert source.count("recorder.before_render(") == 1
    assert source.count("recorder.after_render()") == 1
    assert "sim.step(render=True)" not in source
    assert "simulation_app.update()" in source
    assert source.count("simulation_app.update()") == 1
    assert "setter((VIDEO_WIDTH, VIDEO_HEIGHT))" in source
    assert "getter()" in source
    assert source.index("_configure_active_viewport()") < source.index("for _ in range(config.warmup_renders)")
    assert "ActiveViewportVideoRecorder(config.run_dir)" in source
    assert "viewport_provider=" not in source
    assert "create_live_sensing_backends(\n                sim=sim, robot=robot\n            )" in source
    assert source.index("_validate_initial_observation(current_observation)") < source.index("recorder.start()")


def test_runtime_never_uses_quality_diagnostics_as_a_success_veto() -> None:
    result_layers = {
        "trial_validity": {"checks": {"continuous_physics": True}},
        "task_success": {"checks": {"physical_traversal": True}},
        "quality_and_reference_diagnostics": {
            "checks": {
                "within_30_percent": False,
                "feedback_correction_reference_bounded": False,
                "final_wheel_targets_zero": False,
                "measured_wheel_velocity_stable_decay": False,
            },
            "blocks_task_success": False,
        },
    }

    assert _physical_acceptance_failures(result_layers) == []


def test_pre_simulation_app_import_boundary_and_lazy_runtime_imports() -> None:
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
    top_import_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_import_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            top_import_roots.add((node.module or "").split(".")[0])
    assert top_import_roots <= {"argparse", "json", "math", "sys", "traceback", "dataclasses", "pathlib", "typing"}
    source = RUNTIME.read_text(encoding="utf-8")
    construct = source.index("launcher = AppLauncher(args)")
    initial_update = source.index("simulation_app.update()", construct)
    lazy_run = source.index("return _run_live(config, simulation_app)", initial_update)
    assert construct < initial_update < lazy_run
    assert "accepted_steps" not in source.lower()
    assert "semantic_segments" not in source.lower()
    assert "root_state" not in source.lower().replace("root_state_write_count", "")
    assert "total_drive_bias = tuple(" in source
    assert "drive_feedback_bias_full12=total_drive_bias" in source
    assert "drive_feedback_bias_full12[:8]" not in source


def test_initial_observation_rejects_unverified_pairs_and_geometry() -> None:
    def pair(verified=True):
        return SimpleNamespace(pair_verified=verified)

    contacts = {
        f"body_{index}": SimpleNamespace(ground=pair(), obstacle=pair())
        for index in range(13)
    }
    wheels = {
        f"wheel_{index}": SimpleNamespace(
            name=f"wheel_{index}",
            geometry_verified=True,
            center_w_m=(0.0, 0.0, 0.05),
            bottom_w_m=(0.0, 0.0, 0.0),
        )
        for index in range(4)
    }
    observation = SimpleNamespace(
        contacts=contacts,
        wheels=wheels,
        bodies={f"body_{index}": object() for index in range(13)},
        center_of_mass=SimpleNamespace(
            valid=True,
            included_bodies=tuple(contacts),
        ),
        all_finite=True,
        data_quality=(),
    )
    _validate_initial_observation(observation)
    contacts["body_0"].obstacle = pair(False)
    with pytest.raises(RuntimeError, match="unverified"):
        _validate_initial_observation(observation)


def test_launcher_script_is_single_instance_gui_and_clean_pythonpath() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "prepare_clean_project.ps1" in source
    assert "Get-CimInstance Win32_Process" in source
    assert "Another Isaac/Kit process" in source
    assert '$env:PYTHONPATH = $CleanSource' in source
    assert '$env:HEADLESS = "0"' in source
    assert "-P -m wlr50_clean.infrastructure.app_runtime" in source
    assert "Remove-Item" not in source
