from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli


def _passing_audit():
    return {
        "within_five_percent_smoke_amplitude": True,
        "within_one_percent_smoke_amplitude": True,
        "mask_honored_when_exercised": True,
        "rate_limit_tick_count": 1,
        "phase_transition_bridge_count": 12,
        "phase_transition_handoff_hold_count": 12,
        "body_collision_detector_operational": True,
        "wheel_only_climb_detector_operational": True,
        "nonzero_residual_phases": [f"P{index:02d}" for index in range(1, 14)],
        "actuator_target_effect_audit_required": True,
        "actuator_target_effect_audit_complete": True,
        "own_policy_actuator_target_effect_phases": [f"P{index:02d}" for index in range(1, 14)],
    }


@pytest.mark.parametrize("case, expected", (
    ("physical_all_phases", 0),
    ("old_quantization_only", 2),
    ("missing_proof", 2),
    ("one_phase_missing", 2),
    ("exceeds_one_percent", 2),
    ("unsafe", 2),
    ("over_200_seconds", 2),
    ("missing_handoff", 2),
))
def test_bounded_smoke_acceptance_requires_physical_own_phase_effect_without_relaxing_gates(
    tmp_path, monkeypatch, case, expected
):
    audit = _passing_audit()
    if case == "old_quantization_only":
        # The historical tiny pulse populated all logical nonzero phases even
        # when float32 actuator targets were unchanged in every phase.
        audit["maximum_absolute_normalized_policy_action"] = 1.0e-9
        audit["own_policy_actuator_target_effect_phases"] = []
    elif case == "missing_proof":
        for key in tuple(audit):
            if "actuator_target_effect" in key:
                del audit[key]
    elif case == "one_phase_missing":
        audit["own_policy_actuator_target_effect_phases"].remove("P13")
    elif case == "exceeds_one_percent":
        audit["within_one_percent_smoke_amplitude"] = False
    elif case == "missing_handoff":
        audit["phase_transition_handoff_hold_count"] = 11
    result = {
        "all_success": True,
        "body_collision_count": int(case == "unsafe"),
        "wheel_only_climb_count": 0,
        "safety_abort_count": 0,
        "all_under_200_s": case != "over_200_seconds",
        "episodes": [{"action_projection_audit": audit}],
    }
    monkeypatch.setattr(cli, "_run_live_episodes", lambda *args, **kwargs: result)
    args = SimpleNamespace(residual_mode="bounded-smoke", episode_count=1, run_dir=tmp_path)
    # Keep the real rate-limit probe and real acceptance writer in the chain.
    assert cli._baseline_or_gate(args, object()) == expected
    accepted = json.loads((tmp_path / "acceptance.json").read_text())
    assert accepted["passed"] is (expected == 0)
    assert accepted["action_projector_rate_limit_probe"]["applied_to_robot"] is False
    if case == "old_quantization_only":
        assert accepted["mode_specific_checks"]["nonzero_residual_covers_p01_p13"] is True
        assert accepted["mode_specific_checks"]["own_policy_actuator_target_effect_covers_p01_p13"] is False


@pytest.mark.parametrize("mode", ("zero", "bounded-smoke"))
def test_live_episode_enables_audit_only_for_smoke_and_binds_request_before_step(
    tmp_path, monkeypatch, mode
):
    from wlr50_clean.ppo import isaac_fsm_backend, live_stream_writer, residual_direct_env

    calls = []
    raw = (0.001,) * 12
    mask = (1,) * 8 + (0,) * 4
    frame = SimpleNamespace(
        state_id="P13",
        sim_time_s=1.0,
        physics_tick=120,
        termination_signals=SimpleNamespace(
            body_collision=False, wheel_only_climb=False, fall=False,
            nan_inf=False, hard_joint_limit=False, physics_explosion=False,
        ),
    )

    class Backend:
        def __init__(self, app, **kwargs):
            assert kwargs["audit_actuator_target_effect"] is (mode == "bounded-smoke")
            self.request = None
            calls.append("backend")

        def set_actuator_target_audit_request(self, **kwargs):
            self.request = kwargs
            calls.append("request")

    class Env:
        def __init__(self, backend, **kwargs):
            self.backend = backend
            self.frame = frame
            self.phase_actions = SimpleNamespace(mask_for=lambda phase: mask)
            self._bounded_smoke_phase_decisions = {"P13": 100}
            self._bounded_smoke_phase_channels = {"P13": 4}
            self.done = False
            self.decision_count = 0
            self.trace = []

        def reset(self, *, seed):
            assert seed == 1001

        def step(self, action):
            assert action == raw
            assert self._bounded_smoke_phase_decisions == {}
            assert self._bounded_smoke_phase_channels == {}
            if mode == "bounded-smoke":
                assert self.backend.request == {
                    "phase_id": "P13", "raw_policy_action_full12": raw,
                    "phase_mask_full12": mask,
                }
            else:
                assert self.backend.request is None
            calls.append("step")
            self.done = True
            self.decision_count = 1
            self.trace = [{"termination_reason": "SUCCESS"}]
            return SimpleNamespace(reward=0.0, info={})

    class Writer:
        def __init__(self, episode_dir, *, seed, require_actuator_target_effect_audit):
            assert require_actuator_target_effect_audit is (mode == "bounded-smoke")
            self.episode_dir = episode_dir

        def start(self, source):
            # The reset frame has not dispatched an episode command and does
            # not need an actuator effect audit yet.
            assert source is frame

        def write_tick(self, *args):
            pass

        def write_decision(self, info):
            pass

        def finalize(self, *args, **kwargs):
            path = self.episode_dir / "trial_manifest.json"
            cli._json(path, {"action_projection_audit": {}})
            return path

        def abort(self):
            pytest.fail("episode unexpectedly aborted")

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", Backend)
    monkeypatch.setattr(residual_direct_env, "ResidualEpisodeEnv", Env)
    monkeypatch.setattr(live_stream_writer, "LiveStreamWriter", Writer)
    monkeypatch.setattr(cli, "_pinned_runtime_phase_contracts", lambda args: (object(), object()))
    monkeypatch.setattr(cli, "_revalidate_pinned_phase_contracts", lambda *args: None)
    args = SimpleNamespace(num_envs=1, seed=1001, run_dir=tmp_path,
                           maximum_duration_s=200.0, residual_mode=mode)
    result = cli._run_live_episodes(
        args, object(), action_factory=lambda env, tick: raw, episode_count=1
    )
    assert result["all_success"] is True
    assert calls == (["backend", "request", "step"] if mode == "bounded-smoke" else ["backend", "step"])
