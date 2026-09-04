from __future__ import annotations

from types import SimpleNamespace

import pytest

from wlr50_clean.ppo.residual_direct_env import LivePhysicalSignalBuilder


WHEEL_ORDER = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)


def _observation(
    *,
    loads: tuple[float, float, float, float],
    front_right_x: float = 0.40,
    front_right_bottom_z: float = 0.20,
    front_right_obstacle_contact: bool = False,
    front_right_top_latched: bool = False,
) -> dict[str, object]:
    front_x = 0.50
    top_z = 0.20
    wheels: dict[str, object] = {}
    contacts: dict[str, object] = {}
    for index, (name, load) in enumerate(zip(WHEEL_ORDER, loads, strict=True)):
        body_name = f"{name}_body"
        is_front_right = name == "front_right_ankle"
        wheels[name] = {
            "body_name": body_name,
            "velocity_rad_s": 0.0,
            "center_w_m": (
                front_right_x if is_front_right else front_x + 0.10,
                0.0,
                0.25,
            ),
            "bottom_w_m": (
                front_right_x if is_front_right else front_x + 0.10,
                0.0,
                front_right_bottom_z if is_front_right else top_z,
            ),
        }
        obstacle_active = bool(is_front_right and front_right_obstacle_contact)
        contacts[body_name] = {
            "ground": {
                "pair_verified": True,
                "active": load > 0.0 and not obstacle_active,
                "normal_force_n": 0.0 if obstacle_active else load,
            },
            "obstacle": {
                "pair_verified": True,
                "active": obstacle_active,
                "normal_force_n": load if obstacle_active else 0.0,
                "active_history": (obstacle_active,) * 8,
            },
        }
    return {
        "wheels": wheels,
        "contacts": contacts,
        "base": {
            "position_w_m": (0.0, 0.0, 0.0),
            "linear_velocity_w_m_s": (0.0, 0.0, 0.0),
        },
        "support": {"support_count": 4},
        "center_of_mass": {"position_w_m": (0.0, 0.0, 0.0)},
        "obstacle": {"front_x_m": front_x, "top_z_m": top_z},
        "joints": {},
        "guards": {
            "leg_top_loaded_latched:FR": {
                "passed": front_right_top_latched,
            }
        },
    }


def _progress(state_id: str, observation: dict[str, object]):
    frame = SimpleNamespace(
        state_id=state_id,
        info={"raw_observation": observation},
    )
    return LivePhysicalSignalBuilder().progress(frame).normalized_terms


def test_transfer_progress_uses_the_phase_specific_target_load() -> None:
    # FL=10%, FR=70%: P08 must not receive credit for the wrong-side FR load,
    # while P11 must measure that FR target directly.
    fr_loaded = _observation(loads=(10.0, 70.0, 10.0, 10.0))
    fl_loaded = _observation(loads=(70.0, 10.0, 10.0, 10.0))

    assert _progress("P08", fr_loaded)["com_target_progress"] == pytest.approx(0.10)
    assert _progress("P11", fr_loaded)["com_target_progress"] == pytest.approx(0.70)
    assert _progress("P08", fl_loaded)["com_target_progress"] == pytest.approx(0.70)
    assert _progress("P11", fl_loaded)["com_target_progress"] == pytest.approx(0.10)


def test_front_face_contact_cannot_claim_top_placement_or_capture() -> None:
    front_face_contact = _observation(
        loads=(10.0, 10.0, 10.0, 10.0),
        front_right_x=0.49,
        front_right_bottom_z=0.12,
        front_right_obstacle_contact=True,
    )

    terms = _progress("P03", front_face_contact)

    assert terms["fr_top_placement"] == 0.0
    assert terms["contact_capture"] == 0.0


def test_verified_top_contact_or_trusted_latch_can_claim_capture() -> None:
    live_top_contact = _observation(
        loads=(10.0, 10.0, 10.0, 10.0),
        front_right_x=0.51,
        front_right_bottom_z=0.205,
        front_right_obstacle_contact=True,
    )
    trusted_latch = _observation(
        loads=(10.0, 10.0, 10.0, 10.0),
        front_right_x=0.49,
        front_right_bottom_z=0.12,
        front_right_obstacle_contact=False,
        front_right_top_latched=True,
    )

    assert _progress("P03", live_top_contact)["contact_capture"] == 1.0
    latched_terms = _progress("P03", trusted_latch)
    assert latched_terms["fr_top_placement"] == 1.0
    assert latched_terms["contact_capture"] == 1.0
