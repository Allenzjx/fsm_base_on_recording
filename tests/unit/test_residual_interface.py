from wlr50_clean.ppo.residual_interface import (
    OBSERVATION_DIM,
    PPOObservationParts,
    ResidualInterface,
    zero_residual_is_nominal,
)


def _parts() -> PPOObservationParts:
    return PPOObservationParts(
        joint_position_error8=(0.0,) * 8,
        joint_velocity8=(0.0,) * 8,
        wheel_velocity4=(0.0,) * 4,
        wheel_contact_code4=(0.0,) * 4,
        leg_history12=(0.0,) * 12,
        body_orientation_wxyz4=(1.0, 0.0, 0.0, 0.0),
        body_angular_velocity3=(0.0,) * 3,
        obstacle_relative_geometry9=(0.0,) * 9,
        full_body_com3=(0.0,) * 3,
        support_diagnostics4=(0.0,) * 4,
    )


def test_zero_residual_is_exact_nominal() -> None:
    interface = ResidualInterface(residual_enabled=False)
    nominal = tuple(float(index) for index in range(12))
    frame = interface.frame(
        state_id="P01",
        macro_phase=1,
        phase_progress=0.25,
        nominal_action_full12=nominal,
        action_mask_full12=(1,) * 12,
        observation=_parts(),
        previous_action_full12=(0.0,) * 12,
    )
    assert len(frame.observation_vector) == OBSERVATION_DIM
    assert interface.compose_action(frame) == nominal
    assert zero_residual_is_nominal(frame) is True
