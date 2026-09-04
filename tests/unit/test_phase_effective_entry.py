from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from wlr50_clean.ppo import phase_effective_entry as effective_entry_module
from wlr50_clean.ppo.phase_effective_entry import (
    CONTACT_SOURCE,
    DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
    EffectivePhaseEntryError,
    FINGERPRINT_FIELDS,
    PHASE_IDS,
    assert_effective_phase_entry_contract_unchanged,
    binary64_ulp_distance,
    capture_validated_effective_phase_entry_contract,
    validate_effective_phase_entry_comparison,
)
from wlr50_clean.ppo.phase_snapshots import (
    DEFAULT_PHASE_SNAPSHOT_ROOT,
    capture_validated_phase_snapshot_bundle,
)


@pytest.fixture(scope="module")
def snapshot_bundle():
    return capture_validated_phase_snapshot_bundle(DEFAULT_PHASE_SNAPSHOT_ROOT)


@pytest.fixture(scope="module")
def contract(snapshot_bundle):
    return capture_validated_effective_phase_entry_contract(
        expected_snapshot_bundle=snapshot_bundle
    )


def _comparison(entry):
    pairs = {}
    exact = {}
    for wheel in entry["raw_contacts"]:
        if wheel == "signature_sha256":
            continue
        reference = entry["raw_contacts"][wheel]
        pairs[wheel] = {
            pair_name: {
                "pair_verified": pair["pair_verified"],
                "source": pair["source"],
                "force_w_n": list(pair["force_w_n"]),
            }
            for pair_name, pair in (
                ("ground", reference["ground"]),
                ("obstacle", reference["obstacle"]),
            )
        }
        exact[wheel] = {
            "body_name": reference["body_name"],
            "actual_class": reference["classification"],
            "actual_ground_active": reference["ground"]["active"],
            "actual_obstacle_active": reference["obstacle"]["active"],
        }
    return {
        "maximum_errors": dict(entry["post_prime_fingerprint"]),
        "raw_physx_contacts": {"pairs": pairs},
        "exact_contacts": exact,
    }


def _nextafter(value: float, count: int) -> float:
    result = float(value)
    for _ in range(count):
        result = math.nextafter(result, math.inf)
    return result


def _write_contract_copy(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.with_suffix(".sha256").write_bytes(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n".encode("ascii")
    )


def test_checked_in_contract_is_strictly_bound_and_lf_stable(
    contract, snapshot_bundle
) -> None:
    assert tuple(phase for phase, _ in contract.entries) == PHASE_IDS
    assert contract.phase_snapshot_bundle_sha256 == snapshot_bundle.bundle_sha256
    assert b"\r" not in contract.contract_bytes
    assert contract.contract_bytes.endswith(b"\n")
    assert b"\r" not in contract.sidecar_bytes
    assert contract.sidecar_bytes.endswith(b"\n")
    payload = __import__("json").loads(contract.contract_bytes)
    assert payload["portability_scope"] == "same_locked_host_runtime_only"
    assert payload["calibration_status"] == (
        "provisional_pending_independent_fresh_holdout"
    )
    derivation = payload["derivation"]
    assert {row["runtime_content_sha256"] for row in derivation["calibration_artifacts"]} == {
        derivation["runtime_content_sha256"]
    }
    assert {row["identity_config_sha256"] for row in derivation["calibration_artifacts"]} == {
        derivation["identity_config_sha256"]
    }


def test_p01_is_explicitly_outside_the_effective_entry_contract(contract) -> None:
    with pytest.raises(EffectivePhaseEntryError, match="P01"):
        contract.entry("P01")


@pytest.mark.parametrize("tamper", ("fingerprint", "contact", "entry_sha256"))
def test_entry_copy_tamper_cannot_mutate_pinned_contract(
    contract, snapshot_bundle, tamper: str
) -> None:
    phase = "P02"
    original = contract.entry(phase)
    candidate = contract.entry(phase)
    if tamper == "fingerprint":
        candidate["post_prime_fingerprint"][FINGERPRINT_FIELDS[0]] += 1.0
    elif tamper == "contact":
        candidate["raw_contacts"]["front_left_ankle"]["ground"]["active"] = not (
            candidate["raw_contacts"]["front_left_ankle"]["ground"]["active"]
        )
    else:
        candidate["entry_sha256"] = "0" * 64

    assert contract.entry(phase) == original
    assert_effective_phase_entry_contract_unchanged(
        contract, expected_snapshot_bundle=snapshot_bundle
    )


@pytest.mark.parametrize("tamper", ("fingerprint", "contact", "entry_sha256"))
def test_internal_entry_tree_rejects_in_place_tamper(contract, tamper: str) -> None:
    internal = contract.entries[0][1]
    with pytest.raises(TypeError):
        if tamper == "fingerprint":
            internal["post_prime_fingerprint"][FINGERPRINT_FIELDS[0]] = 1.0
        elif tamper == "contact":
            internal["raw_contacts"]["front_left_ankle"]["ground"]["active"] = False
        else:
            internal["entry_sha256"] = "0" * 64


def test_capture_records_every_file_ancestor_surface(tmp_path: Path) -> None:
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    source = nested / "contract.bin"
    source.write_bytes(b"pinned")

    payloads, identities = effective_entry_module._capture_paths_once(
        {"contract": source}
    )

    assert payloads == {"contract": b"pinned"}
    directory_paths = {
        Path(identity[0]) for identity in identities if identity[1] == "directory"
    }
    assert nested.resolve() in directory_paths
    assert tmp_path.resolve() in directory_paths


def test_capture_reads_each_file_through_one_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    original_open = effective_entry_module.os.open
    opened_flags = []

    def tracked_open(path, flags):
        opened_flags.append(flags)
        return original_open(path, flags)

    def forbidden_read_bytes(_path):
        raise AssertionError("Path.read_bytes must not reopen a pinned file")

    monkeypatch.setattr(effective_entry_module.os, "open", tracked_open)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    payloads, _ = effective_entry_module._capture_paths_once({"contract": source})

    assert payloads == {"contract": b"pinned"}
    assert len(opened_flags) == 1
    no_follow = int(getattr(effective_entry_module.os, "O_NOFOLLOW", 0))
    if no_follow:
        assert opened_flags[0] & no_follow


def test_capture_rejects_symlink_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "real.bin"
    target.write_bytes(b"pinned")
    linked = tmp_path / "linked.bin"
    try:
        linked.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(EffectivePhaseEntryError, match="symlink|reparse|redirect"):
        effective_entry_module._capture_paths_once({"contract": linked})


def test_capture_rejects_open_handle_path_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    original = effective_entry_module._handle_identity

    def mismatched_handle(descriptor, path, *, label, directory):
        identity = list(
            original(descriptor, path, label=label, directory=directory)
        )
        identity[3] += 1
        return tuple(identity)

    monkeypatch.setattr(
        effective_entry_module, "_handle_identity", mismatched_handle
    )
    with pytest.raises(EffectivePhaseEntryError, match="opened contract differs"):
        effective_entry_module._capture_paths_once({"contract": source})


def test_capture_rejects_visible_ancestor_aba_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    target = tmp_path.resolve()
    original = effective_entry_module._path_identity
    calls = 0

    def changed_ancestor(path, *, label, directory):
        nonlocal calls
        identity = original(path, label=label, directory=directory)
        if path == target and directory:
            calls += 1
            if calls > 1:
                changed = list(identity)
                changed[3] += 1
                return tuple(changed)
        return identity

    monkeypatch.setattr(
        effective_entry_module, "_path_identity", changed_ancestor
    )
    with pytest.raises(EffectivePhaseEntryError, match="captured path changed"):
        effective_entry_module._capture_paths_once({"contract": source})


def test_capture_allows_unrelated_directory_metadata_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    target = tmp_path.resolve()
    original = effective_entry_module._path_identity
    calls = 0

    def changed_directory_metadata(path, *, label, directory):
        nonlocal calls
        identity = original(path, label=label, directory=directory)
        if path == target and directory:
            calls += 1
            if calls > 1:
                changed = list(identity)
                changed[4] += 1
                changed[5] += 1
                changed[6] += 1
                changed[9] += 1
                return tuple(changed)
        return identity

    monkeypatch.setattr(
        effective_entry_module, "_path_identity", changed_directory_metadata
    )
    payloads, _ = effective_entry_module._capture_paths_once({"contract": source})

    assert payloads == {"contract": b"pinned"}


def test_fingerprint_exact_and_one_ulp_pass_but_two_ulp_fails(contract) -> None:
    phase = "P02"
    entry = contract.entry(phase)
    exact = _comparison(entry)
    proof = validate_effective_phase_entry_comparison(contract, phase, exact)
    assert proof["verified"] is True
    assert set(proof["fingerprint"]) == set(FINGERPRINT_FIELDS)

    field = FINGERPRINT_FIELDS[0]
    one_ulp = _comparison(entry)
    one_ulp["maximum_errors"][field] = _nextafter(
        entry["post_prime_fingerprint"][field], 1
    )
    assert binary64_ulp_distance(
        one_ulp["maximum_errors"][field],
        entry["post_prime_fingerprint"][field],
    ) == 1
    assert validate_effective_phase_entry_comparison(
        contract, phase, one_ulp
    )["verified"] is True

    two_ulp = _comparison(entry)
    two_ulp["maximum_errors"][field] = _nextafter(
        entry["post_prime_fingerprint"][field], 2
    )
    with pytest.raises(EffectivePhaseEntryError, match="2 ULP"):
        validate_effective_phase_entry_comparison(contract, phase, two_ulp)


def test_fingerprint_nan_is_rejected(contract) -> None:
    comparison = _comparison(contract.entry("P03"))
    comparison["maximum_errors"][FINGERPRINT_FIELDS[2]] = math.nan
    with pytest.raises(EffectivePhaseEntryError, match="finite"):
        validate_effective_phase_entry_comparison(contract, "P03", comparison)


def test_live_force_values_are_diagnostic_but_threshold_class_is_hard(contract) -> None:
    phase = "P02"
    comparison = _comparison(contract.entry(phase))
    active_pair = next(
        (wheel, pair_name)
        for wheel, pairs in comparison["raw_physx_contacts"]["pairs"].items()
        for pair_name, pair in pairs.items()
        if math.sqrt(sum(value * value for value in pair["force_w_n"])) >= 0.25
    )
    wheel, pair_name = active_pair
    comparison["raw_physx_contacts"]["pairs"][wheel][pair_name]["force_w_n"] = [
        0.0,
        0.0,
        0.25,
    ]
    assert validate_effective_phase_entry_comparison(
        contract, phase, comparison
    )["verified"] is True

    comparison["raw_physx_contacts"]["pairs"][wheel][pair_name]["force_w_n"] = [
        0.0,
        0.0,
        0.249,
    ]
    with pytest.raises(EffectivePhaseEntryError, match="raw contact contract"):
        validate_effective_phase_entry_comparison(contract, phase, comparison)


@pytest.mark.parametrize("tamper", ["source", "pair_verified", "classifier", "double"])
def test_contact_tamper_fails_closed(contract, tamper: str) -> None:
    phase = "P02"
    comparison = _comparison(contract.entry(phase))
    wheel = next(iter(comparison["raw_physx_contacts"]["pairs"]))
    pairs = comparison["raw_physx_contacts"]["pairs"][wheel]
    if tamper == "source":
        pairs["ground"]["source"] = CONTACT_SOURCE + ".tampered"
    elif tamper == "pair_verified":
        pairs["ground"]["pair_verified"] = False
    elif tamper == "classifier":
        comparison["exact_contacts"][wheel]["actual_ground_active"] = not (
            comparison["exact_contacts"][wheel]["actual_ground_active"]
        )
    else:
        pairs["ground"]["force_w_n"] = [0.0, 0.0, 1.0]
        pairs["obstacle"]["force_w_n"] = [0.0, 0.0, 1.0]
        comparison["exact_contacts"][wheel].update(
            {
                "actual_class": "GROUND_AND_OBSTACLE",
                "actual_ground_active": True,
                "actual_obstacle_active": True,
            }
        )
    with pytest.raises(EffectivePhaseEntryError):
        validate_effective_phase_entry_comparison(contract, phase, comparison)


def test_loader_rejects_sidecar_tamper(snapshot_bundle, tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract_copy(contract_path, DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.read_bytes())
    contract_path.with_suffix(".sha256").write_bytes(b"0" * 64 + b"  contract.json\n")
    with pytest.raises(EffectivePhaseEntryError, match="sidecar mismatch"):
        capture_validated_effective_phase_entry_contract(
            contract_path, expected_snapshot_bundle=snapshot_bundle
        )


@pytest.mark.parametrize("kind", ["duplicate", "nan"])
def test_loader_rejects_duplicate_keys_and_nonfinite_json(
    snapshot_bundle, tmp_path: Path, kind: str
) -> None:
    payload = DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.read_bytes()
    if kind == "duplicate":
        payload = payload.replace(
            b'{\n  "schema":',
            b'{\n  "schema": "duplicate",\n  "schema":',
            1,
        )
        expected = "duplicate JSON key"
    else:
        payload = payload.replace(
            b'  "fingerprint_max_ulp_distance": 1,',
            b'  "fingerprint_max_ulp_distance": NaN,',
            1,
        )
        expected = "non-finite JSON constant"
    contract_path = tmp_path / "contract.json"
    _write_contract_copy(contract_path, payload)
    with pytest.raises(EffectivePhaseEntryError, match=expected):
        capture_validated_effective_phase_entry_contract(
            contract_path, expected_snapshot_bundle=snapshot_bundle
        )
