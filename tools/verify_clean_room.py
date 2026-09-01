"""Fail closed when the clean controller contains forbidden old-FSM dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "clean_room_import_manifest.json"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "forbidden_import_scan.json"

FORBIDDEN_TEXT = {
    "old namespace": re.compile(r"\bwlr50\.fsm\b", re.IGNORECASE),
    "old chain owner": re.compile(r"\bRRFirstV010ChainOwner\b"),
    "old P08 owner": re.compile(r"\bP08_DYNAMIC_PRODUCTION\b"),
    "old trial owner": re.compile(r"\bTrial(?:71|72|73)\b.*\bowner\b", re.IGNORECASE),
    "old release gate": re.compile(r"\bold[ _-]?release[ _-]?gate\b", re.IGNORECASE),
    "old project path": re.compile(r"fsm_50mm_recording_sensor_fsm_v1", re.IGNORECASE),
    "recording cursor": re.compile(r"\brecording_cursor\b", re.IGNORECASE),
}

RUNTIME_ROOTS = (
    SOURCE_ROOT / "wlr50_clean" / "fsm",
    SOURCE_ROOT / "wlr50_clean" / "sensing",
    SOURCE_ROOT / "wlr50_clean" / "infrastructure",
    SOURCE_ROOT / "wlr50_clean" / "ppo",
)
RUNTIME_RECORDING_PATTERNS = (
    re.compile(r"accepted_steps\.jsonl", re.IGNORECASE),
    re.compile(r"semantic_segments\.json", re.IGNORECASE),
    re.compile(r"reference[/\\\\]v010", re.IGNORECASE),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[Path]:
    return sorted(path for path in SOURCE_ROOT.rglob("*") if path.suffix in {".py", ".json", ".yaml", ".yml", ".toml"})


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data.get("reused_files"), list):
        raise ValueError("manifest.reused_files must be a list")
    return data


def scan_text() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_TEXT.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"kind": label, "path": str(path), "line": line, "match": match.group(0)})
        if any(root in path.parents for root in RUNTIME_ROOTS):
            for pattern in RUNTIME_RECORDING_PATTERNS:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    findings.append({"kind": "runtime recording access", "path": str(path), "line": line, "match": match.group(0)})
    return findings


def verify_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required = {"source_path", "source_sha256", "destination_path", "destination_sha256", "reuse_reason"}
    for index, row in enumerate(manifest["reused_files"]):
        missing = sorted(required - set(row))
        if missing:
            failures.append({"kind": "manifest fields", "index": index, "missing": missing})
            continue
        source = Path(row["source_path"])
        destination = PROJECT_ROOT / row["destination_path"]
        for role, path, expected in (
            ("source", source, row["source_sha256"]),
            ("destination", destination, row["destination_sha256"]),
        ):
            if not path.is_file():
                failures.append({"kind": "manifest file missing", "index": index, "role": role, "path": str(path)})
            else:
                actual = sha256(path)
                if actual.lower() != str(expected).lower():
                    failures.append({"kind": "manifest hash mismatch", "index": index, "role": role, "path": str(path), "expected": expected, "actual": actual})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    try:
        manifest = load_manifest()
        findings = scan_text()
        manifest_failures = verify_manifest(manifest)
        errors = findings + manifest_failures
        report = {
            "schema": "wlr50_clean.forbidden_import_scan.v1",
            "project_root": str(PROJECT_ROOT),
            "scanned_files": len(source_files()),
            "manifest_entries": len(manifest["reused_files"]),
            "passed": not errors,
            "errors": errors,
        }
    except Exception as exc:
        report = {
            "schema": "wlr50_clean.forbidden_import_scan.v1",
            "project_root": str(PROJECT_ROOT),
            "passed": False,
            "errors": [{"kind": "verifier error", "detail": f"{type(exc).__name__}: {exc}"}],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

