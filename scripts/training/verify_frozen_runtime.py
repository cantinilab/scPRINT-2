#!/usr/bin/env python3
"""Create and verify fail-closed provenance for the primary training runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_PACKAGES = {
    "lamindb": "2.1.1",
    "torch": "2.8.0",
}

INVENTORY_PROGRAM = r"""
import importlib.metadata as metadata
import json
import re
import sys
rows=[]
for distribution in metadata.distributions():
    name=distribution.metadata.get('Name') or distribution.name
    canonical=re.sub(r'[-_.]+','-',name).lower()
    direct_url=None
    text=distribution.read_text('direct_url.json')
    if text:
        try:
            value=json.loads(text)
            vcs=value.get('vcs_info') or {}
            direct_url={
                'url':value.get('url'),
                'editable':bool((value.get('dir_info') or {}).get('editable')),
                'vcs':vcs.get('vcs'),
                'commit_id':vcs.get('commit_id'),
            }
        except json.JSONDecodeError:
            direct_url={'raw':text}
    rows.append({'name':canonical,'version':distribution.version,'direct_url':direct_url})
rows.sort(key=lambda row:(row['name'],row['version'],json.dumps(row['direct_url'],sort_keys=True)))
print(json.dumps({'python':sys.version,'packages':rows},sort_keys=True,separators=(',',':')))
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def git_clean(path: Path) -> bool:
    return not subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def runtime_inventory(
    runtime_root: Path, project: Path, scdataloader: Path
) -> dict[str, Any]:
    python = runtime_root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(python)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{project}:{scdataloader}"
    output = subprocess.run(
        [str(python), "-c", INVENTORY_PROGRAM],
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    ).stdout
    return json.loads(output)


def package_versions(inventory: dict[str, Any]) -> dict[str, str]:
    return {row["name"]: row["version"] for row in inventory["packages"]}


def assert_expected_packages(inventory: dict[str, Any]) -> None:
    versions = package_versions(inventory)
    for name, expected in EXPECTED_PACKAGES.items():
        if versions.get(name) != expected:
            raise RuntimeError(
                f"runtime package mismatch for {name}: {versions.get(name)!r} != {expected!r}"
            )


def create_manifest(
    runtime_root: Path,
    project: Path,
    scdataloader: Path,
    scprint_commit: str,
    scdataloader_commit: str,
    uv_bin: Path,
) -> dict[str, Any]:
    manifest_path = runtime_root / "runtime_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if git_head(project) != scprint_commit or not git_clean(project):
        raise RuntimeError("scPRINT source is not the exact clean reviewed commit")
    if git_head(scdataloader) != scdataloader_commit or not git_clean(scdataloader):
        raise RuntimeError("scDataLoader source is not the exact clean reviewed commit")
    inventory = runtime_inventory(runtime_root, project, scdataloader)
    assert_expected_packages(inventory)
    python = runtime_root / ".venv" / "bin" / "python"
    uv_version = subprocess.run(
        [str(uv_bin), "--version"], text=True, capture_output=True, check=True
    ).stdout.strip()
    manifest = {
        "status": "accepted",
        "runtime_root": str(runtime_root.resolve()),
        "python_executable": str(python.resolve()),
        "python_executable_sha256": sha256(python.resolve()),
        "python_version": inventory["python"],
        "package_inventory": inventory["packages"],
        "package_inventory_sha256": hashlib.sha256(
            canonical_json(inventory).encode()
        ).hexdigest(),
        "expected_packages": EXPECTED_PACKAGES,
        "scprint_commit": scprint_commit,
        "scdataloader_commit": scdataloader_commit,
        "uv_version": uv_version,
        "uv_bin": str(uv_bin.resolve()),
        "uv_bin_sha256": sha256(uv_bin.resolve()),
        "uv_lock_sha256": sha256(project / "uv.lock"),
        "pyproject_sha256": sha256(project / "pyproject.toml"),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def verify_manifest(
    runtime_root: Path,
    project: Path,
    scdataloader: Path,
    scprint_commit: str,
    scdataloader_commit: str,
) -> dict[str, Any]:
    manifest_path = runtime_root / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "accepted":
        raise RuntimeError("runtime manifest is not accepted")
    if (
        manifest["scprint_commit"] != scprint_commit
        or git_head(project) != scprint_commit
    ):
        raise RuntimeError("scPRINT runtime/source commit mismatch")
    if (
        manifest["scdataloader_commit"] != scdataloader_commit
        or git_head(scdataloader) != scdataloader_commit
    ):
        raise RuntimeError("scDataLoader runtime/source commit mismatch")
    if not git_clean(project) or not git_clean(scdataloader):
        raise RuntimeError("runtime source worktree is dirty")
    if manifest["uv_lock_sha256"] != sha256(project / "uv.lock"):
        raise RuntimeError("uv.lock hash mismatch")
    if manifest["pyproject_sha256"] != sha256(project / "pyproject.toml"):
        raise RuntimeError("pyproject hash mismatch")
    python = runtime_root / ".venv" / "bin" / "python"
    if manifest["python_executable_sha256"] != sha256(python.resolve()):
        raise RuntimeError("runtime Python executable hash mismatch")
    inventory = runtime_inventory(runtime_root, project, scdataloader)
    assert_expected_packages(inventory)
    inventory_sha256 = hashlib.sha256(canonical_json(inventory).encode()).hexdigest()
    if manifest["package_inventory_sha256"] != inventory_sha256:
        raise RuntimeError("runtime package inventory hash mismatch")
    if manifest["package_inventory"] != inventory["packages"]:
        raise RuntimeError("runtime package inventory changed")
    return {
        "status": "accepted",
        "runtime_root": manifest["runtime_root"],
        "python_version": inventory["python"],
        "package_inventory_sha256": inventory_sha256,
        "uv_lock_sha256": manifest["uv_lock_sha256"],
        "scprint_commit": scprint_commit,
        "scdataloader_commit": scdataloader_commit,
        "expected_packages": EXPECTED_PACKAGES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("runtime_root", type=Path)
        child.add_argument("project", type=Path)
        child.add_argument("scdataloader", type=Path)
        child.add_argument("scprint_commit")
        child.add_argument("scdataloader_commit")
        if command == "create":
            child.add_argument("--uv-bin", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = create_manifest(
            args.runtime_root,
            args.project,
            args.scdataloader,
            args.scprint_commit,
            args.scdataloader_commit,
            args.uv_bin,
        )
    else:
        result = verify_manifest(
            args.runtime_root,
            args.project,
            args.scdataloader,
            args.scprint_commit,
            args.scdataloader_commit,
        )
    print("FROZEN_RUNTIME_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
