#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Discover the latest compatible adapter versions for the CI test matrix."""

import json
import os
import subprocess
import sys


def latest_compatible_version(package: str, python_version: str) -> str | None:
    """Find the latest available package version for a Python version."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "index",
        "versions",
        package,
        "--pre",
        "--json",
        "--no-cache-dir",
        "--python-version",
        python_version,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode:
        if "No matching distribution found" in result.stderr:
            print(
                f"::warning::No compatible {package} "
                f"for Python {python_version}",
                flush=True,
            )
            return None

        raise RuntimeError(
            f"Discovery failed for {package} "
            f"on Python {python_version}:\n"
            f"{result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid index response for {package} "
            f"on Python {python_version}: {exc}"
        ) from exc

    return data.get("latest")


def main() -> None:
    """Generate the GitHub Actions test matrix."""
    python_versions = json.loads(os.environ["PYTHON_VERSIONS_JSON"])
    runners = json.loads(os.environ["TEST_RUNNERS_JSON"])
    linux_runner = json.loads(os.environ["LINUX_RUNNER_JSON"])
    default_python_version = os.environ["DEFAULT_PY_VERSION"]

    if not isinstance(python_versions, list) or not python_versions:
        raise ValueError("PYTHON_VERSIONS_JSON must be a nonempty JSON array")

    if not isinstance(runners, list) or not runners:
        raise ValueError("TEST_RUNNERS_JSON must be a nonempty JSON array")

    entries = []

    for python_version in python_versions:
        vgf_version = latest_compatible_version(
            "vgf-adapter-model-explorer", python_version
        )
        tosa_version = latest_compatible_version(
            "tosa-adapter-model-explorer", python_version
        )

        available = bool(vgf_version and tosa_version)

        for runner in runners:
            entries.append(
                {
                    "python-version": python_version,
                    "runner": runner,
                    "vgf-version": vgf_version or "unavailable",
                    "tosa-version": tosa_version or "unavailable",
                    "available": available,
                    "coverage": (
                        available
                        and python_version == default_python_version
                        and runner == linux_runner
                    ),
                }
            )

    if not any(entry["coverage"] for entry in entries):
        raise RuntimeError(
            f"No Python {default_python_version} Linux runner "
            "found for coverage"
        )

    if len(entries) > 256:
        raise RuntimeError(
            f"Matrix contains {len(entries)} jobs (GitHub Actions limit: 256)"
        )

    print(json.dumps(entries, indent=2))

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(
            "matrix=" + json.dumps(entries, separators=(",", ":")) + "\n"
        )


if __name__ == "__main__":
    main()
