#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Discover the latest compatible adapter versions for the CI test matrix."""

import json
import os
import re
import subprocess  # nosec B404


def latest_compatible_version(package: str, python_version: str) -> str | None:
    """Find the latest available package version for a Python version."""
    command = [
        "uv",
        "pip",
        "compile",
        "-",
        "--no-deps",
        "--no-build",
        "--prerelease",
        "allow",
        "--no-header",
        "--no-annotate",
        "--no-cache",
        "--no-python-downloads",
        "--python-version",
        python_version,
    ]

    result = subprocess.run(  # nosec B603
        command,
        input=f"{package}\n",
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode:
        # uv reports an empty/incompatible index as an unsatisfiable
        # resolution. Authentication and transport failures must still fail CI.
        diagnostic = result.stderr.lower()
        index_failed = any(
            marker in diagnostic
            for marker in (
                "401",
                "403",
                "authentication",
                "could not be queried",
                "failed to fetch",
                "connection",
                "timed out",
            )
        )
        if (
            result.returncode == 1
            and "no solution found" in diagnostic
            and not index_failed
        ):
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

    match = re.search(
        rf"^{re.escape(package)}==([^\s;]+)$", result.stdout, re.MULTILINE
    )
    if match is None:
        raise RuntimeError(
            f"Invalid uv resolution for {package} on Python {python_version}"
        )
    return match.group(1)


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
        if not isinstance(python_version, str) or not re.fullmatch(
            r"3\.\d+", python_version
        ):
            raise ValueError(f"Invalid Python version: {python_version}")
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
