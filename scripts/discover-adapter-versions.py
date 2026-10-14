#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Build the CI test matrix from pip's published adapter versions on stdin."""

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

PROJECT_FILE = Path(__file__).resolve().parents[1] / "pyproject.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="Adapter dependency to discover")
    args = parser.parse_args()

    with PROJECT_FILE.open("rb") as project_file:
        project = tomllib.load(project_file)
    requirements = map(Requirement, project["project"]["dependencies"])
    requirement = next(
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name)
        == canonicalize_name(args.package)
    )

    index = json.load(sys.stdin)
    versions = sorted(
        {
            parsed
            for version in index["versions"]
            if not (parsed := Version(version)).is_prerelease
            and not parsed.is_devrelease
            and parsed in requirement.specifier
        }
    )
    if not versions:
        raise ValueError(f"No published stable versions satisfy {requirement}")

    version_list = [str(version) for version in versions]
    output = (
        f"versions={json.dumps(version_list)}\n"
        f"latest-version={version_list[-1]}\n"
    )
    print(output, end="")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as file:
        file.write(output)


if __name__ == "__main__":
    main()
