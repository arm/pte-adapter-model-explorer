# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import glob
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from ..builder import builder

FIXTURES_ROOT = os.path.join(os.path.dirname(__file__), "fixtures")

test_case_dirs = [
    d for d in glob.glob(os.path.join(FIXTURES_ROOT, "*")) if os.path.isdir(d)
]


_KEY_QUOTE_PATTERN = re.compile(
    r"(?m)^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:(?=\s)"
)


def load_expected_fixture(path: str):
    with open(path) as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        normalised = _KEY_QUOTE_PATTERN.sub(
            lambda match: f'{match.group("indent")}"{match.group("key")}":',
            raw,
        )
        try:
            return json.loads(normalised)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Unable to parse expected fixture {path!r} into JSON."
            ) from exc


@pytest.mark.parametrize(
    "case_dir", test_case_dirs, ids=lambda d: os.path.basename(d)
)
def test_e2e(case_dir):
    """Test parsing for each PTE file and compare against expected graph output."""
    input_pte = Path(os.path.join(case_dir, "input.pte"))
    expected_main_json = os.path.join(case_dir, "expected_main.json")
    expected_delegates_json = os.path.join(case_dir, "expected_delegates.json")
    settings_json = os.path.join(case_dir, "settings.json")

    settings = {}

    if os.path.exists(settings_json):
        with open(settings_json, "r") as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {settings_json}: {e}"
                ) from e

    graph_collections = builder.PteGraphBuilder(
        input_pte, settings=settings
    ).graph_collections

    got = [asdict(gc) for gc in graph_collections]

    expected_main = load_expected_fixture(expected_main_json)

    assert got[0] == expected_main, (
        f"Test failed for {input_pte}. Expected and actual output differ."
    )

    if os.path.exists(expected_delegates_json):
        expected_delegates = load_expected_fixture(expected_delegates_json)
        assert got[1] == expected_delegates, (
            f"Test failed for {input_pte}. Expected and actual output differ."
        )


@pytest.mark.parametrize(
    "fixture_pair",
    [
        (
            os.path.join(
                FIXTURES_ROOT, "mobilenet_v2_vgf", "expected_main.json"
            ),
            os.path.join(
                FIXTURES_ROOT,
                "mobilenet_v2_vgf_segments",
                "expected_main.json",
            ),
        )
    ],
    ids=["mobilenet_v2_vgf_vs_segments"],
)
def test_expected_main_fixtures_identical(fixture_pair):
    """Ensure fixtures that should align share identical expected_main outputs."""

    canonical_fixture, segmented_fixture = fixture_pair

    assert os.path.exists(canonical_fixture), (
        f"Missing canonical expected_main fixture: {canonical_fixture}"
    )
    assert os.path.exists(segmented_fixture), (
        f"Missing segmented expected_main fixture: {segmented_fixture}"
    )

    canonical_expected = load_expected_fixture(canonical_fixture)
    segmented_expected = load_expected_fixture(segmented_fixture)

    assert segmented_expected == canonical_expected, (
        "mobilenet_v2 expected_main outputs diverge between the segmented and "
        "non-segmented fixtures."
    )
