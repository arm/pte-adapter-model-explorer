# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""VGF adapter response compatibility for companion PTE export."""

from types import SimpleNamespace

import pytest

from pte_adapter_model_explorer.delegates import vgf


@pytest.mark.parametrize(
    "shape", ["graphs", "dict_collection", "object_collection"]
)
def test_vgf_delegate_accepts_legacy_and_collection_responses(
    monkeypatch, shape
):
    """A single delegate graph is preserved across adapter response versions."""
    graph = object()
    if shape == "graphs":
        response = {"graphs": [graph]}
    elif shape == "dict_collection":
        response = {"graphCollections": [{"graphs": [graph]}]}
    else:
        response = {"graphCollections": [SimpleNamespace(graphs=[graph])]}
    monkeypatch.setattr(
        vgf.VGFAdapter, "convert", lambda *_args, **_kwargs: response
    )
    assert vgf.vgf_delegate_handler(b"delegate", {}) is graph  # nosec B101


@pytest.mark.parametrize(
    "response",
    [
        {"graphs": []},
        {"graphCollections": []},
        {"graphs": [None, None]},
        {"graphCollections": [{"graphs": [None]}, {"graphs": [None]}]},
    ],
)
def test_vgf_delegate_rejects_empty_or_ambiguous_responses(
    monkeypatch, response
):
    """Unsupported shapes fail instead of silently discarding graphs."""
    monkeypatch.setattr(
        vgf.VGFAdapter, "convert", lambda *_args, **_kwargs: response
    )
    with pytest.raises(ValueError):
        vgf.vgf_delegate_handler(b"delegate", {})
