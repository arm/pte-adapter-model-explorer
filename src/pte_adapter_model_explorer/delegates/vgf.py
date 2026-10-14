# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

from model_explorer import graph_builder as gb
from vgf_adapter_model_explorer.main import VGFAdapter

VGF_DELEGATE_NAME = "VgfBackend"


def vgf_delegate_handler(blob: bytes, settings: Dict) -> gb.Graph:
    """
    Convert VGF delegate blob to a Model Explorer graph.
    Args:
        blob: The delegate blob.
        settings: A dictionary of settings for the conversion process.
    Returns:
        A Model Explorer graph representing the VGF delegate.
    """
    tmp_path: Path | None = None
    with NamedTemporaryFile(suffix=".vgf", delete=False) as tmp:
        tmp.write(blob)
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        vgf_adapter = VGFAdapter()
        vgf_adapter_graphs = vgf_adapter.convert(
            str(tmp_path), settings=settings
        )
        graphs = vgf_adapter_graphs.get("graphs")
        if graphs is None:
            collections = vgf_adapter_graphs.get("graphCollections") or []
            if len(collections) != 1:
                raise ValueError(
                    "VGF delegate requires exactly one graph collection"
                )
            collection = collections[0]
            graphs = (
                collection.get("graphs", [])
                if isinstance(collection, dict)
                else collection.graphs
            )
        if not graphs:
            raise ValueError(
                "VGF adapter did not return any graphs for delegate instruction blob"
            )
        if len(graphs) != 1:
            raise ValueError(
                "VGF delegate requires exactly one top-level graph"
            )
        return graphs[0]

    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
