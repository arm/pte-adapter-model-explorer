# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from pathlib import Path
from typing import Dict

from model_explorer import ModelExplorerGraphs
from model_explorer import graph_builder as gb
from tosa_adapter_model_explorer.main import TosaFlatbufferAdapter

ETHOSU_DELEGATE_NAME = "EthosUBackend"


def ethosu_delegate_handler(blob: bytes, settings: Dict) -> gb.Graph:
    """
    Convert TOSA delegate blob to a Model Explorer graph.
    Caller should ensure settings["delegate_file_paths"] and settings["delegate_call_count"] are set
    appropriately.
    Handler consumes these settings, removing them from the settings dict.
    Args:
        blob: The delegate blob (not used in this function).
        settings: A dictionary of settings for the conversion process.
    Returns:
        A Model Explorer graph representing the TOSA delegate.
    """
    tosa_adapter = TosaFlatbufferAdapter()

    call_count: int = settings.pop("delegate_call_count")
    delegate_file_paths: list[Path] = settings.pop("delegate_file_paths")
    delegate_file_paths = [
        path for path in delegate_file_paths if path.name.endswith(".tosa")
    ]
    delegate_file_index: int = len(delegate_file_paths) - 1 - call_count
    delegate_file_path: Path = delegate_file_paths[delegate_file_index]

    tosa_adapter_graphs: ModelExplorerGraphs = tosa_adapter.convert(
        str(delegate_file_path), settings=settings
    )

    if (
        "graphs" not in tosa_adapter_graphs
        or not tosa_adapter_graphs["graphs"]
    ):
        raise ValueError(
            f"TOSA adapter did not return any graphs for delegate file {delegate_file_path}"
        )
    else:
        return tosa_adapter_graphs["graphs"][0]
