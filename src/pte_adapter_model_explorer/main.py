# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from pathlib import Path
from typing import Dict

from model_explorer import (
    Adapter,
    AdapterMetadata,
    ModelExplorerGraphs,
)

from .builder.builder import PteGraphBuilder


class PTEAdapter(Adapter):
    metadata = AdapterMetadata(
        id="pte_adapter_model_explorer",
        name="PTE Adapter",
        description="PTE adapter for Model Explorer",
        fileExts=["pte"],
    )

    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert a given model to a model-explorer compatible format."""

        return {
            "graphCollections": PteGraphBuilder(
                Path(model_path), settings
            ).graph_collections
        }
