# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from model_explorer import graph_builder as gb

from ..delegates.ethosu import ETHOSU_DELEGATE_NAME, ethosu_delegate_handler
from ..delegates.vgf import VGF_DELEGATE_NAME, vgf_delegate_handler
from ..executorch_flatbuffer import (
    BackendDelegateDataReferenceT,
    BackendDelegateInlineDataT,
    DataLocation,
    DataSegmentT,
)
from ..extended_header import (
    ExtendedHeader,
    ExtendedHeaderError,
    ExtendedHeaderNotFoundError,
    read_extended_header,
)

DELEGATE_HANDLERS = {
    VGF_DELEGATE_NAME: vgf_delegate_handler,
    ETHOSU_DELEGATE_NAME: ethosu_delegate_handler,
}


@dataclass(frozen=True)
class DelegateBuildContext:
    delegate_file_paths: list[Path]
    pte_path: Path


class DelegateGraphBuilder:
    """Helper class to build delegate graphs from backend delegate data."""

    label: str
    settings: Dict
    ref: BackendDelegateDataReferenceT
    context: DelegateBuildContext
    curr_call_count: int

    def __init__(
        self,
        label: str,
        settings: Dict,
        ref: BackendDelegateDataReferenceT,
        context: DelegateBuildContext,
        curr_call_count: int,
    ):
        self.label = label
        self.curr_call_count = curr_call_count
        self.settings = settings
        self.settings["delegate_file_paths"] = context.delegate_file_paths
        self.settings["delegate_call_count"] = self.curr_call_count
        self.ref = ref
        self.pte_path = context.pte_path

        self.delegate_handlers: Dict[
            str, Callable[[bytes, Dict], gb.Graph]
        ] = DELEGATE_HANDLERS

    def _get_data_from_backend_ref(
        self,
        backend_data: list[BackendDelegateInlineDataT],
        segments: list[DataSegmentT],
    ) -> bytes:
        """Retrieve backend delegate data (blob) from a BackendDelegateDataReferenceT."""
        if self.ref.location == DataLocation.INLINE:
            inline = backend_data[self.ref.index]
            if inline.data is not None:
                return bytes(inline.data)
            else:
                raise ValueError(
                    f"Inline data at index {self.ref.index} is None"
                )
        elif self.ref.location == DataLocation.SEGMENT:
            return self._resolve_segment_data(segments)
        else:
            raise ValueError(f"Unknown DataLocation {self.ref.location}")

    def _resolve_segment_data(self, segments: list[DataSegmentT]) -> bytes:
        if self.ref.index >= len(segments):
            raise IndexError(
                f"Segment index {self.ref.index} exceeds available "
                f"segments ({len(segments)})"
            )

        segment = segments[self.ref.index]
        if segment is None:
            raise ValueError(f"Segment at index {self.ref.index} is None")

        try:
            header: ExtendedHeader = read_extended_header(self.pte_path)
        except ExtendedHeaderNotFoundError as exc:
            raise ExtendedHeaderError(
                f"{self.pte_path} does not include an ExecuTorch extended header"
            ) from exc

        if header.segment_base_offset == 0:
            raise ValueError(
                "Extended header reports segment_base_offset=0 but segment data "
                "was requested"
            )

        absolute_offset = header.segment_base_offset + segment.offset
        if (
            header.segment_data_size
            and segment.offset + segment.size > header.segment_data_size
        ):
            raise ValueError(
                f"Segment range ({segment.offset}, {segment.size}) exceeds "
                f"segment_data_size {header.segment_data_size}"
            )

        with self.pte_path.open("rb") as fp:
            fp.seek(absolute_offset)
            blob = fp.read(segment.size)

        if len(blob) != segment.size:
            raise ValueError(
                f"Segment {self.ref.index} expected {segment.size} bytes but "
                f"only read {len(blob)}"
            )
        return blob

    def _delegate_blob_to_graph(
        self,
        blob: bytes,
    ) -> gb.Graph | None:
        if self.label in self.delegate_handlers and blob:
            delegate_graph = self.delegate_handlers[self.label](
                blob, self.settings
            )
            return delegate_graph

        return None

    def get_delegate_graph(
        self,
        backend_data: list[BackendDelegateInlineDataT] | None,
        segments: list[DataSegmentT] | None,
    ) -> gb.Graph | None:
        """Retrieve delegate graph from backend delegate data reference."""

        backend_data = backend_data or []
        segments = segments or []

        blob = self._get_data_from_backend_ref(backend_data, segments)
        return self._delegate_blob_to_graph(blob)
