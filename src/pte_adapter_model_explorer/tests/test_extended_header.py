# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from __future__ import annotations

import pytest

from ..extended_header import (
    ExtendedHeader,
    ExtendedHeaderError,
    ExtendedHeaderNotFoundError,
    read_extended_header,
)


def _make_head(
    *,
    header_size: int,
    program_size: int,
    segment_base_offset: int,
    segment_data_size: int = 0,
) -> bytes:
    """Build a 64-byte head that carries an ExecuTorch extended header."""
    head = bytearray(64)

    head[8:12] = b"eh00"
    head[12:16] = header_size.to_bytes(4, "little")
    head[16:24] = program_size.to_bytes(8, "little")
    head[24:32] = segment_base_offset.to_bytes(8, "little")
    head[32:40] = segment_data_size.to_bytes(8, "little")

    return bytes(head)


def test_parse_minimal_header_assigns_zero_segment_size():
    head = _make_head(
        header_size=24,
        program_size=0x1234,
        segment_base_offset=0x5678,
        segment_data_size=0x9ABC,
    )

    header = ExtendedHeader.parse(head)

    assert header.program_size == 0x1234
    assert header.segment_base_offset == 0x5678
    assert header.segment_data_size == 0


def test_parse_full_header_reads_segment_data_size():
    head = _make_head(
        header_size=40,
        program_size=0x10,
        segment_base_offset=0x20,
        segment_data_size=0x30,
    )

    header = ExtendedHeader.parse(head)

    assert header.program_size == 0x10
    assert header.segment_base_offset == 0x20
    assert header.segment_data_size == 0x30


def test_parse_header_missing_magic_raises_not_found():
    head = bytearray(
        _make_head(
            header_size=24,
            program_size=0,
            segment_base_offset=0,
        )
    )
    head[8:12] = b"bad!"

    with pytest.raises(ExtendedHeaderNotFoundError):
        ExtendedHeader.parse(bytes(head))


def test_parse_header_truncated_payload_raises():
    head = _make_head(
        header_size=40,
        program_size=0,
        segment_base_offset=0,
        segment_data_size=0,
    )
    truncated = head[:47]

    with pytest.raises(ExtendedHeaderError):
        ExtendedHeader.parse(truncated)


def test_read_extended_header_requires_full_head(tmp_path):
    pte_path = tmp_path / "short.pte"
    pte_path.write_bytes(b"\x00" * (63))

    with pytest.raises(ExtendedHeaderError):
        read_extended_header(pte_path)


def test_read_extended_header_returns_parsed_header(tmp_path):
    pte_path = tmp_path / "with_header.pte"
    head = _make_head(
        header_size=40,
        program_size=0xFF,
        segment_base_offset=0x100,
        segment_data_size=0x200,
    )
    pte_path.write_bytes(head)

    header = read_extended_header(pte_path)

    assert header.program_size == 0xFF
    assert header.segment_base_offset == 0x100
    assert header.segment_data_size == 0x200
