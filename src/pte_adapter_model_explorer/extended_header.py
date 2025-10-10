# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ExtendedHeaderError(RuntimeError):
    """Raised when an extended header exists but cannot be parsed."""


class ExtendedHeaderNotFoundError(ExtendedHeaderError):
    """Raised when a .pte file does not include an extended header."""


# Extended header format for PTE files as per
# https://docs.pytorch.org/executorch/1.0/pte-file-format.html
@dataclass(frozen=True)
class ExtendedHeader:
    """Parsed ExecuTorch extended header metadata."""

    program_size: int
    segment_base_offset: int
    segment_data_size: int

    MAGIC = b"eh00"
    NUM_HEAD_BYTES = 64
    HEADER_OFFSET = 8
    MAGIC_SIZE = 4
    SIZE_FIELD_SIZE = 4
    U64_SIZE = 8
    MIN_HEADER_SIZE = (
        MAGIC_SIZE + SIZE_FIELD_SIZE + 2 * U64_SIZE
    )  # No segment_data_size.
    FULL_HEADER_SIZE = MIN_HEADER_SIZE + U64_SIZE

    @classmethod
    def parse(cls, head: bytes) -> ExtendedHeader:
        """Parse the extended header from the provided head bytes.

        Args:
            head: First NUM_HEAD_BYTES bytes from the .pte file.
        Returns:
            Parsed ExtendedHeader.
        Raises:
            ExtendedHeaderNotFoundError: When the header magic is missing.
            ExtendedHeaderError: When the header is malformed.
        """
        if (
            len(head)
            < cls.HEADER_OFFSET + cls.MAGIC_SIZE + cls.SIZE_FIELD_SIZE
        ):
            raise ExtendedHeaderError(
                "Not enough data to parse extended header"
            )

        magic = head[cls.HEADER_OFFSET : cls.HEADER_OFFSET + cls.MAGIC_SIZE]
        if magic != cls.MAGIC:
            raise ExtendedHeaderNotFoundError(
                f"Extended header magic {magic!r} does not match {cls.MAGIC!r}"
            )

        size_offset = cls.HEADER_OFFSET + cls.MAGIC_SIZE
        header_size = int.from_bytes(
            head[size_offset : size_offset + cls.SIZE_FIELD_SIZE],
            byteorder="little",
        )
        if header_size < cls.MIN_HEADER_SIZE:
            raise ExtendedHeaderError(
                f"Extended header length {header_size} is smaller than "
                f"the minimum {cls.MIN_HEADER_SIZE}"
            )

        header_end = cls.HEADER_OFFSET + header_size
        if len(head) < header_end:
            raise ExtendedHeaderError(
                "Not enough bytes were provided to parse the extended header "
                f"(needed {header_size}, got {len(head) - cls.HEADER_OFFSET})"
            )

        program_size_offset = size_offset + cls.SIZE_FIELD_SIZE
        program_size = int.from_bytes(
            head[program_size_offset : program_size_offset + cls.U64_SIZE],
            byteorder="little",
        )
        segment_base_offset = int.from_bytes(
            head[
                program_size_offset + cls.U64_SIZE : program_size_offset
                + 2 * cls.U64_SIZE
            ],
            byteorder="little",
        )

        segment_data_size = (
            int.from_bytes(
                head[
                    program_size_offset
                    + 2 * cls.U64_SIZE : program_size_offset + 3 * cls.U64_SIZE
                ],
                byteorder="little",
            )
            if header_size >= cls.FULL_HEADER_SIZE
            else 0
        )

        return cls(
            program_size=program_size,
            segment_base_offset=segment_base_offset,
            segment_data_size=segment_data_size,
        )


def read_extended_header(pte_path: Path) -> ExtendedHeader:
    """Read and parse the extended header from the provided .pte file."""
    with pte_path.open("rb") as fp:
        head = fp.read(ExtendedHeader.NUM_HEAD_BYTES)

    if len(head) < ExtendedHeader.NUM_HEAD_BYTES:
        raise ExtendedHeaderError(
            f"{pte_path} is only {len(head)} bytes; expected at least "
            f"{ExtendedHeader.NUM_HEAD_BYTES} bytes to parse the extended header"
        )

    return ExtendedHeader.parse(head)
