# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from model_explorer import graph_builder as gb

from ..executorch_flatbuffer import (
    EValueT,
    KernelTypes,
    Program,
)


def safe_decode(value: Any, default: str = "") -> str:
    """Safely decode a value to a string.

    Handles bytes, None, and other types.

    Args:
        value: The value to decode.
        default: Default string if value is None.

    Returns:
        Decoded string or default.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def enum_name(enum_cls, value: int) -> str:
    for attr, attr_value in vars(enum_cls).items():
        if attr.startswith("_"):
            continue
        if attr_value == value:
            return attr
    return str(value)


def load_program(pte_path: Path) -> Program:
    data = pte_path.read_bytes()
    if not Program.ProgramBufferHasIdentifier(data, 0):
        raise ValueError(
            f"{pte_path} does not contain a valid ExecuTorch Program flatbuffer"
        )
    return Program.GetRootAs(data, 0)


def dict_to_key_value_list(dict: Dict[str, Any]) -> List[gb.KeyValue]:
    """Convert a dictionary to a list of key-value pairs."""
    result = []

    for key, value in dict.items():
        mod = getattr(value.__class__, "__module__", "")
        if isinstance(value, str):
            v_str = value
        elif isinstance(value, (bytes, bytearray)):
            v_str = safe_decode(value)
        elif isinstance(value, Iterable):
            v_str = f"[{', '.join(str(v) for v in value)}]"
        elif not mod.startswith("builtins"):
            v_str = value.__class__.__name__
        else:
            v_str = str(value)
        result.append(gb.KeyValue(key=key, value=v_str))
    return result


def is_tensor_output_candidate(evalue: EValueT) -> bool:
    if evalue is None or evalue.val is None:
        return False

    if evalue.valType == KernelTypes.Tensor:
        tensor = evalue.val
        return tensor is not None and int(tensor.dataBufferIdx) == 0

    return evalue.valType in (
        KernelTypes.TensorList,
        KernelTypes.OptionalTensorList,
    )


def to_int_list(values: Any) -> List[int]:
    result = []
    for item in _to_list(values):
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _to_list(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    if hasattr(values, "tolist"):
        converted = values.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [values]


def format_sequence(values: Any) -> str:
    items = _to_list(values)
    if not items:
        return "[]"
    formatted = [_format_scalar(item) for item in items]
    return "[" + ", ".join(formatted) + "]"


def _format_scalar(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return safe_decode(value)
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return _format_scalar(value.item())
        except Exception:
            pass
    return str(value)


def get_from_optional_listT(
    input_listT: Optional[List[Any]], index: Optional[int]
) -> Optional[Any]:
    if input_listT:
        listT = input_listT or []
        if index is not None and 0 <= index < len(listT):
            return listT[index]
    return None
