# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from model_explorer import graph_builder as gb

from ..executorch_flatbuffer import (
    EValueT,
    InstructionArguments,
    InstructionT,
)
from .metadata import metadata_for_evalue
from .util import (
    dict_to_key_value_list,
    enum_name,
    get_from_optional_listT,
    is_tensor_output_candidate,
    safe_decode,
    to_int_list,
)


@dataclass
class MetadataContext:
    metadata_cache: Dict[int, List[gb.KeyValue]]
    tensor_producers: Dict[int, str]
    outputs_seen: Set[int]


def get_instruction_attributes(instruction: InstructionT) -> List[gb.KeyValue]:
    instr_type = instruction.instrArgsType
    instr_name = enum_name(InstructionArguments, instr_type)

    attrs = [gb.KeyValue(key="Instruction Type", value=instr_name)]

    attributes = instruction.instrArgs.__dict__
    attributes["instrArgsType"] = safe_decode(instr_type)
    parsed_attrs = dict_to_key_value_list(attributes)

    for parsed_attr in parsed_attrs:
        attrs.append(parsed_attr)

    return attrs


def get_output_metadata(
    node_id: str,
    evalues: List[EValueT],
    evalue_indices: List[int],
    metadata_ctx: MetadataContext,
) -> List[gb.MetadataItem]:
    output_metadata: List[gb.MetadataItem] = []

    for evalue, evalue_idx in zip(evalues, evalue_indices, strict=False):
        if (
            is_tensor_output_candidate(evalue)
            and evalue_idx not in metadata_ctx.tensor_producers
            and evalue_idx not in metadata_ctx.outputs_seen
        ):
            metadata_item = metadata_for_evalue(
                evalue, evalue_idx, metadata_ctx.metadata_cache
            )
            output_metadata.append(metadata_item)
            metadata_ctx.tensor_producers[evalue_idx] = node_id
            metadata_ctx.outputs_seen.add(evalue_idx)

    return output_metadata


def get_input_metadata(
    evalues: List[EValueT],
    evalue_indices: List[int],
    metadata_ctx: MetadataContext,
):
    inputs_metadata: List[gb.MetadataItem] = []

    for evalue, evalue_idx in zip(evalues, evalue_indices, strict=False):
        if evalue_idx in metadata_ctx.outputs_seen:
            continue
        if (
            not is_tensor_output_candidate(evalue)
            or evalue_idx in metadata_ctx.tensor_producers
        ):
            metadata_item = metadata_for_evalue(
                evalue, evalue_idx, metadata_ctx.metadata_cache
            )
            inputs_metadata.append(metadata_item)

    return inputs_metadata


def get_incoming_edges(
    evalues: List[EValueT],
    evalue_indices: List[int],
    metadata_ctx: MetadataContext,
):
    incoming_edges: List[gb.IncomingEdge] = []
    for evalue, evalue_idx in zip(evalues, evalue_indices, strict=False):
        if evalue_idx in metadata_ctx.outputs_seen:
            continue
        if (
            not is_tensor_output_candidate(evalue)
            or evalue_idx in metadata_ctx.tensor_producers
        ):
            producer = metadata_ctx.tensor_producers.get(evalue_idx)
            if producer:
                incoming_edges.append(
                    gb.IncomingEdge(
                        sourceNodeId=producer,
                        sourceNodeOutputId=str(evalue_idx),
                        targetNodeInputId=str(evalue_idx),
                    )
                )

    return incoming_edges


def get_filtered_instruction_evalues_and_indices(
    evalues: Optional[List[EValueT]], evalue_indices: List[int]
) -> Tuple[List[EValueT], List[int]]:
    """
    Filters out:
    1. None indices from evalue_indices.
    2. Evalues that are None based on the filtered indices.
    Returns a tuple of the filtered evalues and their indices."""
    evalue_indices = [
        evalue_index
        for evalue_index in evalue_indices
        if evalue_index is not None
    ]
    evalue_indices = list(
        filter(
            lambda idx: get_from_optional_listT(evalues, idx) is not None,
            evalue_indices,
        )
    )
    # Type ignore is needed here because mypy cannot infer that the filter above ensures evalue is not None
    evalues_filtered: List[EValueT] = [
        get_from_optional_listT(evalues, evalue_idx)
        for evalue_idx in evalue_indices
    ]  # type: ignore

    return evalues_filtered, evalue_indices


def get_instruction_evalue_indices(instruction: InstructionT) -> List[int]:
    args = instruction.instrArgs
    instr_type = instruction.instrArgsType

    ret = []

    if args is None:
        return ret

    if (
        instr_type == InstructionArguments.KernelCall
        or instr_type == InstructionArguments.DelegateCall
    ):
        ret = to_int_list(getattr(args, "args", []))
    if instr_type == InstructionArguments.MoveCall:
        ret = [getattr(args, "moveFrom", 0), getattr(args, "moveTo", 0)]
    if instr_type == InstructionArguments.JumpFalseCall:
        ret = [getattr(args, "condValueIndex", 0)]
    if instr_type == InstructionArguments.FreeCall:
        ret = [getattr(args, "valueIndex", 0)]

    return ret
