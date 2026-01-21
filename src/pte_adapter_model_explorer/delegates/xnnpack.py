# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from __future__ import annotations

from typing import Dict, List, Optional

from model_explorer import graph_builder as gb

from ..builder.util import dict_to_key_value_list, enum_name
from ..constants import GRAPH_INPUT_ANNOTATION, GRAPH_OUTPUT_ANNOTATION
from ..executorch_flatbuffer import xnnpack_generated as xg

XNNPACK_DELEGATE_NAME = "XnnpackBackend"
_XNNPACK_IDENTIFIER = b"XN01"
_FLATBUFFER_IDENTIFIER_OFFSET = 4
_METADATA_ENUM_KEYS = {
    "quantParamsType": xg.XNNQuantParams,
    "datatype": xg.XNNDatatype,
    "xvalueUnionType": xg.XValueUnion,
    "xnodeUnionType": xg.XNodeUnion,
}
_METADATA_KEY_RENAMES = {
    "dims": "shape",  # ME expects 'shape' instead of 'dims' on tensors to enable viewing on edges
}


def xnnpack_delegate_handler(blob: bytes, settings: Dict) -> gb.Graph:
    """
    Convert an XNNPACK delegate blob into a Model Explorer graph.
    """
    graph_t = _parse_xnnpack_graph(blob)
    result = _build_graph(graph_t)
    return result


def _parse_xnnpack_graph(blob: bytes) -> xg.XNNGraphT:
    """
    Parse the XNNPACK flatbuffer blob into a XNNGraphT.
    """
    try:
        fb_offset = (
            blob.index(_XNNPACK_IDENTIFIER) - _FLATBUFFER_IDENTIFIER_OFFSET
        )
    except ValueError:
        fb_offset = 0

    root = xg.XNNGraph.GetRootAs(blob, fb_offset)
    return xg.XNNGraphT.InitFromObj(root)


def _build_graph(graph: xg.XNNGraphT) -> gb.Graph:
    values = graph.xvalues or []
    value_producers: Dict[int, str] = {}
    nodes: List[gb.GraphNode] = []

    nodes.append(_build_inputs_node(graph, values, value_producers))

    for idx, xnode in enumerate(graph.xnodes or []):
        nodes.append(
            _build_node(
                node_index=idx,
                xnode=xnode,
                value_producers=value_producers,
                values=values,
            )
        )

    nodes.append(_build_outputs_node(graph, values, value_producers))

    return gb.Graph(id=XNNPACK_DELEGATE_NAME, nodes=nodes)


def _build_inputs_node(
    graph: xg.XNNGraphT,
    values: list[xg.XValueT],
    value_producers: Dict[int, str],
) -> gb.GraphNode:
    input_ids = graph.inputIds if graph.inputIds is not None else []
    for vid in input_ids:
        value_producers[vid] = GRAPH_INPUT_ANNOTATION

    return gb.GraphNode(
        id=GRAPH_INPUT_ANNOTATION,
        label="Graph Inputs",
        namespace="GraphInputs",
        outputsMetadata=[
            _metadata_item(values[vid], vid) for vid in input_ids
        ],
    )


def _build_outputs_node(
    graph: xg.XNNGraphT,
    values: list[xg.XValueT],
    value_producers: Dict[int, str],
) -> gb.GraphNode:
    output_ids = graph.outputIds if graph.outputIds is not None else []
    incoming_edges = [
        gb.IncomingEdge(
            sourceNodeId=value_producers.get(vid, GRAPH_INPUT_ANNOTATION),
            sourceNodeOutputId=str(vid),
            targetNodeInputId=str(vid),
        )
        for vid in output_ids
    ]

    return gb.GraphNode(
        id=GRAPH_OUTPUT_ANNOTATION,
        label="Graph Outputs",
        namespace="GraphOutputs",
        incomingEdges=incoming_edges,
        inputsMetadata=[
            _metadata_item(values[vid], vid) for vid in output_ids
        ],
    )


def _build_node(
    node_index: int,
    xnode: xg.XNodeT,
    value_producers: Dict[int, str],
    values: list[xg.XValueT],
) -> gb.GraphNode:
    label = enum_name(xg.XNodeUnion, xnode.xnodeUnionType)
    node_id = str(node_index)
    inputs = _collect_ids(xnode.xnodeUnion, prefix="input")
    outputs = _collect_ids(xnode.xnodeUnion, prefix="output")

    for vid in outputs:
        value_producers[vid] = node_id

    incoming_edges = [
        gb.IncomingEdge(
            sourceNodeId=value_producers.get(vid, GRAPH_INPUT_ANNOTATION),
            sourceNodeOutputId=str(vid),
            targetNodeInputId=str(i),
        )
        for i, vid in enumerate(inputs)
    ]

    return gb.GraphNode(
        id=node_id,
        label=label,
        namespace="forward",
        attrs=[
            *_metadata_attributes(xnode),
            *_metadata_attributes(xnode.xnodeUnion),
        ],
        incomingEdges=incoming_edges,
        inputsMetadata=[_metadata_item(values[vid], vid) for vid in inputs],
        outputsMetadata=[_metadata_item(values[vid], vid) for vid in outputs],
    )


def _metadata_attributes(obj: Optional[object]) -> List[gb.KeyValue]:
    if obj is None:
        return []

    result = dict_to_key_value_list(vars(obj))
    for r in result:
        if r.key in _METADATA_ENUM_KEYS and isinstance(r.value, str):
            enum_type = _METADATA_ENUM_KEYS[r.key]
            r.value = enum_name(enum_type, int(r.value))

        if r.key in _METADATA_KEY_RENAMES:
            r.key = _METADATA_KEY_RENAMES[r.key]

    return result


def _metadata_item(value: xg.XValueT, value_id: int) -> gb.MetadataItem:
    return gb.MetadataItem(
        id=str(value_id),
        attrs=[
            *_metadata_attributes(value),
            *_metadata_attributes(value.xvalueUnion),
        ],
    )


def _collect_ids(obj: object, prefix: str) -> List[int]:
    if obj is None:
        return []
    ids: List[int] = []
    for name, value in vars(obj).items():
        if not name.lower().startswith(prefix):
            continue
        if not name.endswith("Id"):
            continue
        if isinstance(value, int):
            ids.append(int(value))
    return ids
