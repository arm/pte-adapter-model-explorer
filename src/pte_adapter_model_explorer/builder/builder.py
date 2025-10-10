# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from model_explorer import graph_builder as gb

from ..constants import GRAPH_INPUT_ANNOTATION, GRAPH_OUTPUT_ANNOTATION
from ..executorch_flatbuffer import (
    ChainT,
    ExecutionPlanT,
    InstructionArguments,
    InstructionT,
    ProgramT,
)
from .delegate import DelegateBuildContext, DelegateGraphBuilder
from .instruction import (
    MetadataContext,
    get_filtered_instruction_evalues_and_indices,
    get_incoming_edges,
    get_input_metadata,
    get_instruction_attributes,
    get_instruction_evalue_indices,
    get_output_metadata,
)
from .metadata import metadata_for_evalue
from .util import (
    enum_name,
    get_from_optional_listT,
    load_program,
    safe_decode,
    to_int_list,
)


class PteGraphBuilder:
    """Builds Model Explorer graphs from ExecuTorch PTE FlatBuffer files."""

    def __init__(self, pte_path: Path, settings: Dict) -> None:
        self.model_name = pte_path.name
        self._program = load_program(pte_path)
        self._program_def = ProgramT.InitFromObj(self._program)
        self._delegate_context = DelegateBuildContext(
            delegate_file_paths=self._get_delegate_file_paths(settings),
            pte_path=pte_path,
        )
        self._delegate_graphs: List[gb.Graph] = []
        self.settings = settings
        self.graph_collections = self._build_graph_collections()

    def _get_delegate_file_paths(self, settings: Dict) -> List[Path]:
        """
        Get associated backend delegate files (e.g., TOSA) from settings['delegate_file_paths'].
        """
        paths = []

        def extract_tag_number(path):
            match = re.search(r"output_tag(\d+)_", path.name)
            return int(match.group(1)) if match else -1

        if "delegate_file_paths" in settings:
            paths = settings["delegate_file_paths"]
            paths = [Path(path) for path in paths]
            for path in paths:
                if not path.exists() or not path.is_file():
                    raise FileNotFoundError(
                        f"Delegate file path {path} does not exist or is not a file."
                    )

        tosa_file_paths = list(
            filter(lambda path: path.suffix == ".tosa", paths)
        )
        sorted_tosa_paths = sorted(tosa_file_paths, key=extract_tag_number)

        return sorted_tosa_paths

    def _build_graph_collections(self) -> List[gb.GraphCollection]:
        graphs: List[gb.Graph] = []
        self._delegate_graphs = []
        execution_plans = self._program_def.executionPlan or []

        for plan_index, exec_plan in enumerate(execution_plans):
            graph_id = safe_decode(
                exec_plan.name, default=f"ExecutionPlan_{plan_index}"
            )
            graphs.append(self._build_graph_nodes(exec_plan, graph_id))

        if not graphs:
            graphs.append(gb.Graph(id="ExecutionPlan_0", nodes=[]))

        graph_collections: List[gb.GraphCollection] = [
            gb.GraphCollection(label=self.model_name, graphs=graphs)
        ]
        if self._delegate_graphs:
            graph_collections.append(
                gb.GraphCollection(
                    label="delegates", graphs=self._delegate_graphs
                )
            )

        return graph_collections

    def _build_graph_nodes(
        self,
        exec_plan: ExecutionPlanT,
        graph_id: str,
    ) -> gb.Graph:
        metadata_cache: Dict[int, List[gb.KeyValue]] = {}
        tensor_producers: Dict[int, str] = {}

        nodes: List[gb.GraphNode] = []

        nodes.append(
            self._build_graph_input_node(
                exec_plan, metadata_cache, tensor_producers
            )
        )

        nodes.extend(self._build_empty_chain_nodes(graph_id, exec_plan.chains))
        nodes.extend(
            self._build_instruction_nodes(
                graph_id, exec_plan, metadata_cache, tensor_producers
            )
        )
        nodes.append(
            self._build_graph_output_node(
                exec_plan, metadata_cache, tensor_producers
            )
        )

        return gb.Graph(id=graph_id, nodes=nodes)

    def _build_graph_input_node(
        self,
        exec_plan: ExecutionPlanT,
        metadata_cache: Dict[int, List[gb.KeyValue]],
        tensor_producers: Dict[int, str],
    ) -> gb.GraphNode:
        input_indices: list[int] = to_int_list(exec_plan.inputs)
        outputs_metadata = [
            metadata_for_evalue(
                get_from_optional_listT(exec_plan.values, idx),
                idx,
                metadata_cache,
            )
            for idx in input_indices
        ]
        for idx in input_indices:
            tensor_producers[idx] = GRAPH_INPUT_ANNOTATION

        return gb.GraphNode(
            id=GRAPH_INPUT_ANNOTATION,
            label="Graph Inputs",
            namespace="GraphInputs",
            outputsMetadata=outputs_metadata,
        )

    def _build_instruction_nodes(
        self,
        graph_id: str,
        exec_plan: ExecutionPlanT,
        metadata_cache: Dict[int, List[gb.KeyValue]],
        tensor_producers: Dict[int, str],
    ) -> List[gb.GraphNode]:
        nodes: List[gb.GraphNode] = []
        chains: list[ChainT] = exec_plan.chains or []

        per_delegate_call_count = {}
        for chain_index, chain in enumerate(chains):
            instructions: list[InstructionT] = chain.instructions or []
            for instruction_index, instruction in enumerate(instructions):
                node_id = f"{graph_id}_{chain_index}_{instruction_index}"

                args = instruction.instrArgs
                label = enum_name(
                    InstructionArguments, instruction.instrArgsType
                )

                operator = get_from_optional_listT(
                    exec_plan.operators, getattr(args, "opIndex", None)
                )
                label = safe_decode(getattr(operator, "name", label))
                delegate = get_from_optional_listT(
                    exec_plan.delegates, getattr(args, "delegateIndex", None)
                )
                label = safe_decode(getattr(delegate, "id", label))
                evalues_filtered, evalue_indices_filtered = (
                    get_filtered_instruction_evalues_and_indices(
                        exec_plan.values,
                        get_instruction_evalue_indices(instruction),
                    )
                )

                metadata_ctx = MetadataContext(
                    metadata_cache=metadata_cache,
                    tensor_producers=tensor_producers,
                    outputs_seen=set(),
                )
                new_node = gb.GraphNode(
                    id=node_id,
                    label=label,
                    namespace=graph_id,
                    attrs=get_instruction_attributes(instruction),
                    incomingEdges=get_incoming_edges(
                        evalues=evalues_filtered,
                        evalue_indices=evalue_indices_filtered,
                        metadata_ctx=metadata_ctx,
                    ),
                    inputsMetadata=get_input_metadata(
                        evalues=evalues_filtered,
                        evalue_indices=evalue_indices_filtered,
                        metadata_ctx=metadata_ctx,
                    ),
                    outputsMetadata=get_output_metadata(
                        node_id=node_id,
                        evalues=evalues_filtered,
                        evalue_indices=evalue_indices_filtered,
                        metadata_ctx=metadata_ctx,
                    ),
                )

                if delegate and delegate.processed:
                    # Add a delegate subgraph to the main graph collection
                    if label not in per_delegate_call_count:
                        per_delegate_call_count[label] = 0
                    delegate_builder = DelegateGraphBuilder(
                        label,
                        self.settings,
                        delegate.processed,
                        self._delegate_context,
                        per_delegate_call_count[label],
                    )
                    delegate_graph = delegate_builder.get_delegate_graph(
                        self._program_def.backendDelegateData,
                        self._program_def.segments,
                    )
                    if delegate_graph:
                        delegate_graph.id = new_node.id
                        new_node.subgraphIds.append(delegate_graph.id)
                        self._delegate_graphs.append(delegate_graph)
                        per_delegate_call_count[label] += 1

                nodes.append(new_node)

        return nodes

    def _build_empty_chain_nodes(
        self,
        graph_id: str,
        chains: Optional[List[ChainT]],
    ) -> List[gb.GraphNode]:
        nodes: List[gb.GraphNode] = []
        chains: list[ChainT] = chains or []

        for chain_index, chain in enumerate(chains):
            instructions: list[InstructionT] = chain.instructions or []

            if not instructions:
                self._append_empty_chain_node(
                    graph_id, chain, chain_index, nodes
                )

        return nodes

    def _append_empty_chain_node(
        self,
        graph_id: str,
        chain: ChainT,
        chain_index: int,
        nodes: List[gb.GraphNode],
    ) -> None:
        chain_id = safe_decode(getattr(chain, "name", f"Chain_{chain_index}"))
        chain_node = gb.GraphNode(
            id=chain_id,
            label=graph_id,
            namespace=graph_id,
            incomingEdges=[
                gb.IncomingEdge(
                    sourceNodeId=GRAPH_INPUT_ANNOTATION,
                    sourceNodeOutputId="0",
                    targetNodeInputId="0",
                )
            ],
            attrs=[
                gb.KeyValue(key="Chain Index", value=str(chain_index)),
                gb.KeyValue(key="Instructions", value="0"),
            ],
        )

        output_node = gb.GraphNode(
            id=GRAPH_OUTPUT_ANNOTATION,
            label="Graph Outputs",
            namespace="GraphOutputs",
            incomingEdges=[
                gb.IncomingEdge(
                    sourceNodeId=chain_node.id,
                )
            ],
        )

        nodes.extend([chain_node, output_node])

    def _build_graph_output_node(
        self,
        plan,
        metadata_cache: Dict[int, List[gb.KeyValue]],
        tensor_producers: Dict[int, str],
    ) -> gb.GraphNode:
        output_indices: list[int] = to_int_list(plan.outputs)

        incoming_edges = [
            gb.IncomingEdge(
                sourceNodeId=tensor_producers.get(idx, GRAPH_INPUT_ANNOTATION),
                sourceNodeOutputId=str(idx),
                targetNodeInputId=str(idx),
            )
            for idx in output_indices
            if idx in tensor_producers
        ]

        inputs_metadata = [
            metadata_for_evalue(
                get_from_optional_listT(plan.values, idx),
                idx,
                metadata_cache,
            )
            for idx in output_indices
        ]

        return gb.GraphNode(
            id=GRAPH_OUTPUT_ANNOTATION,
            label="Graph Outputs",
            namespace="GraphOutputs",
            incomingEdges=incoming_edges,
            inputsMetadata=inputs_metadata,
        )
