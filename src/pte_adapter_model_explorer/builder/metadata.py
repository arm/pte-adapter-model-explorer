# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from typing import Dict, List

from model_explorer import graph_builder as gb

from ..executorch_flatbuffer.program_generated import EValueT
from .util import dict_to_key_value_list, safe_decode


def metadata_for_evalue(
    evalue: EValueT | None,
    idx: int,
    cache: Dict[int, List[gb.KeyValue]],
) -> gb.MetadataItem:
    if idx not in cache:
        evalue_description = []

        if evalue:
            vals = evalue.val.__dict__
            valtype = evalue.valType
            vals["valType"] = safe_decode(valtype)
            evalue_description = list(dict_to_key_value_list(vals))

        cache[idx] = [
            gb.KeyValue(key="EValue Index", value=str(idx)),
            *evalue_description,
        ]
    return gb.MetadataItem(id=str(idx), attrs=list(cache[idx]))
