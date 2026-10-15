# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Scalar rendering falls back only for expected conversion failures."""

import unittest

import numpy as np
import pytest

from pte_adapter_model_explorer.builder.util import _format_scalar

_ASSERT = unittest.TestCase()


def test_numpy_scalar_formatting():
    _ASSERT.assertEqual(_format_scalar(np.int32(42)), "42")


def test_numpy_array_falls_back_to_text():
    _ASSERT.assertEqual(_format_scalar(np.array([1, 2])), "[1 2]")


def test_unexpected_item_failure_is_not_silenced():
    class BrokenScalar:
        def item(self):
            raise RuntimeError("unexpected conversion failure")

    with pytest.raises(RuntimeError, match="unexpected conversion failure"):
        _format_scalar(BrokenScalar())
