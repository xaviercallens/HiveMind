"""IP-179: no test module may change the process environment at import time.

A module-level ``os.environ["CUDA_VISIBLE_DEVICES"] = ""`` hid the GPU from every module pytest collected after it
(test_kv_lattice, test_rtx_gpu_intense_stress silently ran their CPU branch on a CUDA host).
"""

from __future__ import annotations

import os

import conftest


def test_cuda_visible_devices_is_unchanged_by_collecting_the_suite() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == conftest.CUDA_VISIBLE_DEVICES_AT_CONFTEST_IMPORT
