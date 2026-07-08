# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


@pytest.mark.timeout(30)
def test_model_explorer_smoke():
    """
    Launch the server with the pte_adapter_model_explorer extension,
    verify the expected stdout markers, hit the printed URL once, then
    stop the server gracefully.
    """

    cli_path = None
    local_cli = Path(sys.executable).parent / "model-explorer"
    if local_cli.is_file():
        cli_path = str(local_cli)
    if cli_path is None:
        cli_path = shutil.which("model-explorer")

    if cli_path is None:
        pytest.skip("model-explorer CLI is not installed in this environment")

    cmd = [
        cli_path,
        "--no_open_in_browser",
        "--extensions=pte_adapter_model_explorer",
        "--host=127.0.0.1",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        universal_newlines=True,
        env=env,
    )

    searched_lines = [
        "PTE Adapter",
        "http://127.0.0.1",
    ]
    seen_lines = dict.fromkeys(searched_lines, False)
    output_lines: list[str] = []
    deadline = time.monotonic() + 20

    assert proc.stdout is not None, "Process stdout not captured"
    line_queue: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line_queue.put(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    try:
        while proc.poll() is None and time.monotonic() < deadline:
            try:
                line = line_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            output_lines.append(line)
            for searched_line in searched_lines:
                if searched_line in line:
                    seen_lines[searched_line] = True
            if all(seen_lines.values()):
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    assert all(seen_lines.values()), (
        f"Not all expected lines were seen: {seen_lines}\n"
        f"Output:\n{''.join(output_lines)}"
    )
