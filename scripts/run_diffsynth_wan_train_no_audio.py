#!/usr/bin/env python3
"""Run the frozen DiffSynth Wan trainer without requiring its unused audio stack."""

from __future__ import annotations

import runpy
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path


def parse_wrapper_args(argv: list[str]) -> tuple[Path, list[str]]:
    if len(argv) < 4 or argv[1] != "--upstream-script" or argv[3] != "--":
        raise ValueError("Usage: wrapper --upstream-script TRAIN.py -- [trainer arguments]")
    upstream_script = Path(argv[2]).resolve()
    forwarded = argv[4:]
    if not upstream_script.is_file():
        raise FileNotFoundError(upstream_script)
    return upstream_script, forwarded


def read_option(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise ValueError(f"Missing value after {option}")
    return arguments[index + 1]


def main() -> None:
    upstream_script, forwarded = parse_wrapper_args(sys.argv)
    data_file_keys = read_option(forwarded, "--data_file_keys")
    if data_file_keys is None:
        raise ValueError("Frozen no-audio wrapper requires explicit --data_file_keys")
    if "input_audio" in data_file_keys.split(","):
        raise ValueError("No-audio wrapper refuses datasets containing input_audio")

    # The upstream trainer instantiates LoadAudio in an unused operator map.
    # LoadAudio.__init__ only imports librosa; no attribute is accessed unless an
    # input_audio field is processed. A sentinel module therefore removes the
    # irrelevant dependency without changing the VACE data or training path.
    sentinel = types.ModuleType("librosa")
    sentinel.__doc__ = "FoodStateEdit no-audio sentinel; no audio operation is permitted."
    sentinel.__spec__ = ModuleSpec("librosa", loader=None)

    def forbidden_audio_load(*_args, **_kwargs):
        raise RuntimeError("FoodStateEdit no-audio wrapper forbids librosa.load")

    sentinel.load = forbidden_audio_load
    sys.modules["librosa"] = sentinel
    sys.argv = [str(upstream_script), *forwarded]
    runpy.run_path(str(upstream_script), run_name="__main__")


if __name__ == "__main__":
    main()
