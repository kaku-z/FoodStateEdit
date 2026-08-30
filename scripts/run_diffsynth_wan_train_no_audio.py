#!/usr/bin/env python3
"""Run the frozen DiffSynth Wan trainer without requiring its unused audio stack."""

from __future__ import annotations

import hashlib
import json
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


def install_imageio_pyav_metadata_compatibility() -> None:
    """Supply missing container fps metadata without changing frame decoding."""
    import imageio

    original_get_reader = imageio.get_reader
    if getattr(original_get_reader, "_foodstateedit_pyav_metadata_compat", False):
        return

    def compatible_get_reader(*args, **kwargs):
        reader = original_get_reader(*args, **kwargs)
        original_get_meta_data = reader.get_meta_data

        def compatible_get_meta_data(index=None):
            if index is not None:
                return original_get_meta_data(index=index)
            try:
                metadata = dict(original_get_meta_data())
            except (TypeError, ZeroDivisionError):
                metadata = {}
            if metadata.get("fps"):
                return metadata

            stream = getattr(getattr(reader, "instance", None), "_video_stream", None)
            if stream is None:
                raise RuntimeError("FoodStateEdit could not inspect the PyAV video stream")
            rate = next(
                (
                    candidate
                    for candidate in (
                        getattr(stream, "average_rate", None),
                        getattr(stream, "guessed_rate", None),
                        getattr(stream, "base_rate", None),
                    )
                    if candidate is not None and float(candidate) > 0
                ),
                None,
            )
            if rate is None:
                raise RuntimeError("FoodStateEdit could not derive a positive PyAV frame rate")
            fps = float(rate)
            frame_count = int(getattr(stream, "frames", 0) or reader.count_frames())
            if frame_count <= 0:
                raise RuntimeError("FoodStateEdit could not derive a positive PyAV frame count")
            metadata.update({"fps": fps, "duration": frame_count / fps, "nframes": frame_count})
            return metadata

        reader.get_meta_data = compatible_get_meta_data
        return reader

    compatible_get_reader._foodstateedit_pyav_metadata_compat = True
    imageio.get_reader = compatible_get_reader


def run_video_decode_smoke(path: Path) -> None:
    """Exercise the same metadata and pixel decode path used by LoadVideo."""
    import imageio

    if not path.is_file():
        raise FileNotFoundError(path)
    install_imageio_pyav_metadata_compatibility()
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        frame_count = int(reader.count_frames())
        if frame_count <= 0 or float(metadata["fps"]) <= 0:
            raise RuntimeError("Video decode smoke requires positive frame count and fps")
        first = reader.get_data(0)
        last = reader.get_data(frame_count - 1)
        result = {
            "path": str(path.resolve()),
            "fps": float(metadata["fps"]),
            "frame_count": frame_count,
            "first_shape": list(first.shape),
            "last_shape": list(last.shape),
            "first_dtype": str(first.dtype),
            "last_dtype": str(last.dtype),
            "first_frame_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "last_frame_sha256": hashlib.sha256(last.tobytes()).hexdigest(),
        }
    finally:
        reader.close()
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--video-decode-smoke":
        run_video_decode_smoke(Path(sys.argv[2]).resolve())
        return

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
    install_imageio_pyav_metadata_compatibility()
    sys.argv = [str(upstream_script), *forwarded]
    runpy.run_path(str(upstream_script), run_name="__main__")


if __name__ == "__main__":
    main()
