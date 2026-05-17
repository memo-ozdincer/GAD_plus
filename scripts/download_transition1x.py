#!/usr/bin/env python
"""Download Transition1x.h5 from Figshare via the direct ndownloader host."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests


DEFAULT_URL = "https://ndownloader.figshare.com/files/36035789"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/transition1x.h5",
        help="Destination HDF5 path.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Download URL. Defaults to Figshare's direct ndownloader host.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8 * 1024 * 1024,
        help="Streaming chunk size in bytes.",
    )
    return parser.parse_args()


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = output.with_suffix(output.suffix + ".part")

    headers = {"User-Agent": "GADplus Transition1x downloader"}
    resume_from = part_path.stat().st_size if part_path.exists() else 0
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        print(f"Resuming from {_format_bytes(resume_from)}")

    with requests.get(args.url, headers=headers, stream=True, timeout=60) as response:
        if response.status_code == 202 or response.headers.get("x-amzn-waf-action"):
            raise RuntimeError(
                "Figshare returned a WAF/challenge response. Use "
                "https://ndownloader.figshare.com/files/36035789 or download from a browser."
            )
        if response.status_code == 416 and part_path.exists():
            part_path.rename(output)
            print(f"Download already complete: {output}")
            return
        response.raise_for_status()

        mode = "ab" if response.status_code == 206 and resume_from else "wb"
        if mode == "wb" and part_path.exists():
            part_path.unlink()
            resume_from = 0

        expected_header = response.headers.get("content-length")
        expected = int(expected_header) + resume_from if expected_header else None
        downloaded = resume_from
        next_report = downloaded

        with part_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=args.chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if expected:
                        pct = 100.0 * downloaded / expected
                        print(
                            f"Downloaded {_format_bytes(downloaded)} / "
                            f"{_format_bytes(expected)} ({pct:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(f"Downloaded {_format_bytes(downloaded)}", flush=True)
                    next_report = downloaded + 256 * 1024 * 1024

    if part_path.stat().st_size == 0:
        part_path.unlink()
        raise RuntimeError("Download produced an empty file.")

    with part_path.open("rb") as handle:
        magic = handle.read(len(HDF5_MAGIC))
    if magic != HDF5_MAGIC:
        raise RuntimeError(
            f"Downloaded file does not look like HDF5. First bytes: {magic!r}"
        )

    os.replace(part_path, output)
    print(f"Saved {output} ({_format_bytes(output.stat().st_size)})")


if __name__ == "__main__":
    main()
