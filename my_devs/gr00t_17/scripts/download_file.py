#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import requests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable file download with SHA-256 verification.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")

    if output.exists():
        actual = sha256_file(output)
        if actual != args.sha256:
            raise RuntimeError(f"Existing file hash mismatch: {actual} != {args.sha256}")
        print(f"[OK] already downloaded and verified: {output}")
        return

    for attempt in range(1, args.retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                args.url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(60, 600),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                mode = "ab" if offset and response.status_code == 206 else "wb"
                expected_remaining = response.headers.get("content-length", "unknown")
                print(
                    f"[DOWNLOAD] attempt={attempt} status={response.status_code} "
                    f"resume_offset={offset} remaining={expected_remaining}",
                    flush=True,
                )
                with partial.open(mode) as handle:
                    downloaded = offset
                    last_report = downloaded
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 64 * 1024 * 1024:
                            print(f"[DOWNLOAD] {downloaded / 1024**2:.1f} MiB", flush=True)
                            last_report = downloaded
            actual = sha256_file(partial)
            if actual != args.sha256:
                raise RuntimeError(f"Downloaded file hash mismatch: {actual} != {args.sha256}")
            partial.rename(output)
            print(f"[OK] downloaded and verified: {output}")
            return
        except Exception as exc:
            if attempt == args.retries:
                raise
            delay = min(5 * attempt, 30)
            print(f"[WARN] download attempt {attempt} failed: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)


if __name__ == "__main__":
    main()
