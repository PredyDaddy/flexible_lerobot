#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

from tqdm import tqdm


API_ROOT = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o"
DOWNLOAD_ROOT = "https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o"


def list_objects(prefix: str) -> list[dict]:
    objects: list[dict] = []
    page_token = None
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        url = API_ROOT + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        objects.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def download_object(name: str, dest: Path, *, force: bool = False) -> None:
    size = int(next_obj_size[name])
    if dest.exists() and dest.stat().st_size == size and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded_name = urllib.parse.quote(name, safe="")
    url = f"{DOWNLOAD_ROOT}/{encoded_name}?alt=media"
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        with tqdm(total=size, unit="B", unit_scale=True, desc=name.rsplit("/", 1)[-1] or name) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                pbar.update(len(chunk))


next_obj_size: dict[str, int] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official OpenPI pi05_base JAX params from public GCS.")
    parser.add_argument(
        "--output",
        default="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train/assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params",
    )
    parser.add_argument("--prefix", default="checkpoints/pi05_base/params/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    objects = list_objects(args.prefix)
    if not objects:
        raise RuntimeError(f"No objects found for prefix {args.prefix}")

    global next_obj_size
    next_obj_size = {obj["name"]: int(obj.get("size", 0)) for obj in objects}

    for obj in objects:
        name = obj["name"]
        rel = name.removeprefix(args.prefix)
        if not rel:
            continue
        download_object(name, output / rel, force=args.force)

    print(f"Downloaded {len(objects)} objects to {output}")


if __name__ == "__main__":
    main()
