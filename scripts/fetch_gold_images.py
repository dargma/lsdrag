"""Fetch only selected original MMQA images via HTTP Range, for evaluation QC.

Gold images must not become retrieval corpus assets. Zip directory and per-file
range requests avoid downloading the complete 2.2 GiB reference archive.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import requests


URL = "https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip"


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        response = self.session.head(url, timeout=30)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.bytes_read = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 32 * 1024 * 1024:
            raise ValueError("Refusing an unexpectedly large range read")
        start = self.position
        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{start+size-1}"}, timeout=60, stream=True)
        try:
            if response.status_code != 206:
                raise RuntimeError(f"Range unsupported: HTTP {response.status_code}")
            result = response.raw.read(size + 1)
            if len(result) != size:
                raise IOError(f"Expected {size} bytes, received {len(result)}")
        finally:
            response.close()
        self.position += size
        self.bytes_read += size
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    refs = [json.loads(s) for s in args.sources.read_text().splitlines()]
    if args.limit:
        refs = refs[:args.limit]
    remote = RangeReader(URL)
    with zipfile.ZipFile(remote) as archive:
        members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        for ref in refs:
            target = args.output / Path(ref["path"]).name
            record = {"id": ref["id"], "path": str(target), "source": URL, "status": "failed"}
            try:
                if not target.exists():
                    data = archive.read(members[Path(ref["path"]).name])
                    target.write_bytes(data)
                record.update(status="ok", bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            with (args.output / "download_log.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in ("id", "status")}), flush=True)
    print(json.dumps({"archive_bytes": remote.size, "transferred_bytes": remote.bytes_read}))


if __name__ == "__main__":
    main()
