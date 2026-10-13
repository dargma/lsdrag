"""Background copy of completed local artifacts, verified without moving sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from lsdrag.io import dump_json, sha256, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--include", action="append", required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    source_root = a.source_root.resolve()
    files = []
    for name in a.include:
        source = (source_root / name).resolve()
        if not source.is_relative_to(source_root) or not source.exists():
            raise ValueError(f"Invalid source: {name}")
        files.extend([source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()])
    files = sorted(set(p for p in files if ".partial." not in p.name and not p.name.endswith(".tmp")))
    def copy(source):
        relative = source.relative_to(source_root)
        record = {"path": str(relative), "status": "failed"}
        try:
            if shutil.disk_usage(source_root).free < 1024**3:
                raise RuntimeError("Local disk reserve below 1 GiB; backup copy deferred")
            before = sha256(source)
            target = a.destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) == before:
                record.update(status="verified_existing", sha256=before, bytes=source.stat().st_size)
                return record
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            if sha256(temporary) != before or sha256(source) != before:
                raise RuntimeError("Source changed or backup checksum mismatch")
            temporary.replace(target)
            record.update(status="copied_and_verified", sha256=before, bytes=source.stat().st_size)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        return record
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(copy, files))
    write_jsonl(a.destination / "copy_manifest.jsonl", results)
    report = {"source_root": str(source_root), "destination": str(a.destination), "includes": a.include, "files": len(results), "verified": sum(r["status"] != "failed" for r in results), "bytes_verified": sum(r.get("bytes", 0) for r in results), "failed": [r for r in results if r["status"] == "failed"], "manifest_sha256": sha256(a.destination / "copy_manifest.jsonl"), "policy": "copy only; verify source and destination hashes; no source deletion"}
    dump_json(a.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "failed"} | {"failure_count": len(report["failed"])}))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
