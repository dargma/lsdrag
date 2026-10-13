"""Persist real MinerU 4.x results, including officially materialized assets."""
import argparse
import json
from pathlib import Path
import time

from mineru.config import VlmConfig
from mineru.parser import MinerUParser
from mineru.parser.writer import FileBasedDataWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:8101")
    p.add_argument("--model", default="mineru-2604")
    p.add_argument("--pages", default="all")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    parser = MinerUParser(tier="advanced", image_analysis=False, vlm_config=VlmConfig(server_url=args.server, model=args.model, max_concurrency=8))
    files = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if args.limit:
        files = files[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        dest = args.output / path.stem
        start = time.monotonic()
        record = {"document_id": path.stem, "pages_requested": args.pages, "model": args.model, "image_analysis": False, "status": "failed"}
        try:
            result = parser.parse(path, page_range=args.pages)
            result.save(FileBasedDataWriter(str(dest)))
            middle = json.loads((dest / "middle_json.json").read_text())
            record.update(status="ok", parsed_pages=len(result.pages), saved_files=len(list(dest.rglob("*"))), schema_version=middle.get("schema_version"))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["seconds"] = time.monotonic() - start
        with (args.output / "parse_log.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
