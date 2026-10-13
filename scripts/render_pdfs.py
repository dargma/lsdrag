"""Render corpus URLs with Chromium, retaining source and image-load audit.

Uses M3DocVQA's screen-media PDF approach. HTTP and image failures are logged,
not silently treated as valid evidence. All paths are on the local work disk.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from playwright.async_api import async_playwright
from pypdf import PdfReader


async def render(args):
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.workers)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        async def one(row):
            async with gate:
                start = time.monotonic()
                path = args.output / (row["document_id"] + ".pdf")
                audit_path = path.with_suffix(".json")
                if path.exists() and audit_path.exists() and not args.force:
                    record = json.loads(audit_path.read_text())
                    if record.get("status") == "ok" and record.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
                        print(json.dumps({"document_id": row["document_id"], "status": "cached"}), flush=True)
                        return
                record = {**row, "rendered_at": datetime.now(timezone.utc).isoformat(), "renderer": "Chromium screen media", "status": "failed"}
                page = await browser.new_page(viewport={"width": 1280, "height": 900})
                try:
                    response = await page.goto(row["url"], wait_until="load", timeout=60000)
                    if response is None or response.status >= 400:
                        raise RuntimeError(f"HTTP {response.status if response else 'no response'}")
                    await page.evaluate("""async () => {
                      for (const img of document.images) img.loading = 'eager';
                      await Promise.race([
                        Promise.all(Array.from(document.images).map(img => img.decode().catch(() => {}))),
                        new Promise(resolve => setTimeout(resolve, 15000))
                      ]);
                    }""")
                    record["images"] = await page.evaluate("""() => Array.from(document.images).map(i => ({src:i.currentSrc || i.src, alt:i.alt, loaded:i.complete && i.naturalWidth>0, width:i.naturalWidth, height:i.naturalHeight}))""")
                    record["image_load_failures"] = sum(not im["loaded"] for im in record["images"])
                    await page.emulate_media(media="screen")
                    tmp = path.with_suffix(".partial.pdf")
                    await page.pdf(path=str(tmp), print_background=True)
                    reader = PdfReader(tmp)
                    record["pages"] = len(reader.pages)
                    text = "\n".join(p.extract_text() or "" for p in reader.pages)
                    if not reader.pages or len(text.strip()) < 100:
                        raise RuntimeError("Empty or non-content PDF")
                    record["text_characters"] = len(text)
                    record["sha256"] = hashlib.sha256(tmp.read_bytes()).hexdigest()
                    record["bytes"] = tmp.stat().st_size
                    record["final_url"] = page.url
                    # A PDF with missing images remains auditable but is not valid for final QC.
                    record["status"] = "ok" if not record["image_load_failures"] else "image_load_incomplete"
                    tmp.replace(path)
                    path.with_suffix(".txt").write_text(text)
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                finally:
                    await page.close()
                record["seconds"] = time.monotonic() - start
                audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                with args.log.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k in {"document_id", "status", "pages", "seconds", "error", "image_load_failures"}}), flush=True)
        await asyncio.gather(*(one(row) for row in rows))
        await browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(render(args))


if __name__ == "__main__":
    main()
