"""python -m src.indexing.build — DB 구축 진입점 (05).

기본: config의 docs_in PDF들을 파싱→IR→인덱스로 빌드.
증분: --add <pdf> / --remove <doc_id> / --list.
실제 파싱(Upstage)·임베딩(sentence-transformers) 호출. 키·모델 준비 후 실행.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from src.config import Config
from src.indexing.store import IndexStore, default_embedder
from src.parser import parse_pdf, upstage_to_ir
from src.schema import ParsedDoc


def _log(step, total, name, status="ok"):
    # 비TTY(Claude Code)에선 한 줄 append (덮어쓰기 bar 금지)
    print(f"[{step}/{total}] {name} ... {status}")


def _doc_id(pdf: str) -> str:
    return os.path.splitext(os.path.basename(pdf))[0]


def _parse_to_ir(cfg: Config, pdf: str) -> ParsedDoc:
    resp = parse_pdf(
        pdf,
        api_key_env=cfg.get("parser.api_key_env", "UP_TOKEN"),
        endpoint=cfg.get("parser.endpoint"),
        model=cfg.get("parser.model", "document-parse"),
        base64_categories=cfg.get("parser.base64_categories"),
    )
    return upstage_to_ir(resp, doc_id=_doc_id(pdf), title=_doc_id(pdf),
                         images_dir=cfg.path("images_out"))


def _get_ir(cfg: Config, pdf: str) -> ParsedDoc:
    """P5: 캐시된 IR(data/ir/<doc_id>.json)이 PDF보다 최신이면 재사용 → Upstage 중복 호출 방지.
    없거나 PDF가 더 최신이면 파싱하고 캐시에 저장(`/rag-parse` 산출과 동일 경로)."""
    ir_path = os.path.join(cfg.path("ir_out"), f"{_doc_id(pdf)}.json")
    if os.path.exists(ir_path) and os.path.getmtime(ir_path) >= os.path.getmtime(pdf):
        with open(ir_path, encoding="utf-8") as f:
            return ParsedDoc.from_dict(json.load(f))
    doc = _parse_to_ir(cfg, pdf)
    os.makedirs(os.path.dirname(ir_path), exist_ok=True)
    with open(ir_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False)
    return doc


def _load_or_build_store(cfg: Config) -> IndexStore:
    paths = cfg.index_paths()
    model = cfg.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        return IndexStore.load(paths, model)
    except FileNotFoundError:
        return IndexStore(model)


def cmd_build(cfg: Config) -> int:
    store = IndexStore(cfg.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2"))
    embed = default_embedder(store.model_name)
    pdfs = sorted(glob.glob(os.path.join(cfg.path("docs_in"), "*.pdf")))
    if not pdfs:
        print(f"⚠️ 입력 PDF 없음: {cfg.path('docs_in')}. split_pdf로 준비하세요.", file=sys.stderr)
        return 2
    for i, pdf in enumerate(pdfs, 1):
        store.add_doc(_get_ir(cfg, pdf), embed)
        _log(i, len(pdfs), os.path.basename(pdf))
    store.save(cfg.index_paths())
    print(f"[05_indexing] built {len(store.doc_ids())} docs, {len(store.chunks)} chunks.")
    return 0


def cmd_add(cfg: Config, pdf: str) -> int:
    store = _load_or_build_store(cfg)
    embed = default_embedder(store.model_name)
    store.add_doc(_get_ir(cfg, pdf), embed)
    store.save(cfg.index_paths())
    print(f"[05_indexing] added. now {len(store.doc_ids())} docs: {store.doc_ids()}")
    return 0


def cmd_remove(cfg: Config, doc_id: str) -> int:
    store = _load_or_build_store(cfg)
    n = store.remove_doc(doc_id)
    if n == 0:
        print(f"'{doc_id}' 없음. 현재 목록: {store.doc_ids()}")
        return 1
    store.save(cfg.index_paths())
    print(f"[05_indexing] removed {n} chunks. now: {store.doc_ids()}")
    return 0


def cmd_list(cfg: Config, as_json: bool = False) -> int:
    store = _load_or_build_store(cfg)
    st = store.status()
    if as_json:
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0
    # 사람용 현황 표
    if not st["docs"]:
        print("색인된 문서가 없습니다. `--add <pdf>` 또는 빌드로 문서를 추가하세요.")
        return 0
    print(f"색인 현황 — 문서 {st['total_docs']}개 · 청크 {st['total_chunks']} · figure {st['total_figures']}")
    print(f"{'doc_id':<34} {'pages':>9} {'chunks':>7} {'figs':>5} {'tables':>7} {'imgs':>5}")
    print("-" * 72)
    for did, d in st["docs"].items():
        print(f"{did:<34} {d['pages']:>9} {d['chunks']:>7} {d['figures']:>5} {d['tables']:>7} {d['images']:>5}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--add"); ap.add_argument("--remove"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", action="store_true", help="--list 현황을 JSON으로")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.add:
        return cmd_add(cfg, args.add)
    if args.remove:
        return cmd_remove(cfg, args.remove)
    if args.list:
        return cmd_list(cfg, as_json=args.json)
    return cmd_build(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
