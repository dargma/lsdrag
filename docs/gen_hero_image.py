"""OpenAI 이미지 API로 대표 그림(hero) 생성 → docs/hero.png.

담는 개념: agentic RAG(A-RAG) + Page Index(페이지·figure·표 위치 특정) +
멀티모달(다이어그램 읽기) + HW 문서·반도체 설계 맥락.
"""
import base64
import os
import sys

import requests

KEY = os.environ["OPENAI_API_KEY"]

PROMPT = (
    "A clean academic system-architecture diagram in the style of a Google Research / DeepMind paper figure. "
    "Flat schematic vector style, thin precise strokes, generous white background, restrained muted palette "
    "(slate gray, muted blue, soft green accents), subtle drop shadows, professional and minimal. NO cartoon "
    "characters, NO mascots, NO faces — purely a technical schematic with labeled rectangular modules and "
    "directed arrows showing data flow left to right. "
    "Left module: a hardware technical document / datasheet page icon containing a small register table and a "
    "bitfield row. Arrow into a 'Parse' block, then an 'Index' block depicted as a database cylinder annotated "
    "with page/figure/table location tags. A central rounded module labeled like an agentic retriever, with "
    "three thin branch arrows to small tool chips reading 'locate page', 'find figure', 'read diagram (vision)'. "
    "Right side: a clean result card pinpointing a specific page and figure, next to a small highlighted "
    "register bitfield diagram and a minimalist microchip die icon representing hardware design. "
    "Crisp thin typography for labels, isometric-free flat layout, balanced grid composition, high resolution, "
    "the sober aesthetic of a top-tier ML conference system figure. No real logos."
)


def gen(model, size):
    r = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={"model": model, "prompt": PROMPT, "size": size, "n": 1},
        timeout=300,
    )
    if r.status_code != 200:
        print(f"[{model}] {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return None
    d = r.json()["data"][0]
    if d.get("b64_json"):
        return base64.b64decode(d["b64_json"])
    if d.get("url"):
        return requests.get(d["url"], timeout=120).content
    return None


out = os.path.join(os.path.dirname(__file__), "hero.png")
for model, size in [("gpt-image-1", "1536x1024"), ("dall-e-3", "1792x1024")]:
    img = gen(model, size)
    if img:
        with open(out, "wb") as f:
            f.write(img)
        print(f"wrote {out} via {model} ({len(img)//1024} KB)")
        break
else:
    print("이미지 생성 실패 — 키 권한/모델 접근 확인", file=sys.stderr)
    sys.exit(1)
