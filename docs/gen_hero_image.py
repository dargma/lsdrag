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
    "A polished, vibrant system-architecture figure in the style of a top-tier AI lab paper (ByteDance / "
    "DeepMind / OpenAI), modern flat-vector design with tasteful gradients, rounded modules, crisp icons, "
    "soft shadows and a confident accent palette (indigo, teal, coral, amber) on a clean white background. "
    "Professional and visually rich but uncluttered. NO cartoon mascots or faces — labeled modules and "
    "directed data-flow arrows, left to right. "
    "LEFT — a hardware technical manual page showing THREE modalities together: a paragraph of text, a small "
    "REGISTER bitfield table, and a figure/diagram thumbnail (multimodal input). "
    "MIDDLE-LOWER — a 'Parse' block feeding an 'Index' database cylinder labeled with three location chips: "
    "PAGE, FIGURE, TABLE (a page-index that pinpoints locations). "
    "MIDDLE-TOP — a glowing central 'Agent' module (agentic RAG) depicted as a hub that autonomously routes to "
    "three tool chips with mini icons: 'semantic search', 'page-index lookup', and 'read figure (vision/VLM)'. "
    "RIGHT — a highlighted result card that pinpoints an exact page and a numbered figure, fusing text + a "
    "register bitfield table + the read diagram into one grounded answer, beside a sleek microchip die icon "
    "for hardware design. Subtle connecting arrows from the Index to the Agent (the agent reads the index). "
    "Clean thin typography, balanced composition, high resolution, the energetic yet rigorous aesthetic of a "
    "flagship AI systems paper teaser figure. No real logos."
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
