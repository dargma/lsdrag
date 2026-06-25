"""대표(hero) 이미지를 OpenAI 최신 이미지 모델로 생성 → docs/hero.png.

담는 개념: 사용자 시작점 + agentic 에이전트(여러 단계 도구 호출) + page index(page/figure/table)
+ 멀티모달(텍스트·표·그림) + HW 문서/설계. 탑 AI 논문 teaser 스타일.
"""
import base64
import os
import sys

import requests

KEY = os.environ["OPENAI_API_KEY"]

PROMPT = (
    "A polished teaser figure for a top-tier AI systems paper, modern flat-vector design with tasteful "
    "gradients, rounded modules, crisp icons and soft shadows, confident accent palette (indigo, teal, "
    "coral, amber) on clean white. Left-to-right data flow. NO cartoon mascots or faces. "
    "LEFT: a USER icon starting a query and able to add/remove documents. "
    "Then a hardware technical manual showing three modalities together — a line of text, a small REGISTER "
    "bitfield table, and a figure/diagram thumbnail. "
    "CENTER: a glowing AGENT module (agentic RAG) shown as a hub that runs MULTIPLE NUMBERED STEPS in a loop, "
    "autonomously calling labeled tool chips: 'page-index lookup', 'keyword search', 'read chunk', and "
    "'read figure (vision/VLM)'; a small circular loop arrow suggests repetition. Below it a database "
    "cylinder labeled with three location chips PAGE / FIGURE / TABLE (a page index), with an arrow showing "
    "the agent reads it. "
    "RIGHT: a grounded answer card that pinpoints an exact page and figure and fuses text + a register "
    "bitfield + the read diagram, beside a sleek microchip die icon for hardware design. "
    "Clean thin typography, balanced grid composition, energetic yet rigorous. No real logos."
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
