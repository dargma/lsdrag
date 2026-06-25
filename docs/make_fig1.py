"""AAAI 스타일 Figure 1 (티저+시스템도) → docs/fig1_teaser.png.

한 장에 담는 것:
 (위 레인, 빌드타임)  PDF → ①Parse(Upstage→IR) → ②Build(embed+Page Index) → [Index store]
 (아래 레인, 질의타임) Question → ③Search(agentic 5 tools + VLM Reader) → 정답(페이지·Figure 짚고 그림까지 읽음)
 각 단계는 'swap' 가능(부품 교체). Index store가 두 레인을 잇는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#1b2430"; MUTED = "#6b7785"; LINE = "#c2cad3"
ACC = "#2d6cdf"; GOOD = "#2f9e6b"; WARM = "#e0a23b"; STORE = "#eef3fb"
CARD = "#ffffff"; SWAP = "#8a93a0"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(12.4, 6.8), dpi=175)
ax.set_xlim(0, 124); ax.set_ylim(0, 68); ax.axis("off")


def box(x, y, w, h, fc=CARD, ec=LINE, lw=1.4, r=2.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, size=11, color=INK, weight="normal", ha="left", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, style=style, zorder=6)


def arrow(x1, y1, x2, y2, color=MUTED, lw=2.2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17,
                                 color=color, lw=lw, ls=ls, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def swap(x, y):
    txt(x, y, "swappable", size=8, color=SWAP, ha="center", style="italic")


# 제목
txt(62, 65.2, "Ask a technical manual — get the exact page, the exact figure, and the diagram read for you.",
    size=14, weight="bold", ha="center")

# ───────── 빌드타임 레인 (위) ─────────
txt(2, 57.5, "BUILD  (offline)", size=9.5, color=MUTED, weight="bold")
box(3, 44, 15, 11, CARD); txt(10.5, 51, "PDF", size=12, weight="bold", ha="center")
txt(10.5, 47.4, "manual", size=9.5, color=MUTED, ha="center")

box(24, 44, 19, 11, "#f4f8ff", ec=ACC, lw=1.7)
txt(33.5, 51.6, "① Parse", size=11.5, weight="bold", color=ACC, ha="center")
txt(33.5, 47.6, "Upstage → IR", size=9.6, ha="center")
swap(33.5, 42.0)

box(49, 44, 21, 11, "#f4f8ff", ec=ACC, lw=1.7)
txt(59.5, 51.6, "② Build", size=11.5, weight="bold", color=ACC, ha="center")
txt(59.5, 47.6, "embed + Page Index", size=9.4, ha="center")
swap(59.5, 42.0)

# Index store (중앙, 두 레인 연결)
box(78, 30, 24, 25, STORE, ec=ACC, lw=1.8)
txt(90, 51.5, "Index store", size=11.5, weight="bold", color=ACC, ha="center")
txt(90, 47.8, "vectors", size=9.4, color=INK, ha="center")
txt(90, 44.6, "+ Page Index", size=9.4, color=INK, ha="center")
txt(90, 41.4, "(page · figure · table)", size=8.6, color=MUTED, ha="center")
txt(90, 37.2, "+ figure images", size=9.4, color=INK, ha="center")

arrow(18, 49.5, 24, 49.5, color=ACC)
arrow(43, 49.5, 49, 49.5, color=ACC)
arrow(70, 49.5, 78, 47.5, color=ACC)

# ───────── 질의타임 레인 (아래) ─────────
txt(2, 26.0, "ASK  (online)", size=9.5, color=MUTED, weight="bold")
box(3, 9, 23, 14, CARD, ec=INK, lw=1.6)
txt(4.6, 19.8, "QUESTION", size=8.5, color=ACC, weight="bold")
txt(4.6, 16.2, "“SCTLR_EL1 bitfields —", size=10.4, weight="bold")
txt(4.6, 13.2, "  which page & figure?”", size=10.4, weight="bold")
txt(4.6, 10.4, "ARM v8-A manual", size=8.4, color=MUTED, style="italic")

box(31, 7, 39, 18, "#eef7f1", ec=GOOD, lw=1.8)
txt(50.5, 22.6, "③ Search · agentic", size=11.5, weight="bold", color=GOOD, ha="center")
for i, (t, c) in enumerate([
        ("semantic_search · keyword_search · read_chunk", INK),
        ("page_index_search → locate page/figure/table", ACC),
        ("image_read (VLM) → read the bitfield figure", WARM)]):
    txt(33, 19.0 - i * 3.0, "•  " + t, size=9.2, color=c)
txt(50.5, 9.4, "the agent picks its own tools", size=8.6, color=MUTED, ha="center", style="italic")
swap(50.5, 4.6)

# 검색이 store를 읽음 (위로 점선 화살표)
arrow(70, 18, 80, 30, color=ACC, lw=1.8, ls=(0, (4, 3)), rad=-0.15)
txt(75.5, 26.5, "reads", size=8.2, color=ACC, ha="center", style="italic")

arrow(26, 16, 31, 16, color=GOOD)

# 정답 카드 + 미니 비트필드
box(78, 6, 43, 20, CARD, ec=GOOD, lw=1.9)
txt(80, 23.0, "ANSWER", size=8.5, color=GOOD, weight="bold")
txt(80, 19.4, "Page 13, Figure 2.", size=12.5, weight="bold")
txt(80, 16.0, "SCTLR_EL1 control bits:", size=9.6, color=INK)
bx, by, bw = 80, 9.4, 39
fields = [("M", GOOD), ("A", ACC), ("C", WARM), ("SA", MUTED), ("EE", GOOD), ("WXN", ACC)]
cw = bw / len(fields)
for i, (lab, c) in enumerate(fields):
    ax.add_patch(FancyBboxPatch((bx + i * cw, by), cw - 0.5, 3.3,
                 boxstyle="round,pad=0.1,rounding_size=0.8", fc=c, ec="white", lw=1.2, zorder=3))
    txt(bx + i * cw + (cw - 0.5) / 2, by + 1.65, lab, size=8.2, color="white", weight="bold", ha="center")
txt(80, 7.0, "✓ located in the doc  +  ✓ diagram read by VLM", size=9.2, color=GOOD)

arrow(70, 13.5, 78, 14.5, color=GOOD)
txt(121, 28.2, "Reader: GPT-4.1 mini (multimodal)", size=8.2, color=MUTED, ha="right", style="italic")

# 하단 가치 한 줄
txt(62, 1.4, "parse → build → search are fully decoupled — swap the parser, embedder, or reader without touching the rest.",
    size=9.4, color=MUTED, ha="center", style="italic")

plt.tight_layout(pad=0.3)
out = __file__.rsplit("/", 1)[0] + "/fig1_teaser.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
