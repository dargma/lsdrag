"""대표 그림 — agentic 루프를 직관적으로: 유저 시작점 + 에이전트 단계 + 각 단계 tool.
실제 라이브 검증 trajectory(7 steps)를 그대로 시각화 → docs/hero.png. (영문 라벨: 폰트 안전)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK="#1b2430"; MUTED="#6b7785"; LINE="#c2cad3"; WHITE="#ffffff"
TOOLS={
 "page_index_search":("#2d6cdf","#e9f0fd","page_index_search"),
 "keyword_search":   ("#2f9e6b","#e9f5ef","keyword_search"),
 "read_chunk":       ("#6b7785","#eef0f3","read_chunk"),
 "image_read":       ("#e0a23b","#fbf1df","image_read (VLM)"),
 "answer":           ("#1b2430","#ffffff","final answer"),
}
TRAJ=["page_index_search","keyword_search","read_chunk","keyword_search",
      "read_chunk","read_chunk","answer"]

plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,ax=plt.subplots(figsize=(13,6.6),dpi=170)
ax.set_xlim(0,132); ax.set_ylim(0,68); ax.axis("off")

def box(x,y,w,h,fc=WHITE,ec=LINE,lw=1.5,r=2.2,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.2,rounding_size={r}",fc=fc,ec=ec,lw=lw,zorder=z))
def txt(x,y,s,size=11,color=INK,weight="normal",ha="left",va="center",style="normal"):
    ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,style=style,zorder=6)
def arrow(x1,y1,x2,y2,color=MUTED,lw=2.2,ls="-",rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,color=color,lw=lw,ls=ls,zorder=4,connectionstyle=f"arc3,rad={rad}"))

txt(66,64.5,"One question  →  the agent picks tools and runs a few steps  →  a grounded answer",
    size=14.5,weight="bold",ha="center")

# USER
box(2,44,20,13,"#f4f8ff",ec="#2d6cdf",lw=1.8)
txt(12,53.6,"1 · USER",size=11.5,weight="bold",color="#2d6cdf",ha="center")
txt(12,50.2,"asks a question",size=10,ha="center")
txt(12,47.0,"intervenes: add / remove docs",size=8.0,color=MUTED,ha="center")

# AGENT
box(34,43,26,15,"#eef7f1",ec="#2f9e6b",lw=2.0)
txt(47,54.3,"2 · AGENT (LLM)",size=12,weight="bold",color="#2f9e6b",ha="center")
txt(47,50.8,"selects tools on its own",size=10,ha="center")
txt(47,47.6,"loops as needed",size=9.2,color=MUTED,ha="center")
ax.add_patch(FancyArrowPatch((40,58),(54,58),arrowstyle="-|>",mutation_scale=14,color="#2f9e6b",lw=1.8,connectionstyle="arc3,rad=-0.9",zorder=4))
txt(47,61.6,"repeat  (≤ max_loops)",size=8.6,color="#2f9e6b",ha="center",style="italic")

# Index store
box(34,29,26,9,"#eef3fb",ec="#2d6cdf",lw=1.5)
txt(47,35.4,"Index + Page Index",size=9.6,weight="bold",color="#2d6cdf",ha="center")
txt(47,31.8,"vectors · page/figure/table · images",size=8.2,color=MUTED,ha="center")
arrow(47,38,47,43,color="#2d6cdf",lw=1.6,ls=(0,(4,3))); txt(49.5,40.5,"reads",size=8,color="#2d6cdf",style="italic")

# ANSWER
box(98,43,32,15,WHITE,ec=INK,lw=1.8)
txt(100,54.3,"3 · GROUNDED ANSWER",size=11,weight="bold",ha="left")
txt(100,50.6,"pinpoints page & figure,",size=9.6,ha="left")
txt(100,47.6,"and reads the diagram",size=9.6,ha="left")

arrow(22,50.5,34,50.5,color="#2d6cdf")
arrow(60,50.5,98,50.5,color="#2f9e6b")
txt(79,52.6,"after several tool calls",size=8.4,color=MUTED,ha="center",style="italic")

# 실제 trajectory
txt(4,23.5,"Actual trajectory of this query  —  each step = one tool call  (live-verified)",
    size=10.5,weight="bold",color=INK)
n=len(TRAJ); x0=4; gap=124/n
for i,tk in enumerate(TRAJ):
    c,bg,label=TOOLS[tk]; x=x0+i*gap
    box(x,8,gap-3.5,11,bg,ec=c,lw=1.7)
    txt(x+(gap-3.5)/2,16.2,f"Step {i+1}",size=9,weight="bold",color=c,ha="center")
    txt(x+(gap-3.5)/2,11.6,label,size=8.0 if len(label)<14 else 7.2,color=INK,ha="center")
    if i<n-1: arrow(x+gap-3.3,13.5,x+gap-0.2,13.5,color=MUTED,lw=1.6)

# 범례
txt(4,3.0,"tools:",size=8.6,color=MUTED)
for j,k in enumerate(["page_index_search","keyword_search","read_chunk","image_read"]):
    c,bg,label=TOOLS[k]
    ax.add_patch(FancyBboxPatch((13+j*29,1.6),2.2,2.2,boxstyle="round,pad=0.1,rounding_size=0.6",fc=c,ec="white",zorder=3))
    txt(16+j*29,2.7,label,size=8,color=INK)

plt.tight_layout(pad=0.3)
out=__file__.rsplit("/",1)[0]+"/hero.png"
fig.savefig(out,bbox_inches="tight",facecolor="white")
print("wrote",out)
