"""HippoRAG 2를 lsdrag와 '동등 조건'으로 평가 — 같은 페이지 passage·같은 평가셋·같은 Reader(gpt-4.1-mini)·같은 지표.
vllm/gritlm는 스텁(OpenAI 경로만). 결과: benchmarks/results/HIPPORAG_<set>.md
"""
import json, os, string, sys, time
from hipporag import HippoRAG

_ART={"a","an","the"}
def _norm(s):
    s=(s or "").lower(); s="".join(c if c not in string.punctuation else " " for c in s)
    return [t for t in s.split() if t not in _ART]
def f1_em(pred,golds):
    p=_norm(pred); bf,be=0.0,0
    for g in golds:
        gt=_norm(g); em=int(p==gt or " ".join(gt) in " ".join(p))
        n=sum(min(p.count(t),gt.count(t)) for t in set(p)&set(gt))
        f1=0.0 if (n==0 or not p or not gt) else (lambda pr,rc:2*pr*rc/(pr+rc))(n/len(p),n/len(gt))
        bf=max(bf,f1); be=max(be,em)
    return bf,be

def recall(item, txt):
    txt=txt.lower()
    if item.get("gold_sources"):
        srcs=item["gold_sources"]; return sum(1 for s in srcs if s.lower() in txt)/len(srcs)
    if item.get("gold_page") and item["gold_page"].lower() in txt: return 1.0
    kw=item.get("gold_keywords") or []
    return 1.0 if (kw and any(k.lower() in txt for k in kw)) else (0.0 if (item.get("gold_page") or kw) else 1.0)

def main():
    passages=json.load(open(os.environ.get("HIPPO_PASSAGES","/tmp/hippo_passages.json"),encoding="utf-8"))
    hr=HippoRAG(save_dir=os.environ.get("HIPPO_SAVE_DIR","outputs/hipporag_idx"),
                llm_model_name="gpt-4.1-mini", embedding_model_name="text-embedding-3-small")
    t0=time.time(); hr.index(docs=passages); print(f"INDEXED {len(passages)} passages in {time.time()-t0:.0f}s", flush=True)
    concise=" Answer in as few words as possible (just the key term/label)."
    for setname in ["eval_set_100","eval_longctx"]:
        data=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{setname}.json"),encoding="utf-8"))["items"]
        n=len(data); sr=sf=se=0.0; rows=[]
        for it in data:
            sol,answers,_=hr.rag_qa(queries=[it["question"]+concise])
            qs=sol[0]; ans=qs.answer or ""; ret=" ".join(qs.docs or [])
            rc=recall(it,ret); f1,em=f1_em(ans,it["gold_answers"])
            sr+=rc; sf+=f1; se+=em; rows.append((it["id"],rc,f1,em,ans[:70]))
            print(f"{it['id']}: recall={rc:.2f} F1={f1:.2f} EM={em}", flush=True)
        print(f"SUMMARY_HIPPO set={setname} n={n} recall={sr/n:.4f} f1={sf/n:.4f} em={se/n:.4f}", flush=True)
        out=f"benchmarks/results/HIPPORAG_{setname}.md"
        L=[f"# HippoRAG 2 — {setname} ({n}문항, reader=gpt-4.1-mini, 1798 page passages)\n",
           f"| 지표 | 값 |\n|--|--|\n| recall | {sr/n:.2%} |\n| F1 | {sf/n:.3f} |\n| EM | {se/n:.2%} |\n",
           "\n| ID | recall | F1 | EM | 답변 |\n|--|:-:|:-:|:-:|--|"]
        for cid,rc,f1,em,a in rows: L.append(f"| {cid} | {rc:.2f} | {f1:.2f} | {em} | {a.replace(chr(124),'/')} |")
        open(out,"w",encoding="utf-8").write("\n".join(L))
    print("DONE_HIPPO", flush=True)

if __name__=="__main__":
    main()
