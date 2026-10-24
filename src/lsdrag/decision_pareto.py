"""Conservative empirical preferences; two samples do not certify dominance."""
def cheaper_quality_pairs(records):
    pairs=[]
    for i,r in enumerate(records):
        by={a:[s for s in r['samples'] if s['action']==a] for a in range(4)}
        for a,ss in by.items():
            if len(ss)!=2 or any(s['errors'] for s in ss) or min(s['f1'] for s in ss)<=0:
                continue
            for b,tt in by.items():
                if a==b or len(tt)!=2 or any(s['errors'] for s in tt):continue
                if min(s['f1'] for s in ss)>=max(s['f1'] for s in tt) and max(s['generated'] for s in ss)<min(s['generated'] for s in tt):
                    pairs.append((i,a,b))
    return pairs
