"""Emit one line per meaningful event of the full-FT runs (iteration done, eval summary, failure, completion).
State is kept in watch_state.json so a re-armed watcher does not repeat old events."""
import json, os, sys, time
from pathlib import Path
R = Path('/content/local_fast/lsdrag-text-v1/results'); S = Path('/content/local_fast/lsdrag-text-v1/watch_state.json')
RUNS = ['mh_native_A1', 'mh_head_B1', 'mh_headonly_C1', 'mh_headonly_C2_lr2e-4']
seen = set(json.loads(S.read_text())) if S.exists() else set()
def emit(key, msg):
    if key not in seen:
        seen.add(key); S.write_text(json.dumps(sorted(seen))); print(msg, flush=True)
end = time.time()+float(sys.argv[1]) if len(sys.argv) > 1 else float('inf')
while time.time() < end:
    for r in RUNS:
        d = R/r
        try:
            st = json.loads((d/'status.json').read_text())
        except Exception:
            continue
        if st['phase'] in ('failed', 'complete'):
            emit(f'{r}:{st["phase"]}', f'{r} {st["phase"].upper()} {st.get("error", "")[:300]}')
        elif st.get('pid') and not os.path.exists(f'/proc/{st["pid"]}'):
            emit(f'{r}:dead:{st["pid"]}', f'{r} PROCESS GONE in phase {st["phase"]} (pid {st["pid"]})')
        if (d/'curves.json').exists():
            for c in json.loads((d/'curves.json').read_text()):
                emit(f'{r}:it{c["iteration"]}', f'{r} it{c["iteration"]} train F1 {100*c["mean_f1"]:.1f} calls {c["mean_calls"]:.2f} 3+ {c.get("three_plus_calls", 0):.2f} '
                     f'signal {c["groups_with_signal"]:.2f} kl {c.get("kl")} gap {c.get("gap")} head_changed {c.get("head_changed_vs_loaded")} updates {c["updates"]}')
        for i in range(5):
            f = d/f'eval/i{i}/summary.json'
            if f.exists():
                s = json.loads(f.read_text())
                emit(f'{r}:eval{i}', f'{r} dev64 i{i}: F1 {100*s["f1"]:.2f} EM {100*s["em"]:.2f} calls {s["tool_calls"]:.2f} 3+ {s["three_plus_calls"]:.2f} '
                     f'gen {s["generated_tokens"]:.1f} inp {s["prompt_tokens"]:.0f} empty {s["empty_answer"]:.2f} err {s["errors"]}')
    time.sleep(30)
