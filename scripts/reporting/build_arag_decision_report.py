"""Build the Korean HTML report from audited, small committed evidence. CPU only."""
from pathlib import Path
import json, html
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/reports/arag_decision_head_20261019'
D=json.loads((OUT/'evidence.json').read_text())
E=lambda x:html.escape(str(x),quote=True)
BASE='https://github.com/dargma/lsdrag/blob/jev_v1.1/'
def link(path,label):return f'<a href="{BASE}{path}">{E(label)}</a>'
def table(head,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def raw(x):return '<pre>'+E(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
def section(i,title,body):return f'<section id="{i}"><span class="section-no">{i.upper()}</span><h2>{title}</h2>{body}</section>'
parts=[]
parts.append(section('background','1. 배경', '''
<p>출발점은 Jev 같은 decision 모델의 아이디어였다. <strong>“도구를 고르는 데 필요한 텍스트 생성을 작은 분류 head로 바꾸면, A-RAG의 정확도를 유지하면서 빨라질까?”</strong> 이를 검색·읽기·종료를 반복하는 텍스트 QA에서 확인했다.</p>
<p>비교군 Native는 같은 Qwen reader가 도구 이름·인자·답변을 직접 생성하는 이 저장소의 A-RAG 방식 구현이다. Jev 모델을 사용한 실험이나 공식 A-RAG 전체 벤치마크 재현은 아니다. <a href="https://arxiv.org/abs/2602.03442">A-RAG 원 논문</a> · <a href="https://typesafe.ai/">TypeSafe</a></p>
'''))
parts.append(section('method','2. 실험 방법',table(['항목','설정'],[
['Reader','Qwen3-4B-Instruct-2507 · frozen · BF16 · greedy · non-thinking'],
['데이터','MuSiQue-Ans v1.0, answerable 2-hop. 질문별 제공 문단 안에서 검색.'],
['학습 / 검증','teacher 32문항 → train24(82상태) / validation8(29상태).'],
['주 평가','미사용 dev100 × Native/Head = 200 episode. 모델·head 고정, 문항별 실행 순서 교대.'],
['검색 / GPU','embeddinggemma-2(768차원) · RTX PRO 6000 Blackwell 약 96GB, 공유 환경.'],
['실행 한도','최대 8 tool calls + FINAL · context 16,384 · 출력 512 tokens.']])+'''
<h3>학습한 것은 무엇인가?</h3>
<p>teacher는 같은 reader의 Native 경로다. 각 결정 직전의 마지막 hidden(2560차원)에 <strong>4-way linear action head</strong>를 붙여 teacher의 행동을 모방했다. 종료는 별도 head가 아니라 <code>FINAL</code> class다. backbone은 고정하고 작은 head만 학습했다.</p>
<p><strong>정답 trajectory만 골라 학습한 것은 아니다.</strong> train24 중 teacher가 EM 정답을 낸 것은 5문항이다. 따라서 현재 학습 목표는 최적 행동이나 최종 정답률이 아니라 teacher 행동 일치다. validation8도 개발에 노출된 자료다. 주 평가100은 기존 질문·하위질문 family와 겹치지 않게 사전 선택했으며 학습·튜닝에 사용하지 않았다.</p>
<h3 id="dispatch-flow">실행 흐름: 번호 선택 → 인자 생성 → 함수 호출</h3>
<div class="flow" aria-label="번호 선택에서 함수 실행까지"><div>질문 + 관측 → reader prefill<br><small>hidden + KV cache</small></div><b>↓</b><div>action head → 합법 행동 중 argmax<br><small>0 keyword / 1 semantic / 2 read / 3 FINAL</small></div><b>↓</b><div>예: 번호 1 → semantic_search로 매핑</div><b>↓</b><div>고정 호출 구문을 batch 처리<br>같은 KV에서 LLM이 query 인자 생성</div><b>↓</b><div>인자 검증 → Python handler 실행<br>결과를 대화에 추가 → 다음 turn</div></div>
<p><strong>번호도 도구 이름도 텍스트로 생성하지 않는다.</strong> head가 번호를 고르면 runtime이 도구 이름과 고정 구문을 공급한다. 이 구문은 한 번에 처리하고, 인자만 같은 KV cache에서 이어 생성한다. 줄어드는 것은 도구 선택·고정 구문의 <strong>순차 decoding</strong>이다. 인자와 최종 답변 생성은 남는다.</p>
<div class="table-wrap"><table id="generation-comparison"><thead><tr><th>구분</th><th>도구 선택·호출 구문</th><th>LLM이 생성하는 부분</th></tr></thead><tbody><tr><td>Native</td><td>LLM이 함께 생성</td><td><code>&lt;tool_call&gt;{"name":"semantic_search","arguments":{"query":"Andrew Lang death year"}}&lt;/tool_call&gt;</code></td></tr><tr><td>Decision head</td><td>head가 <strong>1 선택</strong> → 코드가 semantic_search 구문 공급</td><td><code>query": "Andrew Lang death year"}</code></td></tr></tbody></table></div>
<p class="small">Head는 <code>{"</code>까지 공급된 상태에서 이어 생성한다. 공백·줄바꿈을 생략한 설명용 예시이며 실제 raw는 아래에 있다.</p>
<p>KV 재사용은 같은 turn의 decision→인자/답변 사이에 적용된다. 다음 turn은 새 관측을 포함해 다시 prefill한다. <code>FINAL</code>은 답변 생성으로 분기한다.</p>
<details><summary>학습·데이터 세부 설정</summary>
<p>추가 read-slot head(2560→5)는 GENERATE 또는 관측 ID 후보 4개 중 하나를 고른다. action/read head는 독립 linear이며 공유 trainable trunk가 없다. 학습 파라미터는 총 23,049개다. action CE + read 상태의 slot CE, AdamW(lr 0.001, weight decay 0.01), 100 epochs, seed 20261016을 사용했다. 정규화 통계는 train에서만 계산했다.</p>
<p>학습 상태: keyword24 / semantic18 / read16 / FINAL24. read 라벨은 teacher 인자와 후보의 정확 일치로 만들었다. GENERATE5 / slot1 2 / slot2 6 / slot3 3 / slot4 0개로, 마지막 후보의 학습 양성이 없었다. 고정 validation 상태의 action macro-F1은 0.814였지만 자율 QA 성능과는 다른 지표다.</p>
<p>최종100은 answerable 2-hop dev 1,252개 중 기존 노출 제외 후 1,099개에서 hash 순서와 family 중복 배제로 선정했다. dev 전체의 무작위 표본이나 사전학습 비노출 표본은 아니다. Runtime에는 문단 ID·제목·본문만 제공하고 gold 답·support·질문 분해는 제외했다.</p>
</details>
<h3 id="tools">A-RAG tool call 현황</h3>
'''+table(['번호 / 행동','인자','기능','Native / Head 호출'],[
['0 · keyword_search','keywords, top_k','문자열 빈도×검색어 길이 점수로 검색','84 / 101'],
['1 · semantic_search','query, top_k','query·문단 embedding 내적으로 검색','68 / 66'],
['2 · read_chunk','chunk_ids','관측 ID의 전체 문단 읽기','74 / 72'],
['3 · FINAL','answer, support_ids','답변 후 종료; 외부 도구 호출 아님','유효 답변 96 / 100']])+'''
<p>검색은 ID·제목·앞 300자 snippet을 반환한다. top_k는 기본 5(1–20), read는 1–8개 ID를 받는다. snippet만으로 답할 수도 있다. <strong>도구 호출 합계는 226→239회</strong>로 늘었고, head는 100문항 모두 keyword 검색으로 시작했다. 빈 keyword 결과는 Native 23/84회, Head 29/101회였다. 따라서 이번 속도 차이를 호출 수 감소로 설명할 수 없다.</p>
'''))
n=D['groups']['native'];h=D['groups']['head_generate']
metrics=[('답변 F1 (%)','f1',100,2),('EM 정답률 (%)','em',100,1),('근거 ID F1 (%)','support_f1',100,2),('평균 시간 (초)','seconds',1,3),('평균 순차 생성 tokens','completion_tokens',1,2),('평균 tool calls','tool_calls',1,2),('형식 오류 문항','error_episodes',1,0)]
parts.append(section('results','3. 결과',table(['지표 · 새100문항','Native','Head + 인자 생성'],[[label,f'{n[k]*scale:.{digits}f}',f'{h[k]*scale:.{digits}f}'] for label,k,scale,digits in metrics])+'''
<p><strong>평균 시간은 26.8% 감소했고, 답변 점수는 비슷했다.</strong> F1 기준 5승·90동률·5패, EM 기준 Head만 정답 4문항·Native만 정답 3문항이었다. F1 차이 +0.46%p의 bootstrap 95% 구간은 −4.94~+5.96%p로, 품질 향상이나 비열등성을 입증하지는 못했다.</p>
<h3>왜 빨라졌는가?</h3>
<p>도구 호출 prefix는 호출당 13 tokens, 평균 2.39회 호출로 문항당 31.07 tokens였다. 이를 순차 생성 대신 batch로 처리했다. decoding 시간은 1.144→0.744초로 줄었고 prefix batch에 0.042초가 추가됐다. 전체 차이 0.358초와 정합적이다. 반면 입력 tokens는 3,914→4,106으로 늘었다. <strong>전체 token이나 도구 수보다 순차 생성량 감소가 관측 시간 차이와 맞는다.</strong></p>
<div class="callout amber"><strong>해석의 범위.</strong> 공유 GPU에서 질문·조건당 한 번 측정했으며 head와 함께 구문 처리·cache 연결·schema 제약도 바뀌었다. 26.8%를 head 단독의 통제된 속도 효과로 볼 수 없다. Native 형식 오류 4문항을 포함한 전체100이 주 분석이다. 오류 없는 96문항만 보는 사후 분석에서는 F1 차이 −0.57%p, EM 차이 0%p였다.</div>
<h3>인자 선택까지 head에 맡기면?</h3>
<p>read 인자 선택은 <strong>생성을 줄였지만 품질이 떨어져 채택하지 않았다.</strong> 같은 checkpoint의 validation8에서 F1이 21.88→9.38%로 하락했다. 기존20문항의 후속 ON/OFF 비교에서도 read15회 중 8회의 인자 생성을 생략했지만, 생성 tokens 60.55→57.00과 함께 F1 30.43→25.43%, EM 25→20%로 떨어졌다. 최종100에서는 검색·read 인자를 모두 LLM이 생성했다.</p>
<details><summary>상세 지표와 이전 실패 기록</summary>
'''+table(['지표 / 문항','Native','Head'],[[label,f'{n[k]:.3f}',f'{h[k]:.3f}'] for label,k in [('interaction turns','interaction_turns'),('강제 prefix tokens','forced_prefix_tokens'),('입력 tokens','prompt_tokens'),('full-prefill 수','prefill_count'),('decoder forward 수(기록에서 산출)','decoder_forward_count_derived'),('중앙 시간(초)','latency_median_seconds'),('p90 시간(초)','latency_p90_seconds')]])+'''
<p>시간 포함: 제어·인자/답변 생성·검색/query embedding·계측 tokenization. 제외: 모델 로딩·문서 사전 embedding·채점·저장. 생성+prefix는 94.32→91.88 tokens다. 입력과 prefix를 중복 합산하거나 token 수를 FLOPs로 해석하지 않는다. Native의 argument_tokens=0은 별도 미계측이다.</p>
<p>초기 v5의 별도 dev10은 F1 35.00→12.86%로 품질 NoGo였다. 이후 생성 경로 정렬과 v3 학습을 거쳤다. fresh20에서는 F1 29.00→30.43%, EM 양쪽25%였다. 질문·설정이 달라 단계별 점수를 개선 곡선으로 합치지 않는다.</p>
<p>별도의 trajectory 단축 probe에서는 중간 단계 대체가 유효한 문항이 1개로 사전 기준 3개에 미달했다. 압축 정책·RL은 학습하지 않았다. 모든 실패 결과는 기존 보고서에 보존했다.</p>
</details>
<h3>정성 사례: 같은 관측에서 달라진 다음 행동</h3>
'''+table(['질문','Native','Head'],[
['Prince Prigio 작가의 사망연도','첫 검색 후 책 문단을 read → 출판연도 1889로 오답','같은 첫 검색 후 작가 사망연도 semantic 검색 → 1912 정답'],
['The Wealth of Nations 저자 이름을 딴 상의 수여 기관','첫 검색 후 semantic 검색 → University of Cambridge 정답','같은 첫 검색 후 read → The Swedish Academy 오답']])+'''
<p>다음 도구의 선택이 이후 관측과 답변을 바꿀 수 있다. 다만 두 사례의 방향이 반대이므로 “read보다 search가 낫다” 같은 단일 규칙으로 일반화할 수 없다.</p>
<h3 id="raw">대표 문항의 단계별 raw 출력</h3>
<p><code>2hop__391857_162018</code> — What year did the author of Prince Prigio die?<br>정답: <strong>1912</strong>. 아래는 Head의 기록된 인자·최종 JSON이다. 행동 이름은 실행 기록에서 읽었으며 head가 텍스트로 생성한 값이 아니다. <code>raw_json</code>에는 공급된 시작 구문도 포함된다.</p>
'''))
case=next(c for c in D['cases'] if c['qid']=='2hop__391857_162018');ep=case['policies']['head_generate'];calls=ep['reader_calls']
trace=''
for i in range(len(ep['trajectory'])+1):
 final=i==len(ep['trajectory']);action='FINAL' if final else ep['trajectory'][i]['tool_name'];dec,gen=calls[2*i:2*i+2]
 trace+=f'<article class="step"><h4>Step {i+1} · {E(action)}</h4><pre>{E(gen["usage"]["raw_json"])}</pre>'
 if not final:trace+='<details><summary>도구 응답 전체</summary>'+raw(ep['trajectory'][i]['result'])+'</details>'
 trace+='<details><summary>reader_calls 원본 · token IDs / cache / timer</summary>'+raw([dec,gen])+'</details></article>'
trace+='<p>두 번의 도구 호출 후 정답을 냈다. <a href="raw_head_generate_2hop__391857_162018.json">Head 전체 원본</a> · <a href="raw_native_2hop__391857_162018.json">Native 전체 원본</a></p>'
# Keep trace inside results rather than repeating a second raw section.
parts[-1]=parts[-1].removesuffix('</section>')+trace+'</section>'
parts.append(section('implications','4. 시사점과 다음 검증', '''
<p><strong>도구 선택의 텍스트 생성을 줄이는 구조는 동작했다.</strong> 그러나 인자·답변 생성 비용은 남고, 현재 모방 학습만으로 더 나은 정책이 만들어졌다고 보기는 어렵다. 후속 작업은 세 가지로 좁힌다.</p>
<ol><li><strong>최적화된 Native와 비교.</strong> 동일한 구문 처리·cache·schema 제약을 제공해도 head가 시간이나 품질에서 추가 이득을 주는지 확인한다.</li>
<li><strong>행동 라벨을 최종 효용에 연결.</strong> train에서 현재 head가 방문한 상태의 대안 행동을 실행하고, 남은 episode의 정답·근거·비용으로 학습한다. 단순 teacher 모방, 성공-only 모방, 행동별 가치 학습을 같은 예산으로 비교한다.</li>
<li><strong>필요할 때 구조를 확장.</strong> 먼저 빈 검색·남은 budget 같은 공개 상태 정보를 보강한다. read 후보 scoring이나 head-only RL은 이득의 여지가 확인된 뒤 검토한다. 다른 agent loop 전이는 별도 과제에서 평가한다.</li></ol>
<p>위 항목은 제안이며 실행 결과가 아니다. 현재 정책으로 만든 행동 가치는 새 정책에서도 그대로 유지되지 않으므로, 개선 head는 episode 전체를 다시 평가해야 한다. 기존 평가100은 후속 학습에 사용하지 않는다.</p>
<details><summary>관련 연구와 기여 범위</summary>
<p><a href="https://arxiv.org/abs/2604.17866">LAnR</a>는 latent retrieval·경량 제어, <a href="https://arxiv.org/abs/2607.10463">GRASP</a>는 검색·읽기 정책 학습, <a href="https://arxiv.org/abs/2510.14337">Stop-RAG</a>는 추가 검색의 가치와 종료를 다룬다. <a href="https://arxiv.org/abs/2609.34358">FORGE</a>의 frozen LLM routing, <a href="https://arxiv.org/abs/1406.5979">AggreVaTe/NRPI</a>의 cost-to-go 학습도 관련된다.</p>
<p>작은 head나 생성 생략 자체보다, 최적화된 Native보다 좋은 품질–비용 곡선을 적은 학습·수집 비용으로 만드는지가 연구 기여의 기준이다. 현재 결과는 그 가능성을 살펴본 파일럿이다.</p>
</details>
'''))
rows=''
for q in D['pairs']:
 outcome='win' if q['f1_delta']>0 else 'loss' if q['f1_delta']<0 else 'tie'
 rows+=f'<tr data-result="{outcome}"><td><code>{q["qid"]}</code><br>{E(q["question"])}</td><td>{E(q["gold_answer"])}</td><td>{E(q["native_answer"] or "∅")}<br><small>F1 {q["native_f1"]:.3f} / EM {q["native_em"]}</small></td><td>{E(q["head_answer"] or "∅")}<br><small>F1 {q["head_f1"]:.3f} / EM {q["head_em"]}</small></td><td>{q["f1_delta"]:+.3f}</td></tr>'
discordant=[[f'<code>{q["qid"]}</code>',E(q['gold_answer']),E(q['native_answer'] or '∅'),E(q['head_answer'] or '∅')] for q in D['pairs'] if q['em_delta']]
parts.append(section('appendix','부록 · 전체 결과와 근거', '<details><summary>EM 정답 여부가 다른 7문항 전체</summary>'+table(['qid','정답','Native','Head'],discordant)+'</details>'+'''
<details id="all-results"><summary>100문항 검색 · F1 승패 필터</summary><p class="small">필터는 표시만 바꾸며 주 분석의 분모는 전체100이다.</p><div class="filters"><label>검색 <input id="query" type="search" placeholder="질문·qid·답변"></label><label>F1 <select id="filter"><option value="all">전체100</option><option value="win">Head 승5</option><option value="loss">Head 패5</option><option value="tie">동률90</option></select></label><span id="count" aria-live="polite">100문항 표시</span></div><div class="table-wrap allpairs"><table id="pairs"><thead><tr><th>문항</th><th>정답</th><th>Native</th><th>Head</th><th>ΔF1</th></tr></thead><tbody>
'''+rows+'</tbody></table></div></details>'+'''
<details><summary>근거 파일 · 재현 설정 · 검증 범위</summary>
<ul><li><a href="evidence.json">집계·teacher 분포·정성 사례 JSON</a> / <a href="validation.json">검증 기록</a></li>
<li>'''+link('lab/reports/text_integrated_20261016/frozen_head_generalization100_20261017.json','최종100 원 집계')+' / '+link('lab/reports/text_integrated_20261016/frozen_head_generalization100_20261017.csv','200 episode CSV')+'''</li>
<li>'''+link('lab/reports/text_integrated_20261016/improvement_v3_final.json','v3 학습·검증')+' / '+link('lab/reports/text_integrated_20261016/improvement_v3_argument_selection.json','인자 선택')+' / '+link('lab/reports/text_integrated_20261016/v5_final_outcome.json','초기 v5 실패')+' / '+link('lab/reports/text_integrated_20261016/contraction_probe_20261017.json','단축 probe')+'''</li>
<li>'''+link('scripts/run_improvement_v3.py','학습 코드')+' / '+link('src/lsdrag/text_integrated.py','decision·dispatch')+' / '+link('src/lsdrag/text_integrated_aligned.py','cache 생성')+' / '+link('lab/registry/text_integrated_20261016/generalization100_manifest_20261017.json','실행 manifest')+'''</li></ul>
<p>실험 2026-10-17 / 보고 2026-10-19. 원본200 episode 및 두 archive의 해시를 확인하고 주요 평균을 재집계했다. teacher32의 학습 분포도 원본에서 계산했다. 새 모델 추론·학습·평가와 전체 scorer 재실행은 하지 않았다. 실험 HOLD 유지.</p>
<p>Reader revision <code>cdbee75f17c01a7cc42f958dc650907174af0554</code>. PyTorch 2.11.0+cu130 / Transformers 5.18.0 / lm-format-enforcer 0.11.3. 공개 MuSiQue dev의 고정 mirror 해시는 기존 기록과 일치했지만 공식 ZIP과의 byte 동등성은 미검증이다. 상세 데이터·source·checkpoint 해시는 manifest와 validation.json에 있다.</p>
</details>
'''))
css='*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6f9;color:#17243a;font:16px/1.85 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",Arial,sans-serif}a{color:#1858b8;text-underline-offset:3px}header{background:#11233b;color:#f8fafc;padding:70px max(24px,calc((100vw - 1120px)/2));border-bottom:6px solid #5eead4}.eyebrow{color:#5eead4;letter-spacing:.12em;font-size:12px;font-weight:700}h1{word-break:keep-all;font-size:clamp(30px,4.5vw,48px);line-height:1.3;letter-spacing:-.04em;margin:16px 0}header p{color:#cbd5e1;max-width:920px}header .pill{display:inline-block;background:#213b58;color:#e2e8f0;padding:4px 12px;border-radius:30px;font-size:12px;margin:4px 6px 4px 0}main{max-width:1120px;margin:0 auto;padding:32px 24px 80px}nav{display:flex;flex-wrap:wrap;gap:6px 20px;margin-bottom:24px;font-size:14px}section{background:white;padding:38px 42px;margin:26px 0;border:1px solid #dfe6ee;border-radius:12px;scroll-margin-top:20px;box-shadow:0 3px 12px #12243b04}h2{font-size:27px;line-height:1.4;margin:6px 0 24px;letter-spacing:-.03em}h3{font-size:20px;margin:34px 0 14px}h4{margin:8px 0;font-size:20px}.section-no{font-size:11px;letter-spacing:.16em;color:#0f766e;font-weight:800}p{margin:14px 0}.small,small{font-size:13px;color:#53647a}.table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65}th{text-align:left;background:#eaf0f6;color:#243e5b;font-weight:650}th,td{padding:12px 14px;border-bottom:1px solid #e1e8f0;vertical-align:top}tbody tr:nth-child(even){background:#f8fafc}td:first-child{font-weight:550}code{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#10243b;color:#e2ecf8;padding:20px;border-radius:8px;font:12px/1.75 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:700px;overflow:auto}details{border:1px solid #dce5ed;border-radius:8px;padding:12px 16px;margin:12px 0}summary{cursor:pointer;font-weight:600;font-size:14px}details[open] summary{margin-bottom:12px}.callout{border-left:4px solid #0f766e;background:#effaf6;padding:20px 24px;margin:24px 0}.amber{border-color:#c48311;background:#fff9e9}.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.stat-grid>div{border:1px solid #cbdedb;border-radius:9px;padding:20px 14px;background:#f1faf7}.stat-grid strong{display:block;color:#0f766e;font-size:28px;line-height:1.3}.stat-grid span{font-size:12px}.flow{display:flex;flex-direction:column;align-items:center;text-align:center;margin:26px auto;max-width:760px}.flow>div{background:#eef4fc;border:1px solid #c4d5ec;border-radius:8px;padding:12px 24px;width:100%;font-size:15px}.flow>b{color:#0f766e}.step{border-left:3px solid #2c8a82;padding:18px 20px;background:#f7fafb;margin:20px 0}.step-label{font-size:12px;color:#0f766e;font-weight:800}.step p{font-size:13px}.case-head{background:#eaf1fa;border-radius:8px;padding:20px}.timing{width:100%;height:auto;margin:14px 0}.filters{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:14px}input,select{font:inherit;border:1px solid #b9c7d8;padding:9px;border-radius:5px;max-width:100%}.allpairs{max-height:650px;overflow:auto}.allpairs th{position:sticky;top:0}.allpairs td:first-child{min-width:260px}.allpairs td{min-width:110px}footer{text-align:center;color:#64748b;font-size:12px;padding:0 20px 36px}@media(max-width:650px){main{padding:20px 12px}section{padding:24px 18px}header{padding:40px 24px}.stat-grid{grid-template-columns:1fr 1fr}th,td{padding:10px}h2{font-size:23px}.flow>div{padding:12px}.filters label{width:100%}.callout{padding:16px}pre{padding:13px}}@media print{body{background:white;font-size:11pt}header{padding:24px;color:black;background:white;border-color:#0f766e}header p{color:#334155}main{padding:0;max-width:none}section{box-shadow:none;border:0;padding:12px 0;margin:12px 0;break-inside:auto}nav,.filters{display:none}.allpairs{max-height:none;overflow:visible}details:not([open]){display:none}pre{max-height:none;color:black;background:#f3f5f7}.stat-grid strong{font-size:23px}a{color:#174d94}h2,h3,h4{break-after:avoid}tr{break-inside:avoid}}'
js="const query=document.getElementById('query'),filter=document.getElementById('filter');function applyFilter(){let n=0;document.querySelectorAll('#pairs tbody tr').forEach(row=>{const ok=(filter.value==='all'||row.dataset.result===filter.value)&&row.textContent.toLowerCase().includes(query.value.toLowerCase());row.hidden=!ok;if(ok)n++;});document.getElementById('count').textContent=n+'문항 표시';}query.addEventListener('input',applyFilter);filter.addEventListener('change',applyFilter);"
nav=''.join(f'<a href="#{i}">{label}</a>' for i,label in [('background','배경'),('method','방법'),('dispatch-flow','실행 흐름'),('tools','Tool call'),('results','결과'),('raw','Raw 사례'),('implications','시사점'),('appendix','부록')])
page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A-RAG Decision Head 실험 보고서</title><style>'+css+'</style></head><body><header><div class="eyebrow">EXPERIMENT REPORT · 2026.10.19</div><h1>A-RAG decision head 실험</h1><p>도구 선택을 분류로 바꾸고, 인자 생성은 같은 LLM에 맡기면?</p><span class="pill">Qwen3-4B · MuSiQue 2-hop</span><span class="pill">100문항 paired 평가</span></header><main><nav aria-label="목차">'+nav+'</nav><div class="callout"><strong>결과 요약.</strong> 답변 F1 31.11→31.57%, 평균 시간 1.337→0.978초(26.8% 감소)를 관측했다. 품질 향상은 미입증이며 공유 GPU 측정이다. Read 인자 선택은 품질 손실로 채택하지 않았다.</div>'+''.join(parts)+'</main><footer>lsdrag · jev_v1.1 · 기존 실험 기록 분석</footer><script>'+js+'</script></body></html>'
(OUT/'index.html').write_text(page)
print(f'Built {OUT}/index.html ({len(page.encode()):,} bytes)')
