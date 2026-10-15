# Direct A-RAG 통합 검토 — 사용자 의도·입력 계약·실행 한계

작성: 2026-10-15 UTC. 검토 대상은 `tool_registry.py`, `direct_reader.py`, `direct_contract.py`, `agent.py`, `context.py`, `retrieval.py`, 실행·계약 export CLI 및 direct integration tests다. 코드 읽기와 CPU 테스트를 직접 수행했고, 이후 실행 담당이 수행한 실제 GPU 기능 smoke의 원시 artifact를 읽어 확인했다. GPU 실행 자체는 이 검토자가 수행하지 않았다.

## 1. 판정

**사용자가 요청한 A-RAG 도구·멀티모달 입력·직접 dispatch의 구조적 통합과 실제 backbone 기능 연결을 승인한다. 현재 학습된 6-action controller의 판단 품질이나 속도는 승인/입증하지 않는다.** 검토 중 발견한 FINAL 형식 충돌, 종료 의미, 선택 figure의 예산 오류, stale checkpoint 검증 부족, 실패 비용 누락 문제는 아래와 같이 수정되었다. 현재 읽은 통합 코드에서 이를 막는 추가 구조적 blocker는 발견하지 못했다.

원래 keyword/semantic/read의 반복 검색–읽기 흐름과 G1의 text/image 검색 융합을 유지하면서, 명시적으로 선택한 `multimodal_full` profile에 `image_search`와 `figure_read`를 더했다. 정확히는 **도구 5개 + FINAL 1개 = 행동 6개**다. 모든 모드에 기존 Q3 pilot을 억지로 적용하는 변경이 아니다.

기존 Q3 pilot의 라벨·checkpoint·성능 수치는 이 새 구조를 입증하는 근거로 재사용하지 않는다. `jev-direct-head-v1`의 새로운 여섯 행동 checkpoint가 필요하다. 이 검토 시점에 그 controller가 학습되었다는 근거는 없다. 구조·기능 준비와 학습된 정책의 완성은 별개다.

## 2. 사용자 의도와 실제 경로

| 요구 | 현재 코드의 동작 | 판정 |
|---|---|---|
| vanilla A-RAG의 progressive retrieval | `keyword_search`, `semantic_search`는 ID/snippet을 주고 `read_chunk`는 원문·인접 ID를 전달; 여러 iteration 후 답변 | 유지 |
| jev_v1.0 G1의 실제 이미지 검색 신호 | G1 semantic ranking은 Google text와 실제 image index ranking을 RRF로 결합 | 유지. 두 index가 실제 올바른 자산인지 run manifest/integrity 검증은 여전히 필요 |
| search-linked 이미지 관측 | keyword/semantic 반환 항목의 image-linked chunk를 읽고 pending image bytes를 다음 Reader 입력에 추가 | 유지 |
| 명시적인 시각 검색·읽기 | `image_search`는 figure ID/snippet, `figure_read`는 선택 figure와 연결 text를 제공하고 실제 pixels를 다음 입력에 추가 | full profile에서 구현 |
| head가 직접 도구 선택 | full multimodal final-norm hidden → 6-action linear head → 선택 enum → registry의 bound Python handler | 구현. tool 이름을 모델이 생성해 결정하지 않음 |
| 도구 인자 | 이미 선택된 도구의 schema를 주고 별도 prefill에서 argument JSON만 생성; schema 검증 후 호출 | 구현. query/ID 등 인자 생성 자체는 남음 |
| FINAL | 답변 JSON을 생성한 뒤 종료; 직접 선택 FINAL에는 V2의 추가 verification loop를 적용하지 않음 | 수정 완료 |

`legacy` profile에서는 기존 prompt를 그대로 쓰도록 분리했다. full profile은 같은 registry에서 schema·행동 순서·도구 안내·실행 handler를 만든다. `run_experiment.py --reader-mode direct-head`는 G1, `multimodal_full`, model path 및 새 head checkpoint를 요구한다. 다른 arm 또는 네 행동 checkpoint로 조용히 실행하지 않는다.

도구 이름·native 호출 prefix를 **출력으로 decode하지 않는 것**과 그 이름이 이력에서 완전히 사라지는 것은 다르다. runtime은 선택과 인자를 canonical `assistant.tool_calls`로 만들어 tool result와 연결한다. 다음 prefill에서는 tokenizer가 이 사건을 정상 tool history로 직렬화한다. 해당 history의 token 비용은 남는다.

## 3. 이미지 행동과 budget 수정

초기 `figure_read(['f2'])` 구현은 f2의 chunk를 통째로 읽으면서 같은 chunk의 f1까지 먼저 queue할 수 있었다. 남은 image budget이 1이면 사용자가 선택한 f2가 전달되지 않는 오류가 가능했다.

현재는 `figure_read`가 요청한 figure 집합만 `_figure_filter`로 큐잉하고 연결 text는 제공한다. 일반 `read_chunk`와 search-linked auto-read는 원래 chunk-linked image 전달 의미를 유지한다. 요청 figure는 실제 bytes hash를 확인한 뒤 user image block으로 전달된다. 이미지 budget 소진·중복 전달·이미 읽은 text 상태도 별도로 추적한다.

CPU integration test는 같은 chunk에 `unselected`, `selected` 두 figure와 budget=1을 두고 `selected`만 queued되는지 확인한다. 또 다섯 도구를 순서대로 호출하면서 semantic 검색에서 전달한 실제 inline image bytes가 다음 decision 및 FINAL 이력에 있는지 확인한다. 이것은 모델이 픽셀을 이해했다는 증명과 구분한다.

## 4. 인자·답변·이력의 정합성

`DirectHFReader._select`는 multimodal backbone의 마지막 정규화 hidden을 읽는다. 이 단계는 full vocabulary projection이나 action token decode를 하지 않는다. 선택된 handler 이름은 runtime이 넣고, `_generate_json`은 해당 도구의 인자 object만 생성한다. JSON Schema의 필수 키·타입·범위·추가 필드 제한을 통과해야 실제 handler가 호출된다. 잘못된 도구 wrapper나 잘린 인자는 임의로 고쳐서 실행하지 않는다.

초기의 FINAL 지시는 reasoning을 금지했지만 기존 A-RAG system prompt는 reasoning을 요구했다. 현재 FINAL schema는 `answers`를 필수로, `citations` 및 `reasoning`을 선택적으로 허용하며 system instruction과 모순되지 않는다. 유효 JSON이더라도 EOS 없이 길이 제한에 닿으면 `finish_reason=length`가 전달되고 FINAL은 truncated로 기록된다. 인자 출력이 length로 끝나면 dispatch하지 않는다.

Gemma의 pinned chat template는 tool-call arguments를 mapping으로 받는다. 기존 `processor_messages`가 JSON 문자열을 dict로 바꾸고, synthetic call ID가 tool response와 연결되므로 다음 turn의 format은 정합하다. 별도 인자 생성에서 tools schema 목록을 비활성화해도 이전 canonical tool 사건은 history에 남는다. 실제 모델의 JSON 순응률은 GPU 기능 smoke 및 이후 자연 상태 평가가 필요하다.

## 5. checkpoint와 prompt의 stale 방지

새 checkpoint는 format `jev-direct-head-v1`, 완전한 action 순서, tool-contract hash, feature contract, 올바른 weight/bias shape 및 유한 값을 요구한다. 모델 hidden width도 확인한다. 가능한 계약 오류는 GPU 모델을 올리기 전에 거부한다.

feature contract는 다음을 묶는다.

- pinned 40-hex model snapshot revision과 model/tokenizer/processor/chat-template 자산 hash.
- HF 입력 준비 코드 hash, 마지막 normalized hidden이라는 readout 규칙, BF16.
- image soft tokens·context limit, action 순서와 tool schema 계약.
- 허용된 effective system prompt SHA256 목록. 실제 각 호출의 system prompt를 그 목록과 대조.

따라서 Q3 또는 기존 네 행동 head를 이름만 바꿔 여섯 행동 controller처럼 사용하는 경로는 거부된다. 이 검사는 checkpoint가 올바르게 학습됐다는 보증이 아니라 잘못된 입력/출력 계약으로 실행하지 않기 위한 보호다.

`export_direct_tool_contract.py`는 계약과 effective prompt를 내보내며, 학습된 head를 만드는 script가 아니다. Mem-Gallery처럼 공통 prompt 뒤에 benchmark/task 지시를 붙이는 경우 실제 결합 prompt도 허용 목록에 있어야 한다. 기본 prompt 계약만으로 모든 benchmark에 대해 학습·검증되었다고 할 수 없다.

별도 smoke checkpoint의 `purpose=synthetic_dispatch_smoke`는 `diagnostic=True`일 때만 허용한다. 일반 CLI에는 이 flag를 노출하지 않는다. synthetic head를 생산 실험용 학습 모델로 오인하는 경로를 차단하는 방향은 타당하다.

## 6. 비용과 실패 기록

선택 prefill과 인자/답변 prefill은 서로 다른 prompt다. 현재 구현은 안전하지 않은 KV 재사용을 하지 않고 `kv_reused=False`로 기록한다. 두 호출의 preprocessing·prefill·decode·prompt/generated token 수를 합치며 head 시간과 전체 Reader wall time도 남긴다. 따라서 tool prefix decode를 생략했더라도 **별도의 두 번째 prefill이 추가된 현재 구현을 빠르다고 가정할 수 없다.**

잘못된 JSON은 이미 소비한 두 경로의 usage를 가진 `DirectOutputError`로 기록된다. 인자 생성 자체가 상세 usage를 반환하기 전에 실패하면, 알려진 decision 비용과 전체 elapsed wall을 보존하면서 `generation_cost_unknown=True`로 표시한다. 이는 실패한 generation의 token/세부 GPU 비용까지 정확히 복원했다는 뜻이 아니다. 총 wall과 알려진 부분만 보고해야 한다.

현재 사용자는 shared GPU에서 기능 확인을 허용했다. 그런 실행에서 나온 latency는 다른 실행과의 속도 비교 근거로 사용하지 않는다. 과거 Q3의 시간 수치도 새 ARAG 통합의 결과로 옮기지 않는다.

## 7. 직접 확인한 테스트와 미검증 사항

다음 CPU 검사를 이 검토에서 다시 실행했고 **50 passed**를 확인했다.

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_direct_reader.py tests/test_tool_registry.py \
  tests/test_direct_agent_integration.py tests/test_agent.py \
  tests/test_context.py tests/test_retrieval_ir.py
```

처음 bare `python -m pytest` 호출은 설치되지 않은 src package 경로 때문에 collection 단계에서 실패했고, 저장소의 `PYTHONPATH=src`로 수정한 실행이 통과했다. 이 테스트에는 GPU inference가 없다. 통합 fixture는 선택 행동과 출력 JSON을 고정하므로 learned policy accuracy나 실모델 argument 생성 품질을 검증하지 않는다.

실행 담당의 [실제 backbone 기능 smoke 원시 기록](../lab/reports/jev_v1.1/direct_arag_functional_smoke.json)을 읽어 **PASS**를 확인했다. `keyword_search → semantic_search → read_chunk → image_search → figure_read → FINAL`의 6회 Reader 호출에서 다섯 handler가 모두 실행되었고, `f1`, `f2`의 실제 이미지가 전달됐으며 최종 JSON은 유효했다. handler error와 Reader error는 모두 빈 목록이다. 각 호출의 forced prefix token 수는 0이고 실제 생성 token 수는 순서대로 15, 20, 10, 25, 10, 110이다. 이는 인자와 답변이 실제 모델 생성 경로를 지났다는 기능 기록이며 토큰 절감의 비교 결과는 아니다.

`smoke_direct_arag.py`는 실제 Gemma backbone·실제 argument/answer generation을 사용하지만 action은 scripted synthetic head, retrieval은 고정 synthetic embedding/index다. 따라서 확인한 범위는 **다섯 handler·실제 image transport·JSON 생성·종료 경로의 기능 smoke**다. artifact도 `learned_head=false`, `retrieval_quality_evaluated=false`, `latency_comparison_valid=false`로 명시한다. retrieval 품질, 여섯 행동 head의 학습, 자연 문항 정확도, agent transfer, speedup을 증명하지 않는다. 사용자 질문과 synthetic fixture 안내에 미리 준 ID/행동 정보 역시 기능 시험의 일부이며 자연 데이터 평가 입력으로 사용하지 않는다.

실행 담당은 추가로 legacy vLLM 의존 test file을 제외한 전체 CPU 검사 **72 passed**를 보고했다. 이 검토자가 직접 재실행한 범위는 위의 50개이며, 전체 72개를 별도로 재실행한 것으로 주장하지 않는다.

## 8. 다음 연구가 입증해야 하는 것

이 변경은 도구 이름 생성 형식을 바꾸는 데서 끝나지 않고, 실제 A-RAG 상태·검색 결과·이미지 입력을 가진 controller의 실행 기반을 만든다. 그러나 사용자가 원하는 **어떤 상태에서 어느 도구를 선택해야 좋은 답으로 이어지는지**, 그리고 그 판단이 다른 agent·문서에 전이되는지는 새 상태·행동 라벨과 독립 평가가 필요한 연구 질문이다.

우선 새 6-action 계약에서 실패·복구 가능한 상태를 포함한 라벨, 같은 입력을 받는 생성형/제약 first-token/일반 outcome head 대조, 선택 후 실제 원래 agent를 끝까지 실행한 결과가 필요하다. 이미지 embedding 또는 caption만으로 full visual reading의 필요성을 대신했다고 주장하지 않는다. 정확도 향상, agent 전이, 속도, 독창성은 각각의 근거가 생기기 전까지 미검증으로 둔다.
