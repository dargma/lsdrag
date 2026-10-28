# lsdrag v2.0

A-RAG 에이전트(Qwen3-4B-Instruct-2507, MuSiQue 2hop)에서 도구 선택 head를 모방(BC)과 GRPO로 학습하고, LLM이 직접 도구를 고르는 baseline과 비교한 실험의 정리 브랜치입니다.

- 결론 보고: [`docs/JEV_HEAD_POLICY_CONCLUSION_20261028.md`](docs/JEV_HEAD_POLICY_CONCLUSION_20261028.md)
- 재현 가이드: [`docs/REPRODUCE_HEAD_POLICY.md`](docs/REPRODUCE_HEAD_POLICY.md)

| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 |
|---|---:|---:|---:|---:|
| baseline (native) | 24.70 | 97.4 | 4,691 | 2.38 |
| 모방 (BC head) | 22.50 | 101.6 | 4,749 | 2.39 |
| GRPO 적용 | 22.50 | 98.4 | 4,625 | 2.33 |

이 브랜치에는 위 결과를 재현하는 코드와 기록만 있습니다. 다른 실험(멀티모달, 9B, 기타 head 변형, LoRA 계획 등)과 전체 이력은 `jev_v1.1` 브랜치에 있습니다.

```bash
pip install -e .
PYTHONPATH=src:scripts python -m pytest -q tests
python scripts/repro_head_policy.py table
```
