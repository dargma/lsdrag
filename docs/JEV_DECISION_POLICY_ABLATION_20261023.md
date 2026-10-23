# PPO 상태별 판단·전역 편향·종료 분해

2026-10-23 사용자 실행 승인. 새family-disjoint40질문×9조건=360episodes. 기존Qwen4Bnonthinking/도구/인자생성 동일, PPO두seedcheckpoint고정. 비교:anchor, 각seed full/global_bias/stop_only/tool_only. 새질문은기존모든노출qid/sharedsubquestionfamily제외,실행전hash순서동결.

Global bias: 기존anchor logit에4개상수추가. 원train67상태의학습된PPO확률을질문균등KL로근사,Adam.05×200step,L2.001mean(bias²),meanbias0. 기존test결과로fit하지않음. 고정편향이PPO를얼마나설명하는지보는projection이며같은예산으로reward학습한새정책대조가아님. 근사오차도보고.

STOP-only: 같은state에서full의FINAL−bestnonFINAL logitmargin을그대로유지하되nonFINALargmax는anchor로고정. Tool-only는anchor의종료margin을유지하고full의nonFINAL선택을사용. Greedy실행시현재state의gate결정/도구선택이정확보존됨. 경로가달라지면후속state도달라지므로원trajectory종료시점을고정한다는뜻은아님. unit2검사+train67 parity확인후평가.

주비교 full−globalbias(두고정head평균) F1/utility;보조full−STOP/tool/anchor. 같은40질문pairedbootstrap4000(seed20261036),80독립질문아님. CI가0포함하면차이미확정이며동등성입증아님. full이globalbias보다품질/utility모두하한>0이면state별수정의추가기여후보,아니면평균과근사오차/한계대로보고. 다중aux최적선택금지. 기존결과를본뒤세운기전가설을새질문에검증함.

load제외1800초,생성500000/input16000000상한,공유GPU시간은통제latency주장불가. 원본 /content/local_fast/lsdrag-text-v1/results/decision_policy_ablation_20261023. 명령 PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_policy_ablation.py. Source/inputhash manifest lab/registry/decision_policy_ablation_20261023/manifest.json.

## 사용자 질의에 따른 해석 명확화 (평가 도중, 조건 변경 없음)

이실험은도구선호만으로설명가능한지검사하며실제이해를직접증명하지않는다. Globalbias는기존anchor의상태의존성을유지하고PPO추가변화만상수로근사한다. 전체head우월시에도학습된상태별수정이도움된다는증거에한정한다. 후속이해가설은동일질문에서충분근거/핵심근거제거/무관근거추가를통제하고길이·위치·출처표식의지름길을분리해야한다. 이문단은해석만명확화하며동결평가조건/지표/예산은변경하지않음.
