# Interaction 축소 가능성 제한 진단

실행 상태 `complete`, 저장 probe 212건. 새 head 학습 없음. 같은 teacher 첫 tool 관측 이후 상태에서 실제 인자 생성·도구 실행·새 suffix를 비교했다. 기존 train24/validation8이며 새로운 heldout 성능이 아니다.

| 구간 | 조건 | n | F1 | EM | 전체 tool calls¹ | 전체 decision turns¹ | 생성 tokens | suffix latency(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | FINAL | 24 | 0.1111 | 0.0833 | 1.00 | 2.00 | 12.17 | 0.193 |
| train | alt_keyword_search | 23 | 0.2660 | 0.1739 | 2.65 | 3.65 | 70.22 | 1.173 |
| train | alt_read_chunk | 7 | 0.2857 | 0.2857 | 2.14 | 3.14 | 27.57 | 0.479 |
| train | alt_semantic_search | 8 | 0.4583 | 0.3750 | 2.12 | 3.12 | 43.38 | 0.702 |
| train | head | 24 | 0.2676 | 0.2083 | 2.42 | 3.42 | 42.33 | 0.712 |
| train | head_B1 | 24 | 0.2525 | 0.2083 | 2.00 | 3.00 | 34.67 | 0.577 |
| train | native | 24 | 0.2676 | 0.2083 | 2.42 | 3.42 | 51.33 | 0.808 |
| train | native_B1 | 24 | 0.2525 | 0.2083 | 2.00 | 3.00 | 43.67 | 0.673 |
| validation | FINAL | 8 | 0.1250 | 0.1250 | 1.00 | 2.00 | 10.88 | 0.175 |
| validation | alt_keyword_search | 8 | 0.4062 | 0.2500 | 2.38 | 3.38 | 36.12 | 0.597 |
| validation | alt_read_chunk | 3 | 0.0119 | 0.0000 | 2.00 | 3.00 | 50.00 | 0.785 |
| validation | alt_semantic_search | 3 | 1.0000 | 1.0000 | 2.67 | 3.67 | 46.00 | 0.753 |
| validation | head | 8 | 0.2188 | 0.1250 | 2.38 | 3.38 | 32.38 | 0.562 |
| validation | head_B1 | 8 | 0.1827 | 0.1250 | 2.00 | 3.00 | 31.00 | 0.513 |
| validation | native | 8 | 0.4688 | 0.3750 | 2.62 | 3.62 | 49.38 | 0.789 |
| validation | native_B1 | 8 | 0.4327 | 0.3750 | 2.00 | 3.00 | 45.50 | 0.692 |

¹공통prefix의 tool/decision1회를 포함한다. walltime과생성tokens는새suffix비용이며fulltrajectory end-to-end가아니다. B1은첫suffix tool후답변을별도로생성한고정예산baseline이다. native는최초선택만name경계decode이고후속정책은head이다. 추가syntax처리/복제비용을포함해순수learnedselector속도기여로해석하지않는다. alt조건은합법성및head의최초선택에따라선별된서로다른질문subset이다. 특히alt_semantic의높은평균을전체질문에서semantic만고르면되는증거로해석하지않는다.

## Oracle headroom

- train: baseline정답5질문, 품질보존더짧은질문3, 즉시FINAL충분1, 즉시FINAL로설명되지않는nonterminal단축1. 사후보조분석에서head_B1조기종료까지넘는nonterminal interaction추가이득0.
  정답subset전체calls절감=0.25, turn절감=0.17647058823529416; quality/interaction여지gate=True, 중간transition압축후속gate=False.
- validation: baseline정답1질문, 품질보존더짧은질문1, 즉시FINAL충분1, 즉시FINAL로설명되지않는nonterminal단축0. 사후보조분석에서head_B1조기종료까지넘는nonterminal interaction추가이득0.
  정답subset전체calls절감=0.5, turn절감=0.33333333333333337; quality/interaction여지gate=False, 중간transition압축후속gate=False.

Oracle는정답과관측근거를보고시험branch중고른특권적상한이다. baseline오답에서빨리오답을내는것은긍정적압축으로세지않았다. 개별질문별oracle와cost성분은JSON,각probe는CSV와local-fast원본에보존했다. validation8은이미노출된확인진단이며gate재튜닝/통계적비열등성주장없음. 최대2suffix tool계약과branch포괄성한계내결과다. 공유GPU관측latency이며controlledspeedup주장없음. decode forward수는각usage의생성길이와forcedprefix에서산출한값이며직접profiler계측이아니다(JSON에조건별평균저장). train alt_keyword_search 1건의FINAL512token상한truncation오류를그대로포함했다.

## 해석

train 정답5문항 중3문항에서 품질보존 단축이 가능했고 oracle 전체calls12→9(25%),turns17→14(17.65%)였다. 학습정책의 성능이 아니다. nonterminal 단축1문항은사전기준3문항에미달했다. 사후보조분석에서그1문항도head_B1조기종료로같은interaction절감이가능해, 중간transition대체의추가절감은확인되지않았다.

validation8에서head F1.21875/EM.125,최초Native선택 F1.46875/EM.375. 같은관측에서read대신semantic/keyword를고른2문항이정답을회복했다. 이는전체Native정책성능이아닌첫선택의진단이다. baseline정답1문항으로압축확인자료가부족하다. 고정head_B1은trainF1.2676→.2525,validationF1.2188→.1827로낮아져oracle조기종료여지를곧바로고정step축소로배포하면안된다.

권고: 현재도구head+LLM인자생성을유지하고도구선택오류·실패뒤재검색일반화를우선한다. 조기종료는작은가치예측후보로보존한다. 현자료로중간stepdistillation학습이나새인자decoder를시작하지않는다.

## 다음 단계

이진단만으로새학습/RL/decoder를자동실행하지않는다. nonterminal gate가안되면중간stepdistillation학습을시작하지않고, FINAL-only이득은early stopping으로구분한다. 같은도구head+LLM인자생성기준선과원래일반화계획은보존한다.
