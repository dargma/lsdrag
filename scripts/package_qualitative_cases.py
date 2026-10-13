import json,hashlib,shutil
from pathlib import Path
r=Path('/content/local_fast/lsdrag-work');repo=r/'repo';base=repo/'lab/reports/qualitative_cases';base.mkdir(parents=True,exist_ok=True)
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
SPECS=[
 ('happy-feet','m3docvqa','development','8d3bf560c35d4b2798db545f1b0a7b94','펭귄이 등장하는 영화 포스터','카마니 그리핀의 출연작 중 표지에 펭귄이 있는 영화는?','정답 포스터의 원문 캡션은 없고 “Theatrical release poster”라는 주석만 있다. 펭귄이라는 시각 속성은 이 주석에 없다. Google multimodal은 포스터를 실제로 읽고 Happy Feet를 답했다.'),
 ('laura-davies','m3docvqa','additional50','be0baecf90a40b1ba8230d9c771f75ca','매우 짧은 금발의 선수 찾기','2009 Ladies European Tour 상금 순위에 포함된 선수 중 매우 짧은 금발의 선수는?','Qwen은 답변과 이미지 읽기를 완료하지 못했다. Google multimodal은 Laura Davies의 사진을 읽고 정답을 답했다. 전체 성능 차이와 별도로 확인한 개별 성공 사례다.'),
 ('birth-of-venus','memgallery','development','Architecture_Art_Culture_Exhibition_Technology:qa0068','대화에 나온 작품 이름','첨부 현대 미술 이미지와 비교하면 흥미로울 것으로 대화에서 언급된 작품은?','질문 첨부 이미지는 두 조건 모두 입력하지 않았다. 정답 작품명이 데이터셋의 검색 대상 이미지 캡션에도 명시돼 있다. 따라서 이미지 인식 자체의 필요성을 증명하는 사례가 아니라, 더 적절한 대화·캡션 근거를 찾은 사례다.'),
 ('foxglove-count','memgallery','additional50','Baking_Dessert_Daily_Life_Skill:qa0042','관련 사진을 빠짐없이 찾아 개수 세기','가을 나들이에서 Maya가 공유한 폭스글러브 꽃 사진은 모두 몇 장인가?','Qwen은 정답 사진 5장 중 4장을 읽어 4라고 답했다. Google multimodal은 누락됐던 세 번째 사진까지 찾아 5라고 답했다. 두 조건이 실제로 읽은 사진들을 아래에 모두 표시했다.')]
report=[];index=[]
for slug,bench,split,qid,title,translation,note in SPECS:
 folder=base/slug;folder.mkdir(exist_ok=True);images=folder/'images';images.mkdir(exist_ok=True);runroot=r/'results'/('three_arm_arag8_textquery' if split=='development' else 'three_arm_additional50');corpus=r/('data/mem_gallery_prepared/corpus' if bench=='memgallery' else ('data/ir_development' if split=='development' else 'data/ir_final288'));inputs=r/(('data/mem_gallery_prepared/development' if split=='development' else 'data/mem_gallery_prepared/evaluation') if bench=='memgallery' else ('data/manifests/development' if split=='development' else 'data/manifests/evaluation_50'))
 gold=next(x for x in read(inputs/'gold.jsonl') if x['qid']==qid);question=next(x for x in read(inputs/'questions.jsonl') if x['qid']==qid);chunks={x['chunk_id']:x for x in read(corpus/'chunks.jsonl')};figures={x['figure_id']:x for x in read(corpus/'figures.jsonl')};docs={x['document_id']:x for x in read(corpus/'documents.jsonl')};predictions={};asset_map={};chunk_ids=set();figure_ids=set()
 def asset(path):
  path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest();target=images/(digest[:24]+path.suffix.lower());shutil.copyfile(path,target);assert hashlib.sha256(target.read_bytes()).hexdigest()==digest;rel=target.relative_to(folder).as_posix();asset_map[str(path)]={'path':rel,'sha256':digest};return rel
 save(folder/'raw/question.json',question);save(folder/'raw/gold.json',gold)
 for arm in ['B0','C1','G1']:
  src=runroot/bench/arm/(qid+'.json');z=json.loads(src.read_text());predictions[arm]=z;target=folder/'predictions'/(arm+'.json');target.parent.mkdir(exist_ok=True);shutil.copyfile(src,target);assert src.read_bytes()==target.read_bytes();manifest=runroot/bench/arm/'run_manifest.json';shutil.copyfile(manifest,folder/'predictions'/(arm+'_run_manifest.json'));metrics=next(x for x in read(runroot/bench/(arm+'_scores/question_metrics.jsonl')) if x['qid']==qid);save(folder/'metrics'/(arm+'.json'),metrics);chunk_ids.update(z['read_chunk_ids']);figure_ids.update(z['delivered_figure_ids']);figure_ids.update(z['opened_figure_ids'])
  for event in z['trajectory']:
   if isinstance(event['result'],list):
    for item in event['result']:
     if item.get('chunk_id') in chunks:chunk_ids.add(item['chunk_id'])
     if item.get('figure_id') in figures:figure_ids.add(item['figure_id'])
 question_assets=[asset(path) for path in question.get('question_image_paths',[])]
 if bench=='memgallery':
  chunk_ids.update(gold['scenario']+':'+x for x in gold['clue']);scenario=json.loads((r/'data/mem_gallery/data/dialog'/(gold['scenario']+'.json')).read_text());save(folder/'raw/source_question.json',scenario['human-annotated QAs'][gold['source_qa_index']]);session_ids={chunks[c]['document_id'].split(':')[-1] for c in chunk_ids};sessions=[s for s in scenario['multi_session_dialogues'] if s['session_id'] in session_ids];save(folder/'raw/dialogue_sessions.json',sessions)
  for session in sessions:
   for turn in session['dialogues']:
    for ref in turn.get('input_image',[]):asset(r/'data/mem_gallery/data'/ref.removeprefix('../'))
 else:
  save(folder/'raw/source_question.json',gold)
  sources={x['id']:x for x in read(r/'data/manifests/gold_image_sources.jsonl')}
  for item in gold['supporting_context']:
   if item['doc_part']=='image' and item['doc_id'] in sources:asset(r/'data/gold_images'/Path(sources[item['doc_id']]['path']).name)
  pages=[]
  for did,page_idx in sorted({(chunks[c]['document_id'],chunks[c]['page_index']) for c in chunk_ids}):
   doc=docs[did];parse_root=r/'data'/('parsed_development' if (r/'data/parsed_development'/doc['parsed_source_id']).exists() else 'parsed_final_additions')/doc['parsed_source_id'];content=json.loads((parse_root/'structured_content.json').read_text());page=next(v for v in content['pages'] if v['page_idx']==page_idx);pages.append({'document_id':did,'source_document_id':doc['parsed_source_id'],'page':page})
   for block in page['blocks']:
    if block.get('image_source') and (parse_root/block['image_source']).exists():asset(parse_root/block['image_source'])
  save(folder/'raw/parsed_pages.json',pages)
 figure_ids.update(f for c in chunk_ids for f in chunks[c]['figure_ids'])
 local_figures=[]
 for fid in sorted(figure_ids):
  f=figures[fid];rel=asset(f['image_path']);local_figures.append({**f,'original_image_path':f['image_path'],'image_path':rel})
 save(folder/'raw/chunks.json', [chunks[c] for c in sorted(chunk_ids)]);save(folder/'raw/figures.json',[figures[f] for f in sorted(figure_ids)]);save(folder/'figures_local.json',local_figures);save(folder/'source_assets.json',asset_map)
 caption_rows=[]
 for arm in ['B0','C1','G1']:
  for fid in predictions[arm]['delivered_figure_ids']:
   f=figures[fid];caption_rows.append({'arm':arm,'figure_id':fid,'image_path':asset_map[f['image_path']]['path'],'has_caption':f['has_caption'],'caption':f.get('caption',''),'footnotes':f.get('footnotes',''),'has_original_description':f['has_description']})
 save(folder/'caption_inventory.json',caption_rows)
 name='M3DocVQA' if bench=='m3docvqa' else 'Mem-Gallery';actual_gold=[x['answer'] for x in gold['answers']] if bench=='m3docvqa' else gold['answer'];qimage=bool(question_assets);qcaption=bool(gold.get('image_caption'))
 lines=[f'# {name} · {title}',f'\n문항 `{qid}` · {split}. 모델·데이터 다운로드 없이 이 폴더의 JSON과 이미지를 열면 실행 현황을 확인할 수 있습니다.\n',f'**원문 질문:** {question["question"]}\n',f'**정답:** `{json.dumps(actual_gold,ensure_ascii=False)}`\n',f'**원 질문 첨부 이미지:** {"있음" if qimage else "없음"}. **이 질문 첨부 이미지의 실제 입력:** 없음. 검색 근거 이미지는 아래 실제 입력 기록에 표시한다.\n',f'**원 질문 이미지의 캡션:** 원래 {"있음" if qcaption else "없음"} · 실제 입력하지 않음. 검색 자료의 캡션은 사용.\n', '- [원본 질문](raw/question.json) · [정답·주석](raw/gold.json) · [원 데이터 질문](raw/source_question.json)','- 전체 실행: [Qwen](predictions/B0.json) · [Google text-only](predictions/C1.json) · [Google multimodal](predictions/G1.json)','- [원문 청크](raw/chunks.json) · [원본 이미지 메타데이터](raw/figures.json) · [로컬 이미지 경로](figures_local.json) · [캡션 목록](caption_inventory.json)','- [파일 해시·출처](manifest.json). 기존 원본의 절대 경로는 출처 기록이며 `figures_local.json`의 `image_path`는 이 폴더 기준 상대 경로입니다.','\n## 실제 답변·입력 이미지\n','| 조건 | 답변 | 실제 입력 이미지 수 |','|---|---|---:|']
 for arm,label in [('B0','Qwen'),('C1','Google text-only'),('G1','Google multimodal')]:lines.append(f'| {label} | `{json.dumps(predictions[arm]["answers"],ensure_ascii=False)}` | {len(predictions[arm]["delivered_figure_ids"])} |')
 for arm,label in [('B0','Qwen'),('C1','Google text-only'),('G1','Google multimodal')]:
  lines.append(f'\n### {label}\n');lines.append(' '.join(f'<img src="{asset_map[figures[f]["image_path"]]["path"]}" width="150" alt="실제 Reader 입력 이미지">' for f in predictions[arm]['delivered_figure_ids']) or '전달 이미지 없음.')
 (folder/'README.md').write_text('\n'.join(lines)+'\n')
 hashes=[{'path':f.relative_to(folder).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='manifest.json'];save(folder/'manifest.json',{'schema_version':1,'qid':qid,'benchmark':name,'split':split,'question_image_originally_present':qimage,'question_image_provided':False,'question_caption_originally_present':qcaption,'question_caption_provided':False,'image_delivery_policy':predictions['B0']['config'].get('image_delivery_policy','agent-read'),'prediction_files_byte_identical_to_recorded_runs':True,'purpose':'Self-contained inspection bundle for this selected qualitative case; includes recorded candidates/reads, original source rows/pages, captions and images; no inference replay or full benchmark/index bundle.','files':hashes})
 index.append(f'- [{name}: {title}]({slug}/README.md) · `{qid}`');report.append({'slug':slug,'benchmark':name,'qid':qid,'split':split,'title':title,'translation':translation,'gold':actual_gold,'note':note,'question_assets':question_assets,'predictions':{arm:{'answers':z['answers'],'images':[asset_map[figures[f]['image_path']]['path'] for f in z['delivered_figure_ids']]} for arm,z in predictions.items()},'captions':caption_rows})
(base/'README.md').write_text('# 정성 평가 원본 데이터\n\n각 사례 폴더에 원본 질문·정답, 세 조건의 전체 실행 JSON, 관련 원문·이미지·캡션과 파일 해시를 포함했습니다. 별도 데이터 준비 없이 GitHub에서 열거나 clone 후 확인할 수 있습니다. 전체 벤치마크 재실행을 위한 인덱스 묶음은 아닙니다.\n\n'+'\n'.join(index)+'\n')
(r/'recovery/qualitative-report-inputs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Packaged',len(SPECS),'cases;',sum(1 for f in base.rglob('*') if f.is_file()),'files;',sum(f.stat().st_size for f in base.rglob('*') if f.is_file()),'bytes')
