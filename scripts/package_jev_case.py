"""Package the sole recorded B-HF EM0 -> D-head EM1 case. CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pymupdf
from lsdrag.io import read_jsonl,dump_json,sha256

p=argparse.ArgumentParser();p.add_argument('--work-root',type=Path,required=True);a=p.parse_args();r=a.work_root
repo=Path(__file__).resolve().parents[1];dest=repo/'lab/reports/jev_v1.0/qualitative/kerry-condon';dest.mkdir(parents=True,exist_ok=True)
qid='6068540027afe0ca92d9386391bf0889';run=r/'results/jev_v1.0/paired_m3_test20'
summary=json.loads((run/'summary.json').read_text());metrics=[x for x in summary['rows'] if x['qid']==qid]
assert {x['arm']:x['em'] for x in metrics}=={'B-HF':0.,'D-head':1.}
question=next(x for x in read_jsonl(r/'data/manifests/jev_v1.0/fresh_m3_test20/questions.jsonl') if x['qid']==qid)
gold=next(x for x in read_jsonl(r/'data/manifests/jev_v1.0/fresh_m3_test20/gold.jsonl') if x['qid']==qid)
for name,obj in [('question',question),('gold',gold)]:dump_json(dest/'raw'/(name+'.json'),obj)
pred={};chunk_ids=set();figure_ids=set();source_hashes={}
for arm in ['B-HF','D-head']:
 source=run/arm/(qid+'.json');target=dest/'runs'/(arm+'.json');target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target);source_hashes[arm]=sha256(source);assert sha256(source)==sha256(target)
 z=json.loads(source.read_text());pred[arm]=z;chunk_ids.update(z['read_chunk_ids']);figure_ids.update(z['delivered_figure_ids'])
 for t in z['trajectory']:
  if isinstance(t['result'],list):chunk_ids.update(x['chunk_id'] for x in t['result'] if 'chunk_id' in x)
chunks=[c for c in read_jsonl(r/'data/ir_final288/chunks.jsonl') if c['chunk_id'] in chunk_ids]
figures=[f for f in read_jsonl(r/'data/ir_final288/figures.jsonl') if f['figure_id'] in figure_ids]
doc=next(d for d in read_jsonl(r/'data/ir_final288/documents.jsonl') if d['document_id']=='55296a438f6104414645d72f45a08b2e')
for name,obj in [('chunks',chunks),('figures',figures),('document',doc),('metrics',metrics)]:dump_json(dest/'raw'/(name+'.json'),obj)
assert len(figures)==1;image=figures[0];target=dest/'images/reader_portrait.jpg';target.parent.mkdir(exist_ok=True);shutil.copyfile(image['image_path'],target);assert sha256(target)==image['image_sha256']
dump_json(dest/'raw/figure_local.json',{**image,'original_image_path':image['image_path'],'image_path':'../images/reader_portrait.jpg'})
table=next(c for c in chunks if c['chunk_id']=='55296a438f6104414645d72f45a08b2e:p2:b0:s0')
evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
(evidence/'full_table.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Recorded Kerry Condon filmography evidence</title><h1>Recorded filmography table</h1><p>Source: <a href="https://en.wikipedia.org/wiki/Kerry_Condon">Kerry Condon</a>. Experiment snapshot, not a live page. Full table was read only by B-HF; D-head saw a search snippet.</p>'+table['text'].split('\n',1)[1]+'</html>')
for arm,z in pred.items():
 snippet=next(x['snippet'] for x in z['trajectory'][0]['result'] if x['chunk_id']==table['chunk_id'])
 (evidence/(arm+'_table_snippet.txt')).write_text(snippet+'\n')
assert pred['B-HF']['trajectory'][0]['arguments']==pred['D-head']['trajectory'][0]['arguments']
assert pred['B-HF']['trajectory'][0]['result']==pred['D-head']['trajectory'][0]['result']
# Render the stored source PDF, not an invented image. This page was not a
# Reader image input in either arm; included solely for human evidence audit.
pdf=r/'data/pdfs'/('55296a438f6104414645d72f45a08b2e.pdf')
with pymupdf.open(pdf) as source:source[2].get_pixmap(matrix=pymupdf.Matrix(1,1)).save(dest/'images/source_pdf_page3.png')
shutil.copyfile(run/'manifest.json',dest/'raw/paired_manifest.json')
files=[{'path':p.relative_to(dest).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(dest.rglob('*')) if p.is_file() and p.name not in {'manifest.json','README.md'}]
dump_json(dest/'manifest.json',{'qid':qid,'benchmark':'M3DocVQA','selection':'only recorded B-HF EM0 -> D-head EM1 case in fixed paired20; no new solves','prediction_files_byte_identical':True,'source_prediction_sha256':source_hashes,'source_pdf_sha256':sha256(pdf),'source_pdf_page_zero_index':2,'source_pdf_page_is_reader_input':False,'reader_input_images':['images/reader_portrait.jpg'],'question_has_image':False,'question_has_image_caption':False,'retrieved_image_has_caption':True,'retrieved_image_caption':image['caption'],'retrieved_image_footnote':image['footnotes'],'both_first_search_results_identical':True,'source_url':doc['url'],'files':files})
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files)}))
