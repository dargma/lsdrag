"""Bundle terminal run artifacts to Drive and verify exact copied bytes."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();assert Path(args.run).name==args.run
 run=ROOT/'results'/args.run;assert json.loads((run/'status.json').read_text())['phase']=='complete'
 local=ROOT/'results'/(args.run+'.tar.gz');temp=local.with_suffix('.partial')
 with tarfile.open(temp,'w:gz') as t:t.add(run,arcname=run.name)
 temp.replace(local);expected=digest(local);dest=Path('/content/drive/MyDrive/lsdrag-backups/20261023')/local.name;dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_suffix('.partial');shutil.copyfile(local,partial);assert digest(partial)==expected;partial.replace(dest)
 with tarfile.open(dest) as t:assert any(x.name==args.run+'/status.json' for x in t.getmembers())
 record={'path':str(dest),'sha256':expected,'bytes':dest.stat().st_size,'run':str(run),'phase':'complete','backbone_weights_included':False};p=REPO/'lab/registry'/args.run/'archive.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
