import json
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path('/content/local_fast/lsdrag-work')
models=json.loads((root/'repo/lab/reports/model_preflight.json').read_text())
for m in models:
 if '26B' in m['model']: continue
 print('Downloading',m['model'],m['sha'],flush=True)
 path=snapshot_download(m['model'],revision=m['sha'],cache_dir=str(root/'hf'),ignore_patterns=['*.onnx','onnx/*','*.tflite','*.h5','*.msgpack','*.gguf','*.bin'],max_workers=4)
 print('COMPLETE',m['model'],path,flush=True)
