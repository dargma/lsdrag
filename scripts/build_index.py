import argparse
import json
from lsdrag.index import build_index

p = argparse.ArgumentParser()
p.add_argument("--corpus", required=True)
p.add_argument("--output", required=True)
p.add_argument("--channel", choices=["qwen", "gemma"], required=True)
p.add_argument("--modality", choices=["text", "image"], required=True)
p.add_argument("--cache-dir", required=True)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--device", default="cuda")
a = p.parse_args()
print(json.dumps(build_index(a.corpus, a.output, a.channel, a.modality, a.cache_dir, a.batch_size, a.device)))
