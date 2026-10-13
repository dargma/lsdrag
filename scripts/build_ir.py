import argparse
import json
from lsdrag.ir import build_ir

p = argparse.ArgumentParser()
p.add_argument("--parsed-root", required=True)
p.add_argument("--corpus-manifest", required=True)
p.add_argument("--output", required=True)
p.add_argument("--max-words", type=int, default=300)
a = p.parse_args()
print(json.dumps(build_ir(a.parsed_root, a.corpus_manifest, a.output, a.max_words)))
