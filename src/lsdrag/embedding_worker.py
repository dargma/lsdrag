"""Local subprocess adapter for the separately pinned embedding environment."""
import json
import subprocess
import numpy as np


class ProcessEmbedder:
    def __init__(self, python, cache_dir):
        program = '''import json,sys
from lsdrag.embeddings import Embedder
model=Embedder("gemma",sys.argv[1])
print(json.dumps({"ready":True}),flush=True)
for line in sys.stdin:
    req=json.loads(line)
    try:
        vectors=model.encode(req["inputs"],req["role"])
        print(json.dumps({"vectors":vectors.tolist()}),flush=True)
    except Exception as error:
        print(json.dumps({"error":str(error)}),flush=True)
'''
        self.process = subprocess.Popen([str(python), '-u', '-c', program, str(cache_dir)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self._receive()
        if not ready.get('ready'): raise RuntimeError(f'Embedding worker failed: {ready}')

    def _receive(self):
        while True:
            line=self.process.stdout.readline()
            if not line: raise RuntimeError('Local embedding worker exited')
            try: return json.loads(line)
            except json.JSONDecodeError: continue

    def encode(self, inputs, role):
        self.process.stdin.write(json.dumps({'inputs':inputs,'role':role})+'\n')
        self.process.stdin.flush()
        response=self._receive()
        if 'error' in response: raise RuntimeError(response['error'])
        return np.asarray(response['vectors'],dtype=np.float32)

    def close(self):
        self.process.stdin.close()
        try: self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate();self.process.wait(timeout=10)
