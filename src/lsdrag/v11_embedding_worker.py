"""Isolated pinned embedding runtime; images cross the boundary as file paths."""
import json
from pathlib import Path
import subprocess
import threading

import numpy as np

_WORKER = r'''
import json, sys
from PIL import Image
import transformers
from lsdrag.embeddings import Embedder
model = Embedder('gemma', sys.argv[1])
print(json.dumps({'ready': True, 'transformers': transformers.__version__, 'dimension': model.dimension}), flush=True)
for line in sys.stdin:
    images = []
    try:
        request = json.loads(line)
        if request.get('close'):
            break
        values = request['inputs']
        if request['role'] == 'image':
            for path in values:
                with Image.open(path) as source:
                    images.append(source.convert('RGB'))
            values = images
        vectors = model.encode(values, request['role'], titles=request.get('titles'))
        print(json.dumps({'vectors': vectors.tolist()}), flush=True)
    except Exception as error:
        print(json.dumps({'error': type(error).__name__ + ': ' + str(error)}), flush=True)
    finally:
        for image in images:
            image.close()
'''


class V11ProcessEmbedder:
    dimension = 768

    def __init__(self, python='/content/local_fast/lsdrag-v11/embedding-env/bin/python', cache_dir='/content/local_fast/lsdrag-v11/hf'):
        self._lock = threading.Lock()
        self.process = subprocess.Popen([str(python), '-u', '-c', _WORKER, str(cache_dir)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.ready = self._receive()
            if not self.ready.get('ready') or self.ready.get('dimension') != self.dimension:
                raise RuntimeError(f'Unexpected embedding worker handshake: {self.ready}')
        except BaseException:
            self.close()
            raise

    def _receive(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f'Embedding worker exited (returncode={self.process.poll()})')
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                # Third-party startup logs may appear on stdout. Protocol objects
                # alone count as replies; stderr remains inherited and visible.
                continue
            if isinstance(result, dict) and any(k in result for k in ('ready', 'vectors', 'error')):
                return result

    def encode(self, inputs, role, titles=None):
        if role not in {'query', 'document', 'image'}:
            raise ValueError(role)
        values = list(inputs)
        if role == 'image':
            paths = []
            for image in values:
                path = image if isinstance(image, (str, Path)) else getattr(image, 'filename', None)
                if not path or not Path(path).is_file():
                    raise ValueError('Worker images require an existing audited filename')
                paths.append(str(Path(path).resolve()))
            values = paths
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError('Embedding worker is not running')
            self.process.stdin.write(json.dumps({'inputs': values, 'role': role, 'titles': titles})+'\n')
            self.process.stdin.flush()
            response = self._receive()
        if 'error' in response:
            raise RuntimeError(response['error'])
        vectors = np.asarray(response['vectors'], dtype=np.float32)
        if vectors.shape != (len(values), self.dimension) or not np.isfinite(vectors).all():
            raise ValueError('Invalid isolated embedding result')
        return vectors

    def close(self):
        process = getattr(self, 'process', None)
        if process is None:
            return
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
