import json

import numpy as np
from PIL import Image

from lsdrag.v11_backend import V11Retriever


class FakeEmbedder:
    dimension = 2

    def __init__(self):
        self.calls = []

    def encode(self, values, role, titles=None):
        self.calls.append((role, values))
        if role == 'image':
            assert all(isinstance(v, Image.Image) for v in values)
            return np.array([[0., 1.]] * len(values), dtype=np.float32)
        return np.array([[1., 0.]] * len(values), dtype=np.float32)


def test_separate_pixels_cache_and_exclusion(tmp_path):
    image = tmp_path/'image.png'
    Image.new('RGB', (2, 2), 'red').save(image)
    rows = [{'item_id': 't1', 'modality': 'text', 'text': 'passage'},
            {'item_id': 't2', 'modality': 'text', 'text': 'other'},
            {'item_id': 'i1', 'modality': 'image', 'image_path': str(image), 'text': 'caption must not be encoded'}]
    embedding = FakeEmbedder()
    index = V11Retriever(rows, tmp_path/'index', embedder=embedding)
    assert [role for role, _ in embedding.calls] == ['document', 'image']
    assert index.search('query', 'text', exclude=['t1'])[0]['item_id'] == 't2'
    assert index.search('query', 'image')[0]['item_id'] == 'i1'
    assert sum(role == 'query' for role, _ in embedding.calls) == 1
    cached_embedding = FakeEmbedder()
    V11Retriever(rows, tmp_path/'index', embedder=cached_embedding)
    assert not cached_embedding.calls
    old_manifest = json.loads((tmp_path/'index'/'manifest.json').read_text())
    Image.new('RGB', (2, 2), 'blue').save(image)
    V11Retriever(rows, tmp_path/'index', embedder=cached_embedding)
    assert [role for role, _ in cached_embedding.calls] == ['document', 'image']
    assert old_manifest != json.loads((tmp_path/'index'/'manifest.json').read_text())
