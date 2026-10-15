"""Frozen multimodal reader and separately indexed evidence for v1.1 pilots."""
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .hf_reader import processor_messages
from .embeddings import Embedder, MODELS, normalized_float32

ACTIONS = ('ANSWER_AND_STOP', 'TEXT_ACQUIRE', 'IMAGE_ACQUIRE')


class V11Backend:
    def __init__(self, model_path, image_soft_tokens=280, max_context=16384):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True).eval()
        self.model.requires_grad_(False)
        self.image_soft_tokens, self.max_context = image_soft_tokens, max_context
        self.feature = None
        self.hook = self.model.model.language_model.norm.register_forward_hook(self._capture)
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])
        self.label_ids = []
        for label in 'ABC':
            ids = self.processor.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f'Action label must be one token: {label}: {ids}')
            self.label_ids.append(ids[0])

    def _capture(self, module, args, output):
        self.feature = output[:, -1, :].detach().clone()

    def _sync(self):
        self.torch.cuda.synchronize()

    def _prefill(self, messages, budget):
        start = time.perf_counter()
        inputs = self.processor.apply_chat_template(processor_messages(messages), add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors='pt', processor_kwargs={'max_soft_tokens': self.image_soft_tokens})
        inputs = inputs.to(self.model.device)
        self._sync()
        preprocess = time.perf_counter() - start
        if inputs['input_ids'].shape[1] + budget > self.max_context:
            raise ValueError('Actual multimodal context exceeds fixed budget')
        tick = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(**inputs, use_cache=True, logits_to_keep=1)
        self._sync()
        feature = self.feature[0].float().cpu().numpy().copy()
        usage = dict(wall_seconds=0., preprocess_seconds=preprocess, prefill_seconds=time.perf_counter()-tick, decode_seconds=0., prompt_tokens=int(inputs['input_ids'].shape[1]), generated_tokens=0, forced_prefix_tokens=0)
        return inputs, output, feature, usage, start

    def prefill_readout(self, messages):
        inputs, output, feature, usage, start = self._prefill(messages, 1)
        probabilities = output.logits[0, -1, self.label_ids].float().softmax(-1).cpu().tolist()
        self._sync()
        usage['wall_seconds'] = time.perf_counter()-start
        return {'feature': feature, 'probs': probabilities, 'usage': usage}

    def encode_state(self, messages):
        result = self.prefill_readout(messages)
        return {**result, 'action_probs': result['probs']}

    def decide(self, messages, mode='first_token', legal_actions=None, max_new_tokens=48):
        if mode == 'first_token' and legal_actions is not None:
            result = self.prefill_readout(messages)
            indices = [ACTIONS.index(a) if isinstance(a, str) else int(a) for a in legal_actions]
            if not indices:
                raise ValueError('No legal actions')
            probabilities = np.asarray(result['probs'])
            index = max(indices, key=lambda i: probabilities[i])
            result['usage']['generated_tokens'] = 1
            return {**result, 'action': ACTIONS[index], 'index': index}
        result = self.decisions(messages, mode=mode, max_new_tokens=max_new_tokens)
        if legal_actions is not None:
            legal_indices = [ACTIONS.index(a) if isinstance(a, str) else int(a) for a in legal_actions]
            if result['index'] not in legal_indices:
                raise ValueError('Generated illegal action')
        return result

    def generate(self, messages, max_new_tokens=96, prefix=''):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        forced = self.processor.tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
        if len(forced) >= max_new_tokens:
            raise ValueError('Forced prefix exhausts output budget')
        inputs, output, feature, usage, start = self._prefill(messages, max_new_tokens)
        torch = self.torch
        mask = inputs['attention_mask']
        generated = []
        self._sync()
        tick = time.perf_counter()
        with torch.inference_mode():
            if forced:
                ids = torch.tensor([forced], device=self.model.device)
                mask = torch.cat([mask, torch.ones_like(ids)], -1)
                output = self.model(input_ids=ids, attention_mask=mask, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
            for _ in range(max_new_tokens-len(forced)):
                token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                value = int(token.item())
                generated.append(value)
                if value in self.stop_ids:
                    break
                mask = torch.cat([mask, torch.ones_like(token)], -1)
                output = self.model(input_ids=token, attention_mask=mask, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        self._sync()
        usage.update(decode_seconds=time.perf_counter()-tick, wall_seconds=time.perf_counter()-start, generated_tokens=len(generated), forced_prefix_tokens=len(forced))
        return {'text': self.processor.tokenizer.decode(forced+generated, skip_special_tokens=True), 'feature': feature, 'usage': usage}

    def decisions(self, messages, mode='first_token', max_new_tokens=48):
        if mode == 'first_token':
            result = self.prefill_readout(messages)
            index = int(np.argmax(result['probs']))
            result['usage']['generated_tokens'] = 1
        elif mode == 'generated':
            result = self.generate(messages, max_new_tokens=max_new_tokens)
            match = re.search(r'\b([ABC])\b', result['text'])
            if not match:
                raise ValueError(f'Invalid generated decision: {result["text"]!r}')
            index = 'ABC'.index(match[1])
            result['probs'] = None
        else:
            raise ValueError(mode)
        return {**result, 'action': ACTIONS[index], 'index': index}


class V11Retriever:
    """No gold scope or caption substitution: image vectors use image pixels."""
    def __init__(self, items, index_dir, cache_dir=None, embedder=None):
        if isinstance(items, (str, Path)):
            items = [json.loads(line) for line in Path(items).read_text().splitlines() if line.strip()]
        self.items = list(items)
        if len({r['item_id'] for r in self.items}) != len(self.items):
            raise ValueError('Duplicate corpus item IDs')
        if any(r['modality'] not in {'text', 'image'} for r in self.items):
            raise ValueError('Unknown corpus modality')
        self.embedder = embedder or Embedder('gemma', cache_dir)
        self.query_cache = {}
        directory = Path(index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # Include pixel hashes, so changing image bytes invalidates cached vectors.
        source = []
        for row in self.items:
            item = dict(row)
            if row['modality'] == 'image':
                item['pixel_sha256'] = hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()
            source.append(item)
        digest = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        expected = {'source_sha256': digest, 'model': MODELS['gemma'][0], 'revision': MODELS['gemma'][1]}
        manifest = directory/'manifest.json'
        self.rows, self.vectors = {}, {}
        cached = manifest.exists() and json.loads(manifest.read_text()) == expected
        for modality in ('text', 'image'):
            rows = [r for r in self.items if r['modality'] == modality]
            self.rows[modality] = rows
            path = directory/f'{modality}.npy'
            if cached and path.exists():
                vectors = np.load(path, allow_pickle=False)
            elif not rows:
                vectors = np.empty((0, self.embedder.dimension), dtype=np.float32)
            elif modality == 'text':
                vectors = self.embedder.encode([r['text'] for r in rows], 'document', titles=[r.get('title', 'none') for r in rows])
            else:
                from PIL import Image
                images = []
                try:
                    for row in rows:
                        with Image.open(row['image_path']) as img:
                            images.append(img.convert('RGB'))
                    vectors = self.embedder.encode(images, 'image')
                finally:
                    for img in images:
                        img.close()
            if vectors.shape != (len(rows), self.embedder.dimension) or not np.isfinite(vectors).all():
                raise ValueError('Corrupt index dimensions or values')
            self.vectors[modality] = normalized_float32(vectors) if len(rows) else vectors
            if not cached or not path.exists():
                np.save(path, self.vectors[modality], allow_pickle=False)
        manifest.write_text(json.dumps(expected, indent=2)+'\n')

    def search(self, query, modality, k=2, exclude=()):
        if modality not in self.rows:
            raise ValueError(modality)
        if k <= 0:
            return []
        if query not in self.query_cache:
            self.query_cache[query] = self.embedder.encode([query], 'query')[0]
        scores = self.vectors[modality] @ self.query_cache[query]
        excluded = set(exclude)
        order = np.argsort(-scores, kind='stable')
        results = []
        for i in order:
            row = self.rows[modality][int(i)]
            if row['item_id'] in excluded:
                continue
            results.append({**row, 'score': float(scores[i])})
            if len(results) == k:
                break
        return results
