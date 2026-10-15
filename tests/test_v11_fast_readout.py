from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lsdrag.v11_fast_readout import fast_encode_state


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return Inputs(input_ids=torch.tensor([[2, 3, 4]]))


class ForbiddenHead(torch.nn.Linear):
    def forward(self, *args):
        raise AssertionError('Full-vocab projection was invoked')


class Backbone:
    def __call__(self, **kwargs):
        assert kwargs['use_cache'] and kwargs['return_dict']
        assert 'logits_to_keep' not in kwargs
        return SimpleNamespace(last_hidden_state=torch.tensor([[[1., 2.], [3., 4.], [2., -1.]]], dtype=torch.bfloat16))


class Model:
    device = 'cpu'

    def __init__(self):
        self.model = Backbone()
        self.head = ForbiddenHead(2, 8, bias=True, dtype=torch.bfloat16)
        with torch.no_grad():
            self.head.weight.copy_(torch.arange(16).reshape(8, 2)/3.)
            self.head.bias.copy_(torch.arange(8)/5.)
        self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(final_logit_softcapping=2.))

    def get_output_embeddings(self):
        return self.head


def test_backbone_only_and_selected_logits():
    backend = SimpleNamespace(torch=torch, processor=Processor(), model=Model(), image_soft_tokens=1120, max_context=100, label_ids=[1, 3, 6], _sync=lambda: None)
    result = fast_encode_state(backend, [{'role': 'user', 'content': 'question'}])
    np.testing.assert_array_equal(result['feature'], [2., -1.])
    assert result['feature'].dtype == np.float32
    assert result['probs'] is None
    assert result['usage']['vocab_projection'] == 'none'
    assert result['usage']['generated_tokens'] == 0
    assert backend.processor.kwargs['processor_kwargs'] == {'max_soft_tokens': 1120}
    selected = fast_encode_state(backend, [], selected_token_logits=True)
    # Reference uses functional full projection; forbidden module forward is never called.
    h = torch.tensor([[2., -1.]], dtype=torch.bfloat16)
    full = torch.nn.functional.linear(h, backend.model.head.weight, backend.model.head.bias)
    full = torch.tanh(full/2.)*2.
    expected = full[0, backend.label_ids].float().softmax(-1)
    np.testing.assert_allclose(selected['probs'], expected.detach().numpy(), atol=.003)
    assert selected['usage']['vocab_projection'] == '3rows'
    assert selected['usage']['decode_seconds'] == 0
    backend.max_context = 3
    with pytest.raises(ValueError, match='context'):
        fast_encode_state(backend, [])
