import numpy as np

from lsdrag.v11_training import infer_head, load_head, train_heads, training_arrays


def _row(qid, value):
    return {'qid': qid, 'source_group': qid, 'features': [[value, 1., 0.], [value, 1., 1.]], 'stops': [{'f1': .2, 'cost_seconds': 1.}, {'f1': .4, 'cost_seconds': 2.}], 'actions': {action: {'N0': .2, 'J': -.1, 'cells': {cell: {'f1': score, 'cost_seconds': 3.} for cell, score in [('00', .3), ('01', .4), ('10', .5), ('11', .8)]}} for action in ('TEXT_ACQUIRE', 'IMAGE_ACQUIRE')}}


def test_targets_checkpoint_and_matched_capacity(tmp_path):
    rows = [_row('a', 0.), _row('b', 2.)]
    x, y, aux, mask = training_arrays(rows, .01)
    np.testing.assert_allclose(y[:2], [[.19, .27, .27], [.38, .77, .77]], atol=1e-6)
    np.testing.assert_array_equal(mask, [0, 1, 0, 1])
    np.testing.assert_allclose(aux[1], [.2, -.1, .2, -.1])
    result = train_heads(rows, tmp_path, lambdas=(.01,), seeds=(7,), epochs=3)
    assert len(result['runs']) == 2
    heads = [load_head(r['checkpoint']) for r in result['runs']]
    assert sum(p.numel() for p in heads[0]['model'].parameters()) == sum(p.numel() for p in heads[1]['model'].parameters())
    np.testing.assert_allclose(heads[0]['mean'].numpy(), [1., 1., .5])
    for head in heads:
        assert infer_head(head, x).shape == (4, 3)
        np.testing.assert_allclose(infer_head(head, x[0]), infer_head(head, x)[0], atol=1e-7)
        assert np.isfinite(infer_head(head, [100., 1., 0.])).all()
        np.testing.assert_allclose(head['mean'].numpy(), [1., 1., .5])
