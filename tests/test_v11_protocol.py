import json

from lsdrag.v11_protocol import DECISION, contrast_summary, effects, observation_messages


def test_observations_do_not_echo_search_or_private_fields():
    item = {'item_id': 'opaque', 'modality': 'text', 'text': 'Actual corpus text',
            'title': 'Title', 'query': 'DO_NOT_ECHO_QUERY', 'answers': ['PRIVATE_GOLD'],
            'support_ids': ['PRIVATE_SUPPORT'], 'url': 'PRIVATE_URL'}
    encoded = json.dumps(observation_messages('Question?', [item], DECISION))
    assert 'Actual corpus text' in encoded
    for private in ('DO_NOT_ECHO_QUERY', 'PRIVATE_GOLD', 'PRIVATE_SUPPORT', 'PRIVATE_URL'):
        assert private not in encoded


def test_crossed_effects_and_no_spurious_tie_change():
    cells = {k: {'f1': v} for k,v in [('00',0.),('01',.4),('10',.2),('11',.9)]}
    effect = effects({'cells': cells})
    assert abs(effect['N0']-.4) < 1e-8
    assert abs(effect['J']-.3) < 1e-8
    row = {'qid': 'a', 'source_group':'g', 'stops':[{'f1':1.},{'f1':1.}],
           'actions':{name:{'cells': cells, **effect}
                      for name in ('TEXT_ACQUIRE','IMAGE_ACQUIRE')}}
    result = contrast_summary([row])
    assert result['contrast_questions'] == 1
    assert result['strict_preference_change_questions'] == 0
