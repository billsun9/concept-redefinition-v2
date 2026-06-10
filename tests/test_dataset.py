from scripts.make_dataset import control_sentence, target_is_correct, query_word_for

def test_control_semantics():
    assert target_is_correct('mapping') is True
    assert target_is_correct('source_baseline') is False
    assert target_is_correct('mention') is False
    assert query_word_for('target_baseline', 'dog', 'cat') == 'cat'
    assert query_word_for('mapping', 'dog', 'cat') == 'dog'
    assert 'does not mean' in control_sentence('negation', 'dog', 'cat', 'banana')
