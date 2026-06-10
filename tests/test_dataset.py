from scripts.make_dataset import (
    build_prompt,
    control_sentence,
    query_span_in_prompt,
    query_word_for,
    target_is_correct,
)

def test_control_semantics():
    assert target_is_correct('mapping') is True
    assert target_is_correct('source_baseline') is False
    assert target_is_correct('mention') is False
    assert query_word_for('target_baseline', 'dog', 'cat') == 'cat'
    assert query_word_for('mapping', 'dog', 'cat') == 'dog'
    assert 'does not mean' in control_sentence('negation', 'dog', 'cat', 'banana')


def test_query_span_targets_question_not_options():
    prompt = build_prompt(
        "Choose A or B.",
        "The word 'red' has its ordinary English meaning.",
        "red",
        "the color blue",
        "the color red",
    )
    start, end = query_span_in_prompt(prompt, "red")
    assert prompt[start:end] == "red"
    assert start < prompt.index("\nA.")
    assert start > prompt.index("Question:")
