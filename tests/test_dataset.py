from scripts.make_dataset import (
    build_prompt,
    canonical_pair_key,
    control_sentence,
    definition_span_in_prompt,
    query_span_in_prompt,
    query_word_for,
    target_is_correct,
)
from redef.utils import load_json, read_jsonl

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


def test_definition_spans_target_note_not_question_or_options():
    prompt = build_prompt(
        "Choose A or B.",
        "In this note, the word 'red' means 'blue'.",
        "red",
        "the color blue",
        "the color red",
    )
    source_span = definition_span_in_prompt(prompt, "red")
    target_span = definition_span_in_prompt(prompt, "blue")
    assert prompt[slice(*source_span)] == "red"
    assert prompt[slice(*target_span)] == "blue"
    assert source_span[0] < prompt.index("Question:")
    assert target_span[0] < prompt.index("Question:")


def test_every_concept_pair_has_a_category():
    pairs = read_jsonl("data/concept_pairs.jsonl")
    categories = load_json("data/concept_categories.json")
    keys = {
        canonical_pair_key(pair["source"], pair["target"])
        for pair in pairs
    }
    assert keys == set(categories)
