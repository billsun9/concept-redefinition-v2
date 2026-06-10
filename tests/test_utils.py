from transformers import AutoTokenizer
from redef.utils import (
    final_occurrence_span,
    map_prompt_span_to_formatted,
    token_char_span,
)

def test_alignment_tiny_gpt2():
    tok = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2', use_fast=True)
    text = "dog cat dog"
    s,e = final_occurrence_span(text, 'dog')
    idxs = token_char_span(tok, text, s, e)
    assert len(idxs) >= 1


def test_map_prompt_span_to_chat_wrapper():
    raw = "Question: Is 'red' the queried word?"
    start = raw.index("red")
    formatted = f"<user>\n{raw}\n</user>\n<assistant>\n"
    mapped_start, mapped_end = map_prompt_span_to_formatted(
        raw, formatted, start, start + len("red")
    )
    assert formatted[mapped_start:mapped_end] == "red"
