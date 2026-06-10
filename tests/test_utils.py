from transformers import AutoTokenizer
from redef.utils import final_occurrence_span, token_char_span

def test_alignment_tiny_gpt2():
    tok = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2', use_fast=True)
    text = "dog cat dog"
    s,e = final_occurrence_span(text, 'dog')
    idxs = token_char_span(tok, text, s, e)
    assert len(idxs) >= 1
