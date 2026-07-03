"""Tests for the tokenizer.

1. round-trip          — decode(encode(seq)) == seq        (known-answer)
2. [PAD] is id 0       — the padding convention holds       (property)
3. unknown char        — encode fails loud with KeyError    (property)
4. skip_special_tokens — decode drops vs. keeps specials    (known-answer)
"""

from __future__ import annotations

import pytest

from miniesm.tokenizer import Tokenizer


def test_roundtrip_all_residues() -> None:
    """decode(encode(seq)) == seq for an all-residue string."""
    # TODO(you):
    # Toolbox: instantiate Tokenizer; round-trip a string through encode then
    #   decode; compare to the original with ==.
    # (Use a seq that includes every residue at least once for a strong check.)
    tok = Tokenizer()
    seq = "ACDEFGHIKLMNPQRSTVWY"
    encode_seq = tok.encode(seq)
    decode_seq = tok.decode(encode_seq)
    assert seq == decode_seq


def test_pad_is_id_zero() -> None:
    """The [PAD] token must map to id 0."""
    # TODO(you): reach into the vocab and check [PAD]'s id.
    tok = Tokenizer()
    assert tok.encode_map["[PAD]"] == 0


def test_unknown_character() -> None:
    """encode() fails loud on a character outside the 20 residues (e.g. 'X').

    We chose fail-loud (no [UNK] token), so an out-of-vocab character must raise
    KeyError rather than being silently mapped.
    """
    # TODO(you):
    # Toolbox (only if you chose to fail loudly): pytest.raises — a context
    #   manager that asserts a given exception type is raised inside it.
    tok = Tokenizer()
    with pytest.raises(KeyError):
        tok.encode("ACDX")


def test_skip_special_tokens() -> None:
    """decode()'s skip_special_tokens parameter, both modes.

    With skip_special_tokens=True (default) the special tokens are dropped; with
    False they are rendered explicitly (e.g. "[MASK]").
    """
    tok = Tokenizer()
    ids = [1, 3, 5, 7]
    assert tok.decode(ids) == "ADF"
    assert tok.decode(ids, skip_special_tokens=False) == "[MASK]ADF"
