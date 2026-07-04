"""Tests for MaskedProteinDataset.

    1. mask ratio        — over a long seq, masked fraction ~= 0.15   (property)
    2. specials never masked — [PAD]/[CLS] positions are never chosen (property)
    3. padding           — pad positions carry [PAD]; pad_mask marks them (property)

All rely on the dataset's masking being SEEDED (reproducible). Build a Tokenizer
and a tiny synthetic sequence set in-code — never hit the network here (CI runs
these; the real download is a separate, un-CI'd script).

The `# torch equivalent:` comments show the tensor-native way to write each
assertion (would need `import torch`); the plain-Python form is used to keep the
tests dependency-light.
"""

from __future__ import annotations

from miniesm.data import IGNORE_INDEX, MaskedProteinDataset
from miniesm.tokenizer import Tokenizer

SHORT = "ACDEFG"
LONG = "ACDEFGHIKLMNPQRSTVWY" * 5


def test_mask_ratio_approximately_15_percent() -> None:
    """Masked fraction of the MASKABLE positions is ~0.15 (within a tolerance)."""
    # Count masked positions (labels != IGNORE_INDEX) over the maskable count
    # (real tokens minus the [CLS]); assert it lands in a band around 0.15.
    tok = Tokenizer()
    dataset = MaskedProteinDataset([LONG], tok, seed=42)
    example = dataset[0]

    n_masked = (example["labels"] != IGNORE_INDEX).sum().item()
    n_maskable = example["pad_mask"].sum().item() - 1  # real tokens minus [CLS]
    masked_prop = n_masked / n_maskable

    assert masked_prop >= 0.15 * 0.9 and masked_prop <= 0.15 / 0.9


def test_special_tokens_never_masked() -> None:
    """No special-token position is ever chosen for masking/replacement."""
    # [CLS] (position 0) and every [PAD] position (the tail past the real tokens)
    # must keep IGNORE_INDEX labels — i.e. are never selected for masking.
    tok = Tokenizer()
    dataset = MaskedProteinDataset([SHORT], tok, seed=42, max_len=10)
    example = dataset[0]
    n_real = 1 + len(SHORT)  # [CLS] + residues

    assert example["labels"][0].item() == IGNORE_INDEX
    # torch equivalent: torch.equal(example["labels"][0], torch.tensor(IGNORE_INDEX))
    assert example["input_ids"][0].item() == tok.encode_map["[CLS]"]
    # torch equivalent: torch.equal(example["input_ids"][0], torch.tensor(tok.encode_map["[CLS]"]))
    assert (example["labels"][n_real:] == IGNORE_INDEX).all()
    # torch equivalent: torch.all(example["labels"][n_real:] == IGNORE_INDEX)


def test_padding_shapes_and_mask() -> None:
    """Short sequences pad to a common length; pad_mask marks real vs pad."""
    # SHORT (len 6) with max_len 10 forces padding. Check all tensors are max_len
    # long, pad_mask marks exactly the real tokens, and the tail carries [PAD].
    tok = Tokenizer()
    dataset = MaskedProteinDataset([SHORT], tok, seed=42, max_len=10)
    example = dataset[0]
    n_real = 1 + len(SHORT)  # [CLS] + residues
    pad_id = tok.encode_map["[PAD]"]

    assert len(example["input_ids"]) == 10
    # torch equivalent: example["input_ids"].shape[0] == 10
    assert len(example["labels"]) == 10
    assert len(example["pad_mask"]) == 10
    assert example["pad_mask"].sum().item() == n_real
    # torch equivalent: int(torch.sum(example["pad_mask"])) == n_real
    assert (example["input_ids"][n_real:] == pad_id).all()
    # torch equivalent: torch.all(example["input_ids"][n_real:] == pad_id)
