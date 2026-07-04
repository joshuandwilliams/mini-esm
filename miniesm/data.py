"""MaskedProteinDataset: turn raw sequences into BERT-style masked examples.

This is the self-supervised task MiniESM learns from. For each sequence we hide
~15% of the residue positions and ask the model to predict them from context.
One example is (roughly):

    input_ids : the token ids, but with [MASK] substituted at the chosen spots
    labels    : the ORIGINAL ids at the chosen spots, and an ignore-value
                everywhere else (so the Day-5 loss only scores masked spots)
    pad_mask  : 1 = real token, 0 = padding  (feeds Day-2's attention mask hook)

The masking split (BERT): of the chosen positions, 80% -> [MASK], 10% -> a random
residue, 10% -> left unchanged. This stops the model relying on [MASK] appearing
(it never does at inference). [CLS] is prepended at position 0 and never masked.

Design contract (shape / behaviour — the how is yours):
    __len__            : number of sequences
    __getitem__(i)     : ONE masked example for sequence i (not a batch), as a
                         dict with keys "input_ids", "labels", "pad_mask"
    reproducible       : same seed -> same masking, every run (tests depend on it)

See your Obsidian notes for the alphabet; see tokenizer.py for encode/decode.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

# Torch's conventional "ignore this position in the loss" label. Non-masked
# positions in `labels` should carry this so they contribute no gradient.
IGNORE_INDEX = -100


class MaskedProteinDataset(Dataset):
    """A map-style dataset yielding one masked example per sequence.

    Subclasses torch.utils.data.Dataset (the map-style base) and implements
    __len__ + __getitem__, so a DataLoader can batch it (Day 4).
    """

    def __init__(
        self,
        sequences: list[str],
        tokenizer: object,
        *,
        seed: int,
        mask_prob: float = 0.15,
        max_len: int = 128,
    ) -> None:
        """Store the data + config, precompute the residue-id pool (for the 10%
        random-replacement case), and set up a REPRODUCIBLE source of randomness.

        Contract: nothing random should touch the global RNG state; masking must
        be repeatable from `seed` alone.
        """
        # TODO(you): stash the inputs; create a seedable RNG you OWN.
        # Toolbox (bare functions — you decide how to use them):
        #   - random.Random ....... a standalone RNG object seeded from an int;
        #                           has .sample, .random, .randrange, .choice
        #   - (torch alt) torch.Generator().manual_seed(seed) if you prefer torch draws
        # SLIP: do NOT call the global random.* / torch.rand — that reseeds nothing
        #   and your tests will flake between runs.
        super().__init__()
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.residue_ids = [
            key for key in self.tokenizer.decode_map if key not in self.tokenizer.special_ids
        ]

    def __len__(self) -> int:
        """How many sequences are in the dataset."""
        # TODO(you): one number.
        return len(self.sequences)

    def __getitem__(self, i: int) -> dict:
        """Build one masked example for sequence i.

        Returns a dict: {"input_ids", "labels", "pad_mask"} (see module docstring).
        """
        # Roadmap:
        #   1. encode seq i -> ids; truncate to max_len-1; prepend [CLS] (<= max_len).
        #   2. maskable = positions 1.. (never [CLS]); choose ~15% with SEEDED self.rng.
        #   3. for each chosen pos: labels = ORIGINAL id; input via the 80/10/10 split
        #      (80% [MASK], 10% random residue, 10% unchanged). labels stay
        #      IGNORE_INDEX everywhere else.
        #   4. pad to max_len with [PAD]; build pad_mask (1 real / 0 pad).
        #   5. return {"input_ids", "labels", "pad_mask"} as torch tensors.
        # SLIP: use self.rng (never global random); labels get the ORIGINAL id for
        #   EVERY chosen pos (all three buckets); copy lists so input/labels/source
        #   never alias.
        ids = self.tokenizer.encode(self.sequences[i])
        cls_id = self.tokenizer.encode_map["[CLS]"]
        truncated_ids = ids[: self.max_len - 1]  # Truncated to max_len - 1 because [CLS] added
        tokens = [cls_id] + truncated_ids

        maskable = range(1, len(tokens))  # Skipped position 0 because [CLS] is there
        n_mask = round(len(maskable) * self.mask_prob)
        mask_positions = self.rng.sample(maskable, n_mask)

        mask_id = self.tokenizer.encode_map["[MASK]"]
        input_ids = tokens.copy()
        labels = [IGNORE_INDEX] * len(tokens)

        for pos in mask_positions:
            labels[pos] = tokens[pos]
            dice_roll = self.rng.random()
            if dice_roll < 0.8:
                input_ids[pos] = mask_id
            elif dice_roll < 0.9:
                input_ids[pos] = self.rng.choice(self.residue_ids)

        pad_id = self.tokenizer.encode_map["[PAD]"]
        pad_len = self.max_len - len(input_ids)
        pad_mask = ([1] * len(input_ids)) + ([0] * pad_len)
        input_ids.extend([pad_id] * pad_len)
        labels.extend([IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "pad_mask": torch.tensor(pad_mask),
        }
