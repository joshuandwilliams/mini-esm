"""Tokenizer: protein sequence string <-> list of integer ids.

To a model a protein is a string over a 20-letter alphabet plus a few special
tokens. This tokenizer holds that vocabulary and converts both directions:

    encode:  "ACD..."      -> [ids ...]
    decode:  [ids ...]     -> "ACD..."

Vocab = the 20 standard amino acids + three special tokens:
    [PAD]   filler so a batch of sequences can share one length
    [MASK]  a hidden position the model must predict (Day-5 masked-LM task)
    [CLS]   a summary slot at the start (future sequence-level readout)

See your Obsidian "Amino Acid Alphabet" note for the alphabet itself.

Contract to satisfy (your round-trip / known-answer test):
    decode(encode(seq)) == seq   for any all-residue string.
"""

from __future__ import annotations

# --- Given data (reference, not the exercise) -------------------------------
# The 20 standard amino acids, one-letter codes (see your property table).
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Special tokens. CONVENTION to honour: [PAD] must be id 0, so padding is
# literally "id 0" and never collides with a real residue.
SPECIAL_TOKENS = ("[PAD]", "[MASK]", "[CLS]")


class Tokenizer:
    """Hold a fixed vocabulary and convert sequences <-> integer ids.

    Attributes you'll expose (shape/So-what, not the how):
        a char->id lookup and its inverse id->char, covering all 23 tokens
        (3 specials + 20 residues), with [PAD] pinned to id 0.
    """

    def __init__(self) -> None:
        """Build the two lookups (char->id and id->char) from ONE ordered vocab.

        Contract: every residue and every special token gets a unique, stable id;
        [PAD] is id 0; the two lookups are exact inverses of each other.
        """
        # TODO(you): build ONE ordered vocab, then TWO lookups from it.
        # Toolbox (what each does — you decide how they fit together):
        #   - list / tuple concatenation .... to lay the vocab out in a fixed order
        #   - enumerate .................... walks an iterable yielding (index, item)
        #   - dict, or a dict comprehension  a key->value map (and you'll want the inverse too)
        # SLIP: which group goes FIRST decides who gets id 0 -> [PAD] must be id 0.
        # SLIP: derive the inverse FROM the forward map (one source of truth) so
        #       the two can never drift out of sync.
        combined_list = list(SPECIAL_TOKENS) + list(AMINO_ACIDS)
        self.decode_map = dict(enumerate(combined_list))
        self.encode_map = dict((v, k) for k, v in self.decode_map.items())
        self.special_ids = {
            self.encode_map["[PAD]"],
            self.encode_map["[MASK]"],
            self.encode_map["[CLS]"],
        }

    def encode(self, seq: str) -> list[int]:
        """Map a sequence string to its list of integer ids.

        The corpus is pre-filtered to the canonical 20 residues by
        scripts/download_swissprot.py, so a character outside the vocab is a
        genuine error, not something to absorb. DECISION: fail loud — let the
        missing-key lookup raise rather than mapping to an [UNK] token, so a bad
        char surfaces immediately instead of silently polluting training.

        Args:
            seq: a protein sequence over the 20 canonical residues, e.g. "ACDEF".
        Returns:
            the ids for each character, in order.
        Raises:
            KeyError: if seq contains a character outside the 20-residue vocab.
        """
        # TODO(you):
        # Toolbox: a Python string is iterable (per-character access); building a list.
        # DECISION MADE: fail loud on unknown chars (no [UNK]). A bare vocab lookup
        #   already raises KeyError on a miss — which is exactly the behaviour we want.
        ids = []
        for char in seq:
            if char not in self.encode_map:
                raise KeyError(f"Unknown Character: {char}")
            ids.append(self.encode_map[char])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Map a list of integer ids back to a sequence string.

        Args:
            ids: integer ids, each a valid index in the vocab.
            skip_special_tokens: if True (default), drop [PAD]/[MASK]/[CLS] to
                give a clean residue string; if False, render them explicitly
                (e.g. "[MASK]") for a debug view of masked examples.
        Returns:
            the reconstructed sequence string. With skip_special_tokens=True it is
            a pure residue string and satisfies decode(encode(seq)) == seq for any
            all-residue input. With False it may be a mixed string (single-char
            residues plus multi-char special tokens) — a debug view, not
            round-trippable.
        """
        # TODO(you):
        # Toolbox: your id->char lookup; a way to turn a list of single chars into
        #   one string (look at str.join).
        # GOTCHA: decide how special tokens are rendered on the way out (drop them?
        #   keep "[MASK]" visible?). Document it — and make sure your choice keeps
        #   decode(encode(seq)) == seq for an all-residue seq.
        seq = ""
        for token in ids:
            if skip_special_tokens and token in self.special_ids:
                continue
            seq += self.decode_map[token]
        return seq


if __name__ == "__main__":
    # Smoke harness once you've implemented the above (from mini-esm/, in the env):
    #   mamba run -n binder python -m miniesm.tokenizer
    # (Round-trip a sample and eyeball it — this is a harness, not the test.)
    tok = Tokenizer()
    sample = "ACDEFGHIKLMNPQRSTVWY"
    print("encode:", tok.encode(sample))
    print("round-trip ok:", tok.decode(tok.encode(sample)) == sample)
