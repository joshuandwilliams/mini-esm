"""BLOSUM62 — the substitution-matrix reference (your Day-7 benchmark).

BLOSUM62 is a 20x20 log-odds substitution matrix learned from real evolutionary
alignments. Each entry scores how interchangeable two residues are:

    positive  -> the substitution is observed MORE often than chance
    negative  ->                          ... LESS often than chance
    diagonal  -> self vs self, always positive; rarer residues score higher

It is symmetric: score(a, b) == score(b, a). The values are integers on a
log-odds scale — evolutionary bookkeeping, not a physical energy.

On Day 7 you'll score MiniESM's learned residue relationships against this table.
For now it's the "does the biochemistry check out?" oracle behind your 20-AA
property table (see Obsidian "Amino Acid Alphabet").
"""

from __future__ import annotations

from functools import lru_cache

from Bio.Align import substitution_matrices


@lru_cache(maxsize=1)
def load_blosum62():
    """Load (and cache) the BLOSUM62 substitution matrix (a Biopython Array)."""
    return substitution_matrices.load("BLOSUM62")


def score(a: str, b: str) -> float:
    """Return the BLOSUM62 log-odds score for the residue pair (a, b).

    Args:
        a, b: one-letter residue codes. Order is irrelevant (the matrix is symmetric).
    Returns:
        the log-odds score as a float.
    """
    return float(load_blosum62()[a, b])


if __name__ == "__main__":
    # Predict each sign/magnitude from your property table FIRST, then run this
    # to check (from mini-esm/, inside the binder env):
    #   mamba run -n binder python -m reference.blosum
    checks = [
        ("L", "I", "both hydrophobic, similar size"),
        ("K", "D", "opposite charges (+ vs -)"),
        ("W", "G", "huge aromatic vs tiny"),
        ("W", "W", "self-match (diagonal)"),
        ("L", "L", "self-match, for comparison with W-W"),
    ]
    for a, b, note in checks:
        print(f"  {a}-{b}: {score(a, b):+.0f}   ({note})")
