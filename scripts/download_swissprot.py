"""Download the Week-1 training corpus: short reviewed (Swiss-Prot) sequences.

Plumbing — NOT part of the learning objective and NOT run by CI (tests use tiny
in-code sequences instead). It:

  1. streams reviewed UniProt sequences <= max-len residues (server-side filter),
  2. keeps only sequences over the canonical 20-letter alphabet
     (drops any containing X/U/B/Z/O/* — see tokenizer's fail-loud contract),
  3. takes a SEEDED random sample of `n` sequences (reproducible),
  4. writes one sequence per line to data/ (git-ignored).

Uses only the standard library (urllib) so it adds no dependency.

Run from the repo root (mini-esm/), inside the binder env:

    mamba run -n binder python scripts/download_swissprot.py

Options:

    mamba run -n binder python scripts/download_swissprot.py \
        --n 10000 --max-len 128 --seed 0 --out data/swissprot_week1.txt
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

# The 20 standard amino acids — the alphabet the tokenizer accepts.
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "swissprot_week1.txt"


def build_url(max_len: int) -> str:
    """UniProt REST 'stream' query for reviewed sequences up to max_len residues."""
    query = f"(reviewed:true) AND (length:[1 TO {max_len}])"
    params = {"query": query, "format": "fasta", "compressed": "false"}
    return f"{UNIPROT_STREAM}?{urllib.parse.urlencode(params)}"


def stream_fasta(url: str) -> Iterator[str]:
    """Yield sequence strings (headers discarded) from a streamed FASTA response."""
    req = urllib.request.Request(url, headers={"User-Agent": "mini-esm/0.0.1 (learning project)"})
    seq_parts: list[str] = []
    started = False
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith(">"):
                if started:
                    yield "".join(seq_parts)
                started = True
                seq_parts = []
            else:
                seq_parts.append(line)
    if started:
        yield "".join(seq_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000, help="number of sequences to sample")
    parser.add_argument("--max-len", type=int, default=128, help="max residues per sequence")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the sample")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output file")
    args = parser.parse_args()

    url = build_url(args.max_len)
    print(f"[download] streaming reviewed sequences <= {args.max_len} aa ...", file=sys.stderr)

    kept: list[str] = []
    n_total = 0
    n_dropped = 0
    for seq in stream_fasta(url):
        n_total += 1
        seq = seq.upper()
        if seq and set(seq) <= CANONICAL:
            kept.append(seq)
        else:
            n_dropped += 1

    # Log what we dropped — never silently truncate the corpus.
    print(
        f"[download] streamed {n_total}; kept {len(kept)} canonical, "
        f"dropped {n_dropped} with non-standard chars",
        file=sys.stderr,
    )

    rng = random.Random(args.seed)
    if len(kept) >= args.n:
        sample = rng.sample(kept, args.n)
    else:
        print(
            f"[download] WARNING: only {len(kept)} canonical sequences available "
            f"(< requested {args.n}); using all of them",
            file=sys.stderr,
        )
        sample = kept
        rng.shuffle(sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.writelines(s + "\n" for s in sample)

    print(f"[download] wrote {len(sample)} sequences to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
