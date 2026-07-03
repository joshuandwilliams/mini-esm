# data/

Local, **git-ignored** home for downloaded training corpora. Nothing here is
committed (see `.gitignore` in this folder) — so CI never sees it, and the repo
stays small.

## Populate it

From the repo root (`mini-esm/`), inside the binder env:

```bash
mamba run -n binder python scripts/download_swissprot.py
```

This writes **`swissprot_week1.txt`** — one sequence per line: ~10k reviewed
(Swiss-Prot) sequences ≤128 residues, filtered to the canonical 20-letter
alphabet (sequences containing `X/U/B/Z/O/*` are dropped), sampled with a fixed
seed for reproducibility. Re-running with the same `--seed` reproduces the exact
same set.

The tokenizer relies on this filtering: `miniesm/tokenizer.py` `encode()` fails
loud on any non-canonical character, treating it as a real error rather than
mapping it to an `[UNK]` token.
