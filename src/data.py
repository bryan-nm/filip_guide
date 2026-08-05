"""Data loading + group-aware splits for the live-encoding path.

Unlike mini-embed-filip, we do NOT precompute a packed per-token embedding cache:
random protein masking makes ESM-C's output mask-dependent, so the protein side
must be encoded live each step (see PLAN 5.4). This module therefore only loads
(uid, protein, caption) triples, builds fingerprinted group-aware splits, and
serves raw strings; tokenization + masking + encoding happen in the training
step. (Text is never masked and is a candidate for a one-time cache later; kept
live here for simplicity.)

Split helpers are ported verbatim from mini-embed-filip so behavior matches.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


csv.field_size_limit(10 * 1024 * 1024)   # captions can be hundreds of KB


@dataclass
class Pair:
    uid: str
    protein: str
    text: str
    pfam: Optional[str] = None


def load_pairs(
    csv_path: str, *,
    id_col: str = "primary_Accession",
    protein_col: str = "protein_sequence",
    text_col: str = "[final]text_caption",
    pfam_col: Optional[str] = "pfam_label",
    subset_size: int = 0,
) -> List[Pair]:
    """Load (uid, protein, text) triples from the SwissProt-full CSV."""
    pairs: List[Pair] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in (id_col, protein_col, text_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns {missing}. "
                f"Found: {reader.fieldnames}")
        for row in reader:
            pairs.append(Pair(uid=row[id_col], protein=row[protein_col],
                              text=row[text_col],
                              pfam=(row.get(pfam_col) if pfam_col and pfam_col in row else None)))
            if subset_size > 0 and len(pairs) >= subset_size:
                break
    return pairs


# ---------------------------------------------------------------------------
# Splits (ported from mini-embed-filip)
# ---------------------------------------------------------------------------
def dense_group_ids(keys: Sequence) -> np.ndarray:
    mapping: dict = {}
    out = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        gid = mapping.get(k)
        if gid is None:
            gid = len(mapping)
            mapping[k] = gid
        out[i] = gid
    return out


def group_ids_from_accessions(accessions: Sequence[str]) -> np.ndarray:
    return dense_group_ids(accessions)


def group_ids_from_texts(texts: Sequence[str]) -> np.ndarray:
    return dense_group_ids(texts)


def merged_split_group_ids(*id_arrays: Sequence[int]) -> np.ndarray:
    """Connected-component ids over rows linked by ANY shared key (union-find).

    Keeps every protein AND every identical caption wholly on one side of a split.
    """
    arrays = [np.asarray(a) for a in id_arrays]
    n = int(arrays[0].shape[0])
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for arr in arrays:
        assert arr.shape[0] == n, "all id arrays must have one entry per row"
        first: dict = {}
        for i in range(n):
            k = int(arr[i])
            j = first.get(k)
            if j is None:
                first[k] = i
            else:
                union(j, i)

    out = np.empty(n, dtype=np.int64)
    relabel: dict = {}
    for i in range(n):
        r = find(i)
        c = relabel.get(r)
        if c is None:
            c = len(relabel)
            relabel[r] = c
        out[i] = c
    return out


def make_splits(n: int, ratios: Sequence[float], seed: int,
                group_ids: Optional[Sequence[int]] = None) -> dict:
    """Train/val/test split over row indices [0, n), group-aware if ids given."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    if group_ids is None:
        perm = rng.permutation(n)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        train, val, test = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
        n_groups = int(n)
    else:
        g = np.asarray(group_ids)
        assert g.shape[0] == n
        uniq = np.unique(g)
        gperm = rng.permutation(uniq.shape[0])
        ng_train = int(round(ratios[0] * uniq.shape[0]))
        ng_val = int(round(ratios[1] * uniq.shape[0]))
        train_g = set(uniq[gperm[:ng_train]].tolist())
        val_g = set(uniq[gperm[ng_train:ng_train + ng_val]].tolist())
        train_rows, val_rows, test_rows = [], [], []
        for row, gid in enumerate(g.tolist()):
            (train_rows if gid in train_g else val_rows if gid in val_g else test_rows).append(row)
        train = rng.permutation(np.array(train_rows, dtype=np.int64))
        val = rng.permutation(np.array(val_rows, dtype=np.int64))
        test = rng.permutation(np.array(test_rows, dtype=np.int64))
        n_groups = int(uniq.shape[0])
    return {"n": int(n), "seed": int(seed), "ratios": list(ratios), "n_groups": n_groups,
            "train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def build_splits(pairs: List[Pair], ratios: Sequence[float], seed: int,
                 *, group_aware: bool = True) -> dict:
    """Splits keyed by shared-protein OR shared-caption components (no leakage)."""
    if group_aware:
        acc = group_ids_from_accessions([p.uid for p in pairs])
        txt = group_ids_from_texts([p.text for p in pairs])
        gids = merged_split_group_ids(acc, txt)
    else:
        gids = None
    return make_splits(len(pairs), ratios, seed, gids)


# ---------------------------------------------------------------------------
# Dataset + collate (raw strings; tokenization happens in the training step)
# ---------------------------------------------------------------------------
class RawPairsDataset(Dataset):
    """Serves (uid, protein, text) plus per-row group ids for false-neg masking."""

    def __init__(self, pairs: List[Pair], indices: Sequence[int]):
        self.pairs = pairs
        self.indices = list(indices)
        self.acc_ids = group_ids_from_accessions([p.uid for p in pairs])
        self.txt_ids = group_ids_from_texts([p.text for p in pairs])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        p = self.pairs[idx]
        return {"protein": p.protein, "text": p.text,
                "acc_id": int(self.acc_ids[idx]), "txt_id": int(self.txt_ids[idx])}


def raw_collate(batch):
    """Batch raw strings + group ids. Tokenization/masking done in the train step."""
    return {
        "proteins": [b["protein"] for b in batch],
        "texts": [b["text"] for b in batch],
        "acc_ids": torch.tensor([b["acc_id"] for b in batch], dtype=torch.long),
        "txt_ids": torch.tensor([b["txt_id"] for b in batch], dtype=torch.long),
    }
