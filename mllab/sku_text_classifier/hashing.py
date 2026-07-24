"""Deterministic feature hashing for the text classifier (pure Python).

Pipeline: regex sub-token split of alphanumerics (``M8x40`` -> ``m8``, ``x40``,
``8``, ``40``) with word-boundary markers, then hashed char n-grams (3-5).
No torch here so the tokenizer stays trivially testable.
"""

from __future__ import annotations

import re

_ALNUM = re.compile(r"[a-z]+\d+|\d+|[a-z]+")
_DIGIT = re.compile(r"\d+")


def subtoken_split(text: str) -> list[str]:
    """Split into sub-tokens: letter/digit runs plus standalone digit runs."""
    text = text.lower()
    toks: list[str] = []
    for word in text.split():
        parts = _ALNUM.findall(word)
        toks.extend(parts)
        extra = _DIGIT.findall(word)
        # add standalone digits not already captured as their own token
        for e in extra:
            if e not in parts:
                toks.append(e)
    return toks


def char_ngrams(token: str, nmin: int = 3, nmax: int = 5) -> list[str]:
    """Char n-grams of a boundary-marked token, e.g. ``<m8>`` -> ``<m8``, ``m8>``."""
    marked = f"<{token}>"
    grams: list[str] = []
    for n in range(nmin, nmax + 1):
        if len(marked) < n:
            continue
        for i in range(len(marked) - n + 1):
            grams.append(marked[i : i + n])
    return grams


def _fnv1a(s: str) -> int:
    """Deterministic 64-bit FNV-1a hash (stable across processes)."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def hash_document(text: str, n_buckets: int = 2**18) -> list[int]:
    """Return the list of hashed bucket ids for all char n-grams in ``text``."""
    ids: list[int] = []
    for tok in subtoken_split(text):
        for gram in char_ngrams(tok):
            ids.append(_fnv1a(gram) % n_buckets)
    if not ids:  # guarantee at least one feature
        ids.append(0)
    return ids
