"""Random generators derived from the protocol's seeds.

Every random choice in this project (the order of a recording round, a spread start position, the order of a block's
trials, the target of one trial) is drawn from a `numpy.random.Generator` built here from a seed written in the
protocol and the name of the choice. Naming the choice gives it a stream of its own, so one choice never depends on how
many numbers another one drew: the target of a block's twelfth trial is the same whether or not the planner was
restarted before it.

Only `Generator.random` is drawn from, here and in the modules that use these generators. It is the plainest draw
NumPy offers (uniform numbers straight from the bit generator), and a pinned test checks that the rig's NumPy and the
development NumPy produce the same numbers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

#: Separates the parts of a stream's name. It cannot occur in a label, so ("ab", "c") and ("a", "bc") differ.
_SEPARATOR = "\x1f"


def derived_rng(seed: int, *labels: str | int) -> np.random.Generator:
    """A random generator for one named choice, derived from a seed of the protocol.

    The same seed and labels always give the same stream; changing either gives an unrelated one.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"a seed is a whole number, got {seed!r}")
    parts = [str(seed)]
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, (str, int)):
            raise TypeError(f"a stream label is text or a whole number, got {label!r}")
        if isinstance(label, str) and _SEPARATOR in label:
            raise ValueError("a stream label cannot contain the separator character")
        # The type is part of the name, so that the text "2" and the number 2 name different streams.
        parts.append(("s" if isinstance(label, str) else "i") + str(label))
    digest = hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def shuffled[T](items: Sequence[T], rng: np.random.Generator) -> list[T]:
    """The items in a random order. The input is left as it is."""
    keys = rng.random(len(items)).tolist()
    order = sorted(range(len(items)), key=lambda index: (keys[index], index))
    return [items[index] for index in order]


def pick_index(count: int, rng: np.random.Generator) -> int:
    """A random index from 0 to `count - 1`, each equally likely."""
    if count <= 0:
        raise ValueError("there is nothing to pick from")
    return min(int(rng.random() * count), count - 1)
