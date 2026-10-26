"""Per-frame work on a thread pool.

The per-frame stages (frame labels, floor texture) spend their time in C++
calls that release the GIL and in image decoding, so threads run them in
parallel. Results come back in input order, which keeps every product the
same as a sequential run.
"""

from __future__ import annotations

import collections
import concurrent.futures
import os
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def workers(requested: int | None) -> int:
    """Threads to use: `requested`, or one per CPU for 0 / None."""
    if requested:
        return max(1, int(requested))
    return os.cpu_count() or 1


def ordered_map(fn: Callable[[T], R], items: Iterable[T], threads: int,
                window: int | None = None) -> Iterator[R]:
    """fn over items on `threads` threads, results in input order.

    At most `window` items (default 2 * threads) are in flight, so a slow
    consumer does not pile up results. One thread maps sequentially.
    """
    if threads <= 1:
        yield from map(fn, items)
        return
    window = window or 2 * threads
    source = iter(items)
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        pending: collections.deque = collections.deque()
        for item in source:
            pending.append(pool.submit(fn, item))
            if len(pending) >= window:
                break
        while pending:
            result = pending.popleft().result()
            item = next(source, _END)
            if item is not _END:
                pending.append(pool.submit(fn, item))
            yield result


_END = object()
