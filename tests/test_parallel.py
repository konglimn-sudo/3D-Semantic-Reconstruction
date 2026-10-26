import threading
import time

from garage_recon import parallel


def test_ordered_map_keeps_the_input_order():
    def slow_square(x):
        time.sleep(0.01 * (5 - x % 5))  # Later items finish first.
        return x * x

    items = list(range(23))
    assert list(parallel.ordered_map(slow_square, items, 4)) == [
        x * x for x in items]
    assert list(parallel.ordered_map(slow_square, items, 1)) == [
        x * x for x in items]


def test_ordered_map_bounds_the_work_in_flight():
    running, peak = [0], [0]
    lock = threading.Lock()

    def work(x):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.005)
        with lock:
            running[0] -= 1
        return x

    out = parallel.ordered_map(work, range(40), threads=8, window=3)
    assert list(out) == list(range(40))
    assert peak[0] <= 3
    assert parallel.workers(0) >= 1 and parallel.workers(3) == 3
