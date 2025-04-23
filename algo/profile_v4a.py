import tracemalloc
import time
import psutil
import os
from epm_bigraph_v4 import EPMBigraphEnumerator
from math import comb


def count_symmetry_reduced_candidates(n_q: int, n_a: int) -> int:
    R = n_q + n_a
    M_q_types = comb(R, 2)
    count_q = comb(M_q_types + n_q - 1, n_q)
    M_a_types = 2**R - R - 1
    count_a = comb(M_a_types + n_a - 1, n_a) if n_a > 0 else 1
    return count_q * count_a


def profile_epm(n_q, n_a, cur=None, total=None):
    print(f"Profiling #{cur}/{total} (n_q={n_q}, n_a={n_a})")

    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process(os.getpid())

    # Time it
    t0 = time.perf_counter()
    enumerator = EPMBigraphEnumerator()
    graphs = enumerator.enumerate_structural(n_q, n_a)
    t1 = time.perf_counter()

    # Get peak memory used (tracemalloc only tracks allocations in Python)
    current, peak = tracemalloc.get_traced_memory()
    peak_mem_mb = peak / 1024 / 1024
    tracemalloc.stop()

    print(f"  → Time elapsed:       {t1 - t0:.3f} sec")
    print(f"  → Peak memory:        {peak_mem_mb:.2f} MB")
    print(f"  → Eval ratio:         {len(graphs)/enumerator.num_epm_bigraph_enumerated:.3g}")
    print(f"  → EPM graphs eval:    {enumerator.num_epm_bigraph_enumerated}")
    print(f"  → Canonical graphs:   {enumerator.num_epm_bigraph_canonical}")
    print(f"  → Nontrivial graphs:  {len(graphs)}")
    print("")

if __name__ == "__main__":
    count_threshold = 10**8
    pairs_with_counts = []
    for n_q in range(2, 9):
        for n_a in range(0, 6):
            count = count_symmetry_reduced_candidates(n_q, n_a)
            if count < count_threshold:
                pairs_with_counts.append(((n_q, n_a), count))

    total = len(pairs_with_counts)
    pairs_with_counts.sort(key=lambda x: x[1])
    for (i, ((n_q, n_a), count)) in enumerate(pairs_with_counts):
        profile_epm(n_q, n_a, i, total)
        