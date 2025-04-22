import tracemalloc
import time
import psutil
import os
from epm_bigraph_v3 import EPMBigraphEnumerator

def profile_epm(n_q, n_a):
    print(f"Profiling (n_q={n_q}, n_a={n_a})")

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

    print(f"  → Time elapsed:     {t1 - t0:.3f} sec")
    print(f"  → Peak memory:      {peak_mem_mb:.2f} MB")
    print(f"  → Unique graphs:    {len(graphs)}")
    print("")

if __name__ == "__main__":
    # Example usage
    profile_epm(2, 1)
    profile_epm(3, 1)
    profile_epm(4, 1)
    profile_epm(3, 2)
    profile_epm(2, 2)
    # profile_epm(2, 0)
    # profile_epm(3, 0)
    # profile_epm(4, 0)
    # profile_epm(5, 0)
    # profile_epm(5, 2)   # uncomment only if you're confident in memory
