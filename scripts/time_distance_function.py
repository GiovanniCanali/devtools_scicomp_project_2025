import matplotlib.pyplot as plt
import numpy as np
import time
from pyclassify.utils import distance_numpy, distance_numba

sizes = [10**p for p in range(0, 9)]
times_numpy = []
times_numba = []

print("Running benchmarks...")

for size in sizes:
    x = np.random.rand(size)
    y = np.random.rand(size)

    # Time f1
    start = time.perf_counter()
    d = distance_numpy(x, y)
    end = time.perf_counter()
    times_numpy.append(end - start)

    # Time f2
    start = time.perf_counter()
    d = distance_numba(x, y)
    end = time.perf_counter()
    times_numba.append(end - start)

    print(f"Size {size:.0e} done.")

plt.figure(figsize=(12, 6))
plt.plot(sizes, times_numpy, marker='o', label='numpy', color='skyblue')
plt.plot(sizes, times_numba, marker='o', label='numba', color='magenta')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title('Timing Comparison (log-log scale)')
plt.legend()
plt.tight_layout()
plt.savefig('logs/scalability.png')
