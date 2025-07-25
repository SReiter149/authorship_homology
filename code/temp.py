import pstats
import random
from sparse_linear_algebra import Matrix
import cProfile

def test_sparse_package():
    # Build a 1000×1000 sparse matrix with 2000 random entries
    M = Matrix()
    for _ in range(5):
        i = random.randrange(5)
        j = random.randrange(5)
        M.add_nonzero_value(i, j, 1)
    M.convert()
    M.dim_ker_im()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    test_sparse_package()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(20)