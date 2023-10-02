
import multiprocessing
import time
import itertools
from pprint import pprint
import sys



def cube(x):

    averages = [2.6, 2.54, 1.99]

    print(x)

    for i, value in enumerate(x):
        if value:
            print(averages[i], end=' ')
        else:
            print("na", end=' ')
    print("\n")


if __name__ == "__main__":
    pool = multiprocessing.Pool(4)

    superset = list(itertools.product([0,1], repeat=9))
    superset = [list(x) for x in superset]

    print(len(superset))
    sys.exit()

    start_time = time.perf_counter()
    processes = [pool.apply_async(cube, args=(x,)) for x in superset]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")







#pprint(superset)
