import wolfhaley
import multiprocessing
import time
import itertools
from pprint import pprint



def cube(x):
    l = 0
    print(x)
    #for i in range(x ** 2):
    #    l += x
    return(l)

if __name__ == "__main__":
    pool = multiprocessing.Pool(4)

    superset = list(itertools.product([0,1], repeat=20))

    start_time = time.perf_counter()
    processes = [pool.apply_async(cube, args=(x,)) for x in superset]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")







#pprint(superset)
