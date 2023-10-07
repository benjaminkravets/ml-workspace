import multiprocessing
import time

def worker(letter):
    print(letter)
    time.sleep(1)

def main():
    # Create a list of every lowercase letter
    letters = list(map(chr, range(97, 123)))

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=4)

    # Submit the letters to the pool
    for letter in letters:
        pool.apply_async(worker, (letter,))

    result = [p.get() for p in pool]

    # Wait for all tasks to complete
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
