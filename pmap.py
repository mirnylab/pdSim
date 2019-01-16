"""Parallel (multi-threaded) map function for python. 

Uses multiprocessing.Pool with error-resistant importing. There are two map
functions:

1) pmap(function, iterable) -> rapid fork-based multi-threaded map function.

2) low_memory_pmap(function, iterable) -> a more memory-efficient version
    intended for function calls that are individually long & memory-intensive.

"""
import os
from warnings import warn
from pickle import PicklingError
import progressbar 
from time import sleep 
import multiprocessing

def fake_pmap(*args, **kwargs):
    return list(map(*args, **kwargs))

CPUs = multiprocessing.cpu_count()
CHUNKS = 50*CPUs
def pmap(func, Iter, processes=CPUs-1):
    with multiprocessing.Pool(processes=processes) as P:
        return P.map(func, Iter)

def low_memory_pmap(func, Iter, processes=int(round(CPUs/2)), chunksize=1):
    with multiprocessing.Pool(processes=processes) as P:
        return [result for result in P.imap(func, Iter)]
        
def large_iter_pmap(func, Iter, processes=CPUs-1, status_bar=True, nice=10, wait_interval=1):
    os.nice(nice)
    chunksize = max(1, int(round(len(Iter)/CHUNKS)))
    try:
        with multiprocessing.Pool(processes=processes) as P:
            rs = P.map_async(func, Iter, chunksize=chunksize)
            maxval = rs._number_left
            bar = progressbar.ProgressBar(
                maxval=maxval, 
                widgets=[progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.ETA()])
            while not rs.ready():
                sleep(wait_interval)
                bar.update(maxval - rs._number_left)
            bar.finish()
            return rs.get()

    except PicklingError:
        warn("Lambda functions cannot be Pickled for Parallelization. Using single Process.", RuntimeWarning)
        return fake_pmap(func, Iter)
