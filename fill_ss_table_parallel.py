#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import time
import itertools
import multiprocessing
import multiprocessing.managers
import os
import sys
from os import path
from functools import partial
from collections import deque 
from multiprocessing import Value, Lock, Pool, RawArray

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

def g(x, y, g_table):
    table_n = g_table.shape[0] - 1
    i = int(x*table_n)
    j = int(y*table_n)
    return g_table[i, j]

def populate_ss_parallel(g_table, ss_n, start_idx, end_idx):

    counter = Counter(0)

    X = RawArray('d', g_table.shape[0] * g_table.shape[1])
    X_shape = g_table.shape
    # Wrap X as an numpy array so we can easily manipulate its data.
    X_np = np.frombuffer(X).reshape(g_table.shape)
    # Copy data to our shared array.
    np.copyto(X_np, g_table)

    starttime = int(time.time())
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(X, X_shape, counter, ss_n, starttime)) as pool:
        result = pool.map(worker_func, range(start_idx, end_idx))
        ss_table = np.stack(result, axis=0)
        return ss_table

def worker_func(i):
    counter = var_dict['counter']
    ss_n = var_dict['ss_n']
    g_table_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])

    ss_row = np.zeros((ss_n+1, ss_n+1))
    for j in range(ss_n+1): # tau 
        for k in range(j+1): # y
            y = k/ss_n 
            tau = j/ss_n
            tau_ceil = min((j+1)/ss_n, 1)
            s = [0]*min((i+2), ss_n+1)
            s[0] = 1 - g(0, y, g_table_np) + (1/ss_n)*np.sum([g(d/ss_n, tau_ceil, g_table_np) for d in range(i)])
            for l in range(1, min(i+2, ss_n+1)): # theta
                s[l] = s[l-1] + g((l-1)/ss_n, y, g_table_np) - g(l/ss_n, y, g_table_np)  + (1/ss_n)*g(l/ss_n, y, g_table_np) - (1/ss_n)*g(l/ss_n, tau_ceil, g_table_np)
            ss_row[j,k] = min(s)
    counter.increment()
    v = counter.value()
    if v % 5 == 1:
        starttime = var_dict['starttime']
        currenttime = int(time.time())
        print("Filled %d out of %d rows of ss_table. Elapsed time: %d minutes, %d seconds."%(v-1, ss_n, (currenttime - starttime)//60, (currenttime - starttime)%60))
    return ss_row 

def init_worker(X, X_shape, counter, ss_n, starttime):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape
    var_dict['counter'] = counter
    var_dict['ss_n'] = ss_n
    var_dict['starttime'] = starttime 

def main():
    MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

    g_table_n = None
    ss_table_n = None
    start_idx = None
    end_idx = None
    if len(sys.argv) < 5:
        print("Usage: python fill_ss_table_parallel.py g_table_n ss_table_n start_idx end_idx")
        g_table_n = 4096
        ss_table_n = 512
        print("One or both of g_table_n and ss_table_n not specified. \
            Defaulting to g_table_n = %d and ss_table_n = %d"%(g_table_n, ss_table_n)) 
    else:
        g_table_n = int(sys.argv[1])
        ss_table_n = int(sys.argv[2])
        start_idx = int(sys.argv[3])
        end_idx = int(sys.argv[4])
        print("Received command line input g_table_n = %d, ss_table_n = %d, start_idx = %d, end_idx = %d."%(g_table_n, ss_table_n, start_idx, end_idx))

    n = 50
    g_values = np.loadtxt('g_values_{0}.txt'.format(n))

    # Load g table if file exists. 
    g_table_filename = 'g_table_{0}.txt'.format(g_table_n)
    g_table = None
    if path.exists(g_table_filename):
        g_table = np.loadtxt(g_table_filename)
        print("Loaded g table from file %s."%g_table_filename)
    else:
        print("%s not found in current directory. Please create the g table before running this code."%g_table_filename)
        return 

    # Load ss table if file exists, create it otherwise. 
    ss_table_filename = 'ss_table_{0}_{1}_{2}.npy'.format(ss_table_n, start_idx, end_idx)
    if path.exists(ss_table_filename):
        print("ss table file %s already exists. Aborting."%ss_table_filename)
        return 
    else:
        print("%s not found in directory. Proceeding to create ss table."%ss_table_filename)
        t0 = time.time()
        ss_table = populate_ss_parallel(g_table, ss_table_n, start_idx, end_idx)
        t1 = time.time()
        print("Finished filling ss table. Time taken: %d minutes, %d seconds."%((t1 - t0)//60, (t1 - t0)%60))
        np.save(ss_table_filename, ss_table)
        print("Saved ss table to %s"%ss_table_filename)


if __name__ == '__main__':
    multiprocessing.freeze_support()

    main()

    
