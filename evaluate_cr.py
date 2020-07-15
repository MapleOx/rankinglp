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


def worker_func(i):
    counter = var_dict['counter']
    g_n = var_dict['g_n'] - 1
    ss_n = var_dict['ss_n'] - 1
    cr_n = var_dict['cr_n']
    g_table_np = np.frombuffer(var_dict['g_table']).reshape(var_dict['g_shape'])
    table = np.frombuffer(var_dict['ss_table']).reshape(var_dict['ss_shape'])

    cr_row = np.zeros(cr_n+1)

    gam = i/cr_n
    ssi = int(gam*ss_n)
    for j in range(cr_n+1):
        tau = j/cr_n
        ssj = int(tau*ss_n)
        first_integral = (1/cr_n)*np.sum([g(l/cr_n, j/cr_n, g_table_np) for l in range(i)])
        # trapezoidal sum       
        cr_row[j] = (1-i/cr_n)*(1-j/cr_n) + (1-j/cr_n)*first_integral + (1/ss_n)*(np.sum([table[ssi, ssj, y] for y in range(1, ssj)]) + 0.5*(table[ssi, ssj, 0] + table[ssi, ssj, ssj]))

    counter.increment()
    v = counter.value()
    if v % 5 == 1:
        starttime = var_dict['starttime']
        currenttime = int(time.time())
        print("Filled %d out of %d rows of cr table. Elapsed time: %d minutes, %d seconds."%(v-1, cr_n, (currenttime - starttime)//60, (currenttime - starttime)%60))
    
    return cr_row 

def init_worker(g_table, ss_table, g_shape, ss_shape, counter, g_n, ss_n, cr_n, starttime):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['g_table'] = g_table
    var_dict['ss_table'] = ss_table
    var_dict['g_shape'] = g_shape
    var_dict['ss_shape'] = ss_shape
    var_dict['counter'] = counter
    var_dict['g_n'] = g_n
    var_dict['ss_n'] = ss_n
    var_dict['cr_n'] = cr_n
    var_dict['starttime'] = starttime 

def evaluate_crs(cr_n, g_table = None, ss_table = None):
    # calculates the bound in Lemma 4.1
    # discretizes in parts of 1/n when calculating integrals
    # checks every 1/n when evaluating the inner min

    ss_n = ss_table.shape[0]
    g_n = g_table.shape[0]

    counter = Counter(0)

    g_table_raw = RawArray('d', g_table.shape[0] * g_table.shape[1])
    g_shape = g_table.shape
    # Wrap X as an numpy array so we can easily manipulate its data.
    g_np = np.frombuffer(g_table_raw).reshape(g_table.shape)
    # Copy data to our shared array.
    np.copyto(g_np, g_table)

    ss_table_raw = RawArray('d', ss_table.shape[0] * ss_table.shape[1] * ss_table.shape[2])
    ss_shape = ss_table.shape
    # Wrap X as an numpy array so we can easily manipulate its data.
    ss_np = np.frombuffer(ss_table_raw).reshape(ss_table.shape)
    # Copy data to our shared array.
    np.copyto(ss_np, ss_table)

    starttime = int(time.time())
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(g_table_raw, ss_table_raw, g_shape, ss_shape, counter, g_n, ss_n, cr_n, starttime)) as pool:
        result = pool.map(worker_func, range(0, cr_n+1))
        crs = np.stack(result, axis=0)
        return crs


def main():
    g_table_n = None
    ss_table_n = None
    cr_n = None
    if len(sys.argv) < 4:
        print("Usage: python fill_ss_table_parallel.py g_table_n ss_table_n cr_n")
        g_table_n = 16384
        ss_table_n = 1024
        cr_n = 4096
        print("One or both of g_table_n and ss_table_n not specified. \
            Defaulting to g_table_n = %d and ss_table_n = %d"%(g_table_n, ss_table_n)) 
    else:
        g_table_n = int(sys.argv[1])
        ss_table_n = int(sys.argv[2])
        cr_n = int(sys.argv[3])
        print("Received command line input g_table_n = %d, ss_table_n = %d, cr_n = %d."%(g_table_n, ss_table_n, cr_n))

    n = 50

    # Load g table if file exists. 
    g_table_filename = 'g_table_{0}.txt'.format(g_table_n)
    g_table = None
    if path.exists(g_table_filename):
        g_table = np.loadtxt(g_table_filename)
        print("Loaded g table from file %s."%g_table_filename)
    else:
        print("%s not found in current directory. Please create the g table before running this code."%g_table_filename)
        return 

    # load ss table
    ss_table = None
    ss_table_filename = 'ss_table_{0}.npy'.format(ss_table_n)
    if path.exists(ss_table_filename):
        print("Starting to load %s."%ss_table_filename)
        ss_table = np.load(ss_table_filename)
        print("Finished loading %s."%ss_table_filename)
    else:
        print("%s not found in directory. Aborting."%ss_table_filename)
        return 

    crs_table_filename = 'crs_table_{0}.txt'.format(cr_n)
    if path.exists(crs_table_filename):
        print("cr file %s already exists. Aborting."%crs_table_filename)
        return 
    else:
        crs = evaluate_crs(cr_n, g_table, ss_table)
        np.savetxt(crs_table_filename, crs)
        print("Saved crs to %s."%crs_table_filename)
        print("Minimum cr: %1.5f"%np.min(crs))


if __name__ == '__main__':
    multiprocessing.freeze_support()

    main()

