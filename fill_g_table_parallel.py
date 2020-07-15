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
from multiprocessing import Value, Lock, RawArray, Pool

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

# Given the discretized solution g_vars returned by the LP
# Interpolate by triangulating to calculate the function value at any point (x, y) in the unit square
def g_func(g_vars, n, x, y):
    assert 0 <= x <= 1 and 0 <= y <= 1
    
    a = [i/n for i in range(n+1)]
    
    def robust_floor(x, tol):
        # returns floor(x), but robust to numerical errors
        # tol is desired precision
        # e.g if x = 0.99999999997, then robust_floor(x, 1e-9) returns 0 whereas int(x) returns 0
        x_round = np.rint(x)
        if np.abs(x_round - x) < tol:
            return int(x_round)
        else:
            return int(np.floor(x))
    tol = 1e-9
    # find i s.t. i/n <= x < (i+1)/n
    i = robust_floor(n*x, tol)
    
    # find j s.t. j/n <= y < (j+1)/n
    j = robust_floor(n*y, tol)    
    
    # corner cases
    if x == 1 and y == 1:
        return g_vars[n, n]
    elif x == 1:
        mu = j+1-n*y
        return mu*g_vars[n, j] + (1-mu)*g_vars[n, j+1]
    elif y == 1:
        lbda = i+1-n*x
        return lbda*g_vars[i, n] + (1-lbda)*g_vars[i+1, n]
    
    # interpolate by triangulating 
    # triangulate in the way that makes the hypotenuses go from top left to bottom right
    if x + y < (i+j+1)/n:
        # (x, y) is in the "lower" triangle
        lbda = i + 1 - n*x
        b = lbda*g_vars[i, j] + (1-lbda)*g_vars[i+1, j] # foot of perpendicular from (x,y) to the base of the square
        t = lbda*g_vars[i, j+1] + (1-lbda)*g_vars[i+1, j] # intersection of vertical line from (x,y) up to hypotenuse of triangle
        mu = (i+j+1-n*x-n*y)/(i+1-n*x)
        return mu*b + (1-mu)*t
    else:
        # (x, y) is in the "upper" triangle
        lbda = i + 1 - n*x
        b = lbda*g_vars[i, j+1] + (1-lbda)*g_vars[i+1, j+1] # intersection of perpendicular from (x,y) with the top edge of the square
        t = lbda*g_vars[i,j+1] + (1-lbda)*g_vars[i+1, j] # intersection of vertical line from (x,y) down to hypotenuse of triangle
        mu = (i+j+1-n*x-n*y)/(i-n*x)
        return mu*b + (1-mu)*t
    
def worker_func(i):
    # fills row i of g_table
    n = var_dict['X_shape'][0]-1
    g_table_n = var_dict['g_table_n']
    g_values_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape']) 
    counter = var_dict['counter']
    result = np.array([g_func(g_values_np, n, i/g_table_n, j/g_table_n) for j in range(0, g_table_n+1)])
    counter.increment()
    v = counter.value()
    if v % 10 == 0:
        print("Filled %d out of %d rows of g table."%(v, g_table_n))
    return result

    
def populate_g_parallel(g_values, gtable_n):

    counter = Counter(0)

    X = RawArray('d', g_values.shape[0] * g_values.shape[1])
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(g_values.shape)
    # Copy data to our shared array.
    np.copyto(X_np, g_values)

    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(X, g_values.shape, counter, g_table_n)) as pool:
        result = pool.map(worker_func, range(g_table_n+1))
        g_table = np.vstack(result)
        return g_table

def init_worker(X, X_shape, counter, g_table_n):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape
    var_dict['counter'] = counter
    var_dict['g_table_n'] = g_table_n


# We need this check for Windows to prevent infinitely spawning new child
# processes.
if __name__ == '__main__':
    multiprocessing.freeze_support()

    g_table_n = None
    if len(sys.argv) >= 2:
        g_table_n = int(sys.argv[1])
        print("Received command line input g_table_n = %d."%g_table_n)
    else:
        g_table_n = 128
        print("Usage: python fill_g_table_parallel.py g_table_n.")
        print("No command line input received for g_table_n. Defaulting to g_table_n = %d."%g_table_n)

    n = 50
    g_values = np.loadtxt('g_values_{0}.txt'.format(n))

    # Load g table if file exists, create it otherwise. 
    g_table_filename = 'g_table_{0}.txt'.format(g_table_n)
    g_table = None
    if path.exists(g_table_filename):
        print("g table file %s already exists. Aborting."%g_table_filename)
    else:
        print("%s not found in directory. Proceeding to create g table."%g_table_filename)
        t0 = time.time()
        g_table = populate_g_parallel(g_values, g_table_n)
        t1 = time.time()
        print("Finished filling g table. Time taken: %d minutes, %d seconds."%((t1 - t0)//60, (t1 - t0)%60))
        np.savetxt(g_table_filename, g_table)
        print("g table saved as %s."%g_table_filename)


