# Improved Analysis of RANKING for Online Vertex-Weighted Bipartite Matching
This repository contains the code for the paper "Improved Analysis of RANKING for Online Vertex-Weighted Bipartite Matching". (https://arxiv.org/abs/2007.12823)

## Usage
Warning: With the values of the parameters below, the code may take a long time to run. You can replace them with smaller numbers to experiment.
1. Run the code in `lp.ipynb` to generate the file `g_values_50.txt`, which contains the values of g on a 50x50 discretized grid. 
2. Run the command `python fill_g_table_parallel.py 16834` to compute a 16384x16384 table `g_table_16384.txt` with the values of g on the finer discretized grid.
3. Run the command `python fill_ss_table_parallel.py 16834 1024` to compute a 1024x1024x1024 table `ss_table_1024.npy` with the values of the inner minimum. 
4. Run the command `python evaluate_cr.py 16384 1024 16384` to evaluate the bound in Theorem 1, using the above precomputed tables. 

## Upper bound
The file `LP Upper Bound.ipynb` contains the code to create and solve the LP described in Section 7 of the paper, which provides an upper bound on the best competitive ratio obtainable using our methods. 
