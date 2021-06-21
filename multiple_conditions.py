import networkx as nx
from collections import defaultdict
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite

N = 3 # Number of stocks
sig_p = 0.9 # Expected return from n stocks (not average currently)
f = 3 # Fixed number of stocks that can be chosen
G = nx.Graph()
G.add_edges_from([(i, j) for i in range(N) for j in range(i + 1, N)])

# Create the covariance matrix and returns list
cov = pd.read_csv("cov_matrix.csv")
returns = pd.read_csv("mean_returns.csv")

# The matrix where we add the objective and the constraint
Q = defaultdict(int)

'''
I tried a few experiments with this
1) If I increase lagrange1 much more than lagrange2, 3rd stock is preferred over 2nd which is in turn preferred over 1st, for a high expected return (0.6). This is because this is the order of the mean returns.
2) If I increase lagrange2 more than lagrange1, solutions obtained tend to satisfy "number of stocks used = f".

Some combination of the two can be useful.
'''

# Constraint1 minimises the difference between expected return and actual return
lagrange1 = 1

for i in range(N):
    Q[(i, i)] += (returns.iloc[i] * (-2 * sig_p + returns.iloc[i])) * lagrange1
    for j in range(i + 1, N):
        Q[(i, j)] += 2 * lagrange1 * returns.iloc[i] * returns.iloc[j]

# Constraint2 specifying only f stocks should be used
lagrange2 = 2
for i in range(N):
    Q[(i, i)] += -(2 * f - 1) * lagrange2
    for j in range(i + 1, N):
        Q[(i, j)] += 2 * lagrange2

# Objective function
for i in range(N):
    Q[(i, i)] += cov.iloc[i, i]

for i, j in G.edges:
    Q[(i, j)] += cov.iloc[i, j]

sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample_qubo(Q, num_reads=10, chain_strength=1)

# Print the entire sampleset, that is, the entire table
print(sampleset)

# For the lowest energy, find the actual return
actual_return = 0.0
distribution = sampleset.first.sample
for s_num in distribution.keys():
    if(distribution[s_num] == 1):
        actual_return += returns.iloc[s_num]
print(actual_return)
