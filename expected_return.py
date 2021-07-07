import networkx as nx
from collections import defaultdict
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite

N = 5 # Number of stocks
f = 3
sig_p = 0.3 * f # Expected return from n stocks (not average currently)
G = nx.Graph()
G.add_edges_from([(i, j) for i in range(N) for j in range(i + 1, N)])

# Create the covariance matrix and returns list
cov = pd.read_csv("cov_matrix.csv")
returns = pd.read_csv("mean_returns.csv")

# The matrix where we add the objective and the constraint
Q = defaultdict(int)

# Constraint specifying we should be as close to the return as possible

# Smaller Lagrange focusses more on optimisation, hence risk is lowered a lot, even if returns may be low
lagrange = 1

# Larger Lagrange focusses more on the constraint, hence we try to meet the expected returns, even if that may increase the risk
lagrange = 10

for i in range(N):
    Q[(i, i)] += (returns.iloc[i] * (-2 * sig_p + returns.iloc[i])) * lagrange
    for j in range(i + 1, N):
        Q[(i, j)] += 2 * lagrange * returns.iloc[i] * returns.iloc[j]

# Objective function
for i in range(N):
    Q[(i, i)] += 0.5 * cov.iloc[i, i]

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
# print(actual_return)
print()
print("Best Portfolio-")
print("Desired Returns: ", str(sig_p))
print("Actual Returns: ", actual_return)
print("Samples to be chosen: ", sampleset.first.sample)
vol = 0.0
for i in range(N):
    for j in range(N):
        vol += sampleset.first.sample[i] * sampleset.first.sample[j] * cov.iloc[i, j] / f / f

print("Volatility of Portfolio: ", vol)
print("Risk of Portfolio: ", vol ** 0.5)