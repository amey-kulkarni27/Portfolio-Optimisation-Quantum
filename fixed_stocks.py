import networkx as nx
from collections import defaultdict
import pandas as pd
# from dwave.system import DWaveSampler, EmbeddingComposite

N = 3 # Number of stocks
f = 2 # Fixed number of stocks that can be chosen
G = nx.Graph()
G.add_edges_from([(i, j) for i in range(N) for j in range(i + 1, N)])

# Create the covariance matrix and returns list
cov = pd.read_csv("cov_matrix.csv")
returns = pd.read_csv("mean_returns.csv")

# The matrix where we add the objective and the constraint
Q = defaultdict(int)

# Constraint specifying only
lagrange = 1 # Some temporary value
for i in range(N):
    Q[(i, i)] += -(2 * f - 1) * lagrange
    for j in range(i + 1, N):
        Q[(i, j)] += 2 * lagrange

# Objective function
for i in range(N):
    Q[(i, i)] += cov.iloc[i, i]

for i, j in G.edges:
    Q[(i, j)] += cov.iloc([i, j])