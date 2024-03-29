import networkx as nx
from collections import defaultdict
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
import math

N = 5 # Number of stocks
precision_bits = 6 # For each stock, this is the precision of its weight
max_wt = 1.0 - 1.0 / pow(2, precision_bits)
dim = N * precision_bits # dim stands for matrix dimensions

f = 3 * max_wt # Fixed number of stocks that can be chosen
expected_return = 0.3
sig_p = expected_return * f # Expected return from n stocks (not average currently)

G = nx.Graph()
G.add_edges_from([(i, j) for i in range(dim) for j in range(i + 1, dim)])

# Create the covariance matrix and returns list
cov = pd.read_csv("cov_matrix.csv")
returns = pd.read_csv("mean_returns.csv")

# The matrix where we add the objective and the constraint
Q = defaultdict(int)

# Constraint1 minimises the difference between expected return and actual return
lagrange1 = 0.5
for d in range(dim): # N x precision
    i = d // precision_bits # The stock number
    p = d % precision_bits + 1 # The p^th of the bits we are using to represent the i^th stock
    ri = returns.iloc[i] # i^th stock returns
    Q[(d, d)] += (-2 * sig_p * ri / pow(2, p) + ri * ri / pow(2, 2 * p)) * lagrange1
    for d_dash in range(d + 1, dim):
        j = d_dash // precision_bits # The stock number
        q = d_dash % precision_bits + 1 # The q^th of the bits we are using to represent the j^th stock
        rj = returns.iloc[j] # j^th stock returns
        Q[(d, d_dash)] += 2 * ri * rj * lagrange1 / pow(2, p + q)


# Constraint2 specifying only f stocks should be used
lagrange2 = 0.1
for d in range(dim):
    i = d // precision_bits # The stock number
    p = d % precision_bits + 1 # The p^th of the bits we are using to represent the i^th stock
    Q[(d, d)] += (-2 * f / pow(2, p) + 1 / pow(2, 2 * p)) * lagrange2
    for d_dash in range(d + 1, dim):
        j = d_dash // precision_bits # The stock number
        q = d_dash % precision_bits + 1 # The q^th of the bits we are using to represent the j^th stock
        Q[(d, d_dash)] += 2 * lagrange2 / pow(2, p + q)

# The objective function
for d in range(dim):
    for d_dash in range(d, dim):
        i = d // precision_bits # The stock number
        p = d % precision_bits + 1 # The p^th of the bits we are using to represent the i^th stock
        j = d_dash // precision_bits # The stock number
        q = d_dash % precision_bits + 1 # The q^th of the bits we are using to represent the j^th stock
        covij = cov.iloc[i, j]
        if d == d_dash:
            # i == j and p == q
            Q[(d, d)] += 0.5 * covij / pow(2, 2 * p)
        elif i == j:
            # This means we are talking about the same stock, but different bits
            Q[(d, d_dash)] += covij / pow(2, p + q)
        else:
            # Different stocks
            Q[(d, d_dash)] += covij / pow(2, p + q)


sampler = EmbeddingComposite(DWaveSampler())
print("Response Sent")
sampleset = sampler.sample_qubo(Q, num_reads=10, chain_strength=1)
print("Response Received")

# Print the entire sampleset, that is, the entire table
print(sampleset)

# For the lowest energy, find the actual return
actual_return = 0.0

# distribution = sampleset.first.sample

first_few = 5
distributions = []

for sample, energy in sampleset.data(['sample', 'energy']):
    distributions.append(sample)

ctr = 0
for distribution in distributions:
    wts = [0 for i in range(N)]
    actual_return = 0.0
    for s_num in distribution.keys():
        if(distribution[s_num] == 1):
            i = s_num // precision_bits # Stock number
            p = s_num % precision_bits + 1 # Bit number
            wts[i] += 1 / pow(2, p)
            actual_return += returns.iloc[i] / pow(2, p)
    
    actual_return /= sum(wts)
    wts = [round(wts[i] / sum(wts), 2) for i in range(len(wts))]

    volatility = 0.0
    for i in range(N):
        for j in range(N):
            volatility += wts[i] * wts[j] * cov.iloc[i, j]

    print("\n\n Portfolio " + str(ctr))
    print("Weights: ", wts)
    print("Returns: ", actual_return)
    print("Risk: ", math.sqrt(volatility))
    ctr += 1


# print("Weights: ", wts)
# print("Returns: ", actual_return)
# print("Volatility: ", math.sqrt(volatility))

print("\n\nTime Spent in Quantum Computer: ",sampleset.info["timing"]["qpu_access_time"]/1000,"Milli Seconds")