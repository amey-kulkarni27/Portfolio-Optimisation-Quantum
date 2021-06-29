import networkx as nx
from collections import defaultdict
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
import math


df = pd.read_csv("8yrs_data.csv")

N = 5 # Number of stocks
precision_bits = 2 # For each stock, this is the precision of its weight
max_wt = 1.0 - 1.0 / pow(2, precision_bits)
dim = N * precision_bits # dim stands for matrix dimensions

f = 3 * max_wt # Fixed number of stocks that can be chosen
expected_return = 0.3
sig_p = expected_return * f # Expected return from n stocks (not average currently)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.iloc[:, :N] # We need only the first N + 1 columns, 1st column is the date column

G = nx.Graph()
G.add_edges_from([(i, j) for i in range(dim) for j in range(i + 1, dim)])

# Create the covariance matrix and returns list
cov = pd.read_csv("cov_matrix_4yrs.csv")
returns = pd.read_csv("mean_returns_4yrs.csv")


def find_portfolio(principal):
    '''
    Given the principal amount, find best portfolio
    Return amount earned at the end of the month
    '''
    
    # The matrix where we add the objective and the constraint
    Q = defaultdict(int)

    # Constraint1 minimises the difference between expected return and actual return
    lagrange1 = 0.5
    for d in range(dim):
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

    distribution = sampleset.first.sample
    for s_num in distribution.keys():
        if(distribution[s_num] == 1):
            actual_return += returns.iloc[s_num]
    # Actual return percentage over how much time?
    # For a month
    return (1 + actual_return / 12) * principal

def update_returns(start_date, end_date):
    df_new = df.loc[start_date: end_date]
    # print(df_new)
    rets = df_new.pct_change()
    return rets


rebalance_interval = 21 # 21 working days approximately in a month
MONTHS = 2 # We rebalance for a year
principal = 10000 # We start out with
start_year = 2018
start_date = "2013-1"

for m in range(1, MONTHS + 1):
    # principal = find_portfolio(principal)
    print(m + 1, principal)
    yr = m // 12
    month = m % 12
    end_date = str(start_year + yr) + "-" + str(month)
    daily_returns = update_returns(start_date, end_date)
    cov = daily_returns.cov() * 252
    returns = daily_returns.mean(axis=0) * 252

# print(principal)