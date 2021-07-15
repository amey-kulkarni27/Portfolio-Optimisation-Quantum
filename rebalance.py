import networkx as nx
from collections import defaultdict
import numpy as np
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
import math
import random

gold = False

df = pd.read_csv("indian3.csv")

N = 3 # Number of stocks

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
if gold:
    df = df.iloc[:, np.r_[:N, -1]] # We need only the first N columns, 1st column is the date column
    df = df.fillna(method='ffill')
    N += 1
else:
    df = df.iloc[:, :N]

precision_bits = 6 # For each stock, this is the precision of its weight
max_wt = 1.0 - 1.0 / pow(2, precision_bits)
dim = N * precision_bits # dim stands for matrix dimensions

f = N * max_wt # Fixed number of stocks that can be chosen
expected_return = 0.15
sig_p = expected_return * f # Expected return from n stocks (not average currently)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


G = nx.Graph()
G.add_edges_from([(i, j) for i in range(dim) for j in range(i + 1, dim)])



def find_portfolio(principal, start_year, m):
    '''
    Given the principal amount, find best portfolio
    Return amount earned at the end of the month
    '''
    
    # The matrix where we add the objective and the constraint
    Q = defaultdict(int)

    # Constraint1 minimises the difference between expected return and actual return
    lagrange1 = 1
    for d in range(dim):
        i = d // precision_bits # The stock number
        p = d % precision_bits + 1 # The p^th of the bits we are using to represent the i^th stock
        ri = means.iloc[i] # i^th stock returns
        Q[(d, d)] += (-2 * sig_p * ri / pow(2, p) + ri * ri / pow(2, 2 * p)) * lagrange1
        for d_dash in range(d + 1, dim):
            j = d_dash // precision_bits # The stock number
            q = d_dash % precision_bits + 1 # The q^th of the bits we are using to represent the j^th stock
            rj = means.iloc[j] # j^th stock returns
            Q[(d, d_dash)] += 2 * ri * rj * lagrange1 / pow(2, p + q)


    # Constraint2 specifying only f stocks should be used
    lagrange2 = 0.02
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
    # print("Response Sent")
    sampleset = sampler.sample_qubo(Q, num_reads=10, chain_strength=1)
    # print("Response Received")

    # Print the entire sampleset, that is, the entire table
    # print(sampleset)
    

    wts = [0.0 for i in range(N)]

    distribution = sampleset.first.sample
    for s_num in distribution.keys():
        if(distribution[s_num] == 1):
            i = s_num // precision_bits # Stock number
            p = s_num % precision_bits + 1 # Bit number
            wts[i] += 1 / pow(2, p)
    # For a month
    
    # wts = [1 for i in range(N)]
    wts = [wts[i] / sum(wts) for i in range(len(wts))]

    print(month_names[m])
    print("Axis Bank:", wts[0])
    print("Adani Ports:", wts[1])
    print("Asian Paints:", wts[2])
    print("")

    # Distribution of principal for each stock
    budget = [principal * wts[i] for i in range(N)]
    # print(budget)
    
    # The month in which we are going to do the transaction
    yr = m // 12
    month = m % 12 + 1
    date = str(start_year + yr) + "-" + str(month) 
    # When rebalancing yearly
    # date = str(start_year + yr)

    # Stock prices in that month
    month_prices = df.loc[date]
    # Buy on the first day of the month
    buying_prices = month_prices.iloc[:1, :]
    # Sell on the last day of the month
    selling_prices = month_prices.iloc[-1:, :]

    # print(buying_prices)

    # print(selling_prices)

    # Number bought for each stock
    stocks_bought = [budget[i] // buying_prices.iloc[0, i] for i in range(N)]

    # Money expended in the process
    money_spent = [stocks_bought[i] * buying_prices.iloc[0, i] for i in range(N)]
    # Money leftover, due to rounding
    leftover = principal - sum(money_spent)

    money_gained = [stocks_bought[i] * selling_prices.iloc[0, i] for i in range(N)]

    # We buy stocks from the first day of the month, and sell on the last day
    return sum(money_gained) + leftover

def update_returns(start_date, end_date):
    df_new = df.loc[start_date: end_date]
    # print(df_new)
    rets = df_new.pct_change()
    return rets

def dumb_portfolio(principal, start_date, end_date):
    

    wts = [1 for i in range(N)]
    wts = [wts[i] / sum(wts) for i in range(len(wts))]

    # Distribution of principal for each stock
    budget = [principal * wts[i] for i in range(N)]
    
    # Stock prices in that month
    month_prices = df.loc[start_date: end_date]
    # Buy on the first day of the month
    buying_prices = month_prices.iloc[:1, :]
    # Sell on the last day of the month
    selling_prices = month_prices.iloc[-1:, :]

    # Number bought for each stock
    stocks_bought = [budget[i] // buying_prices.iloc[0, i] for i in range(N)]

    # Money expended in the process
    money_spent = [stocks_bought[i] * buying_prices.iloc[0, i] for i in range(N)]
    # Money leftover, due to rounding
    leftover = principal - sum(money_spent)

    money_gained = [stocks_bought[i] * selling_prices.iloc[0, i] for i in range(N)]

    # We buy stocks from the first day of the month, and sell on the last day
    return sum(money_gained) + leftover

MONTHS = 12 # We monthly rebalance for a year
principal = 100000 # We start out with
start_data = "2010-1" # Data collection starts
end_data = "2011-12" # Data Collection ends
timeline_start = 2012 # Start rebalancing process

print("Starting Amount:", principal)

return_pct = df.loc[start_data: end_data].pct_change()
# Create the covariance matrix and returns list
cov = return_pct.cov() * 252
means = return_pct.mean(axis=0) * 252

data_start = 2010
start_yr = "2012"
end_yr = "2012"

dumb = dumb_portfolio(principal, start_yr, end_yr)
print("Dumb Portfolio:", dumb)

for m in range(MONTHS):
    mdash = m
    
    principal = find_portfolio(principal, timeline_start, mdash)
    
    yr = mdash // 12
    month = mdash % 12 + 1
    # print(month, principal)
    end_date = str(timeline_start + yr) + "-" + str(month)
    start_date = str(data_start + yr) + "-" + str(month % 12 + 1)

    daily_returns = update_returns(start_date, end_date)
    cov = daily_returns.cov() * 252
    means = daily_returns.mean(axis=0) * 252

print("Quantum Annealing Result:", principal)