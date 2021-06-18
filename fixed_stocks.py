import networkx as nx
from collections import defaultdict
# from dwave.system import DWaveSampler, EmbeddingComposite

N = 3 # Number of stocks
G = nx.Graph()
G.add_edges_from([(i, j) for i in range(N) for j in range(i + 1, N)])

# The matrix where we add the objective and the constraint
Q = defaultdict(int)

lagrange = 1 # Some temporary value
