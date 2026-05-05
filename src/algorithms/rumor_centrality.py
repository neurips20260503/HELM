import math
from collections import deque
import numpy as np
import networkx as nx


def convert_graph(G: nx.Graph):
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    adj = [np.array([idx[v] for v in G.neighbors(u)], dtype=np.int32) for u in nodes]
    return nodes, adj


def bfs_tree(parent, order, adj, root):
    n = len(adj)
    parent[:] = -1
    order.fill(0)

    q = deque([root])
    parent[root] = root
    k = 0

    while q:
        u = q.popleft()
        order[k] = u
        k += 1
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)

    return order[:k]


def rumor_centrality_log(parent, order):
    n = len(parent)
    subtree = np.ones(n, dtype=np.int32)
    children = [[] for _ in range(n)]

    for v in range(n):
        if parent[v] != v:
            children[parent[v]].append(v)

    for u in reversed(order):
        for v in children[u]:
            subtree[u] += subtree[v]

    return math.lgamma(n + 1) - float(np.log(subtree).sum())


def estimate_source_exact(G: nx.Graph):
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected (infected subgraph).")

    nodes, adj = convert_graph(G)
    n = len(nodes)

    parent = np.full(n, -1, dtype=np.int32)
    order = np.zeros(n, dtype=np.int32)

    scores = np.zeros(n)
    log_fact = math.lgamma(n + 1)

    best_idx = 0
    best_score = -float("inf")

    for r in range(n):
        order_r = bfs_tree(parent, order, adj, r)

        # compute subtree sizes
        subtree = np.ones(n, dtype=np.int32)
        children = [[] for _ in range(n)]

        for v in order_r[1:]:
            children[parent[v]].append(v)

        for u in reversed(order_r):
            for v in children[u]:
                subtree[u] += subtree[v]

        score = log_fact - float(np.log(subtree).sum())
        scores[r] = score

        if score > best_score:
            best_score = score
            best_idx = r

    return nodes[best_idx], {nodes[i]: scores[i] for i in range(n)}
