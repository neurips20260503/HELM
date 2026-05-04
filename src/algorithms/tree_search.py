import argparse
import os
import random
import logging
import tempfile
import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import src.utils as utils
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Any
from collections import Counter, deque, defaultdict

LOGGER_NAME = "tree_search"

"""
tree_search simulated annealing module

Self-contained module with manifest support (like edge_features and edge_scores).
Supports both train and eval modes:
- train mode: requires true_tree (ground truth) for TPR tracking
- eval mode: optional true_tree; can use T's degree distribution or explicit target

Verbosity levels (used by the Optuna wrapper and SA config):
- 0: errors + short summary (minimal logging)
- 1: info + periodic summaries (records periodic stats; no CSV)
- 2: debug + detailed timeseries (records diagnostics and writes
    a per-trial `trial_timeseries.csv` when `config['trial_dir']` is set)

When `verbosity >= 2`, the SA run emits a CSV with columns
`step,tpr,total_loss` plus one column per loss component.
"""


def convert_to_igraph(graph: nx.Graph):
    # Map arbitrary node labels to consecutive integers required by igraph
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    ig_graph = ig.Graph(directed=graph.is_directed())
    ig_graph.add_vertices(len(node_mapping))
    ig_graph.add_edges([(node_mapping[u], node_mapping[v]) for u, v in graph.edges()])
    return ig_graph, reverse_mapping


def perform_leiden(graph: nx.Graph, resolution, seed=None, n_iterations=2):
    """
    Run Leiden community detection on `graph` and return a mapping from
    original graph node labels -> community id. If `seed` is provided it will
    be passed to `leidenalg.find_partition(..., seed=seed)` for deterministic
    results when the leiden implementation supports it.

    Args:
        graph: NetworkX graph
        resolution: resolution parameter for Leiden
        seed: RNG seed for deterministic results
        n_iterations: number of Leiden refinement iterations (default 2)
    """
    ig_graph, reverse_mapping = convert_to_igraph(graph)
    # leidenalg.find_partition accepts a `seed` parameter in the installed
    # versions; pass it when available to improve determinism.
    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=n_iterations,
    )
    # partition yields lists of igraph vertex ids; map them back to original labels
    return {
        reverse_mapping[node]: comm
        for comm, nodes in enumerate(partition)
        for node in nodes
    }


class TreeStateManager:

    def __init__(
        self,
        graph: nx.Graph,
        true_tree: nx.Graph,
        tree: nx.Graph,
        edge_scores: dict,
        communities: dict,
        config: dict,
        verbosity: int = 0,
    ):
        self.tree = tree
        # Eval mode: true_tree can be None (no ground truth)
        self.has_ground_truth = true_tree is not None
        if self.has_ground_truth:
            self.true_edges = {frozenset((u, v)) for u, v in true_tree.edges()}
            self.tp_score = sum(
                frozenset((u, v)) in self.true_edges for u, v in tree.edges()
            )
        else:
            self.true_edges = None
            self.tp_score = 0

        self.tree_edges = utils.Pool(frozenset((u, v)) for u, v in tree.edges())
        self.graph_edges = {frozenset((u, v)) for u, v in graph.edges()}
        # Optimization: Use defaultdict for 6% faster lookups (no .get() overhead)

        self.edge_scores = defaultdict(float, edge_scores or {})
        self.communities = communities
        self.weights = config["loss_weights"]

        # Optimization: Cache tree adjacency for 5x faster neighbor lookups
        self._adj_cache = {n: list(tree.neighbors(n)) for n in tree.nodes()}

        # Optimization: Cache degrees to eliminate NetworkX degree lookups
        self._degree_cache = dict(tree.degree())

        # Degree distributions (histograms) for loss/MSE
        # _degree_hist_current: counts of nodes per degree in current tree
        # _target_degree_hist: target counts per degree (ground truth or config)
        self._degree_hist_current = self._build_degree_hist(
            self._degree_cache.values(), min_len=1
        )
        self._target_degree_hist = None

        # Optimization: Cache nodes list (immutable during search)
        self._nodes_list = list(tree.nodes())

        self.losses = {}
        self.losses["community"] = self.weights["community"] * sum(
            1
            for u, v in self.tree.edges()
            if self.communities[u] != self.communities[v]
        )
        self.losses["diversity"] = self.weights["diversity"] * sum(
            self.compute_diversity(node) for node in self.tree.nodes()
        )

        # Degree loss: use degree distribution (histogram of counts per degree)
        if self.has_ground_truth:
            target_hist = self._build_degree_hist(
                (deg for _, deg in true_tree.degree()), min_len=1
            )
            self._target_degree_hist = target_hist.astype(np.float64)
        else:
            target_deg_dist = config.get("target_degree_distribution", None)
            if target_deg_dist is not None:
                # Interpret provided list/array as histogram counts per degree
                self._target_degree_hist = np.asarray(target_deg_dist, dtype=np.float64)
            else:
                self._target_degree_hist = None

        if self._target_degree_hist is not None:
            # Align current histogram to target length (pad) for MSE computation
            max_len = max(len(self._target_degree_hist), len(self._degree_hist_current))
            self._degree_hist_current = self._pad_hist(
                self._degree_hist_current, max_len
            )
            target_hist = self._pad_hist(self._target_degree_hist, max_len)
            self.losses["degree"] = self.weights["degree"] * self._hist_mse(
                self._degree_hist_current, target_hist
            )
            # Keep padded target for future delta computations
            self._target_degree_hist = target_hist
        else:
            self.losses["degree"] = 0.0
        # Count edges in the current tree that are not present in the original graph.
        # Use frozenset for undirected comparison (graph_edges stores frozensets).
        self.losses["shortcut"] = self.weights["shortcut"] * sum(
            1 for e in self.tree.edges() if frozenset(e) not in self.graph_edges
        )
        self.losses["score"] = self.weights["score"] * sum(
            self.edge_scores[frozenset((u, v))] for u, v in self.tree.edges()
        )
        # Instrumentation counters for diagnostics (with no-op methods when verbosity=0)
        # Optimization: Avoid conditional checks in hot loop using method indirection
        if verbosity > 0:
            self.move_attempts = {"nni": 0, "spr": 0, "tbr": 0}
            self.move_success = {"nni": 0, "spr": 0, "tbr": 0}
            self.rejection_reasons = {}
            self.total_attempts = 0
            self.total_accepted = 0
            # Enable tracking methods
            self._track_attempt = self._track_attempt_impl
            self._track_success = self._track_success_impl
            self._track_rejection = self._track_rejection_impl
        else:
            # No-op methods for zero overhead when verbosity=0
            self.move_attempts = None
            self.move_success = None
            self.rejection_reasons = None
            self.total_attempts = None
            self.total_accepted = None
            self._track_attempt = self._noop
            self._track_success = self._noop
            self._track_rejection = self._noop

    def _noop(self, *args, **kwargs):
        """No-op method for disabled tracking."""
        pass

    @staticmethod
    def _build_degree_hist(degrees, min_len: int = 0) -> np.ndarray:
        """Build degree histogram (counts per degree) with optional minimum length."""
        degrees = list(degrees)
        if not degrees:
            return np.zeros(max(1, min_len), dtype=np.int64)
        max_deg = max(degrees)
        hist_len = max(max_deg + 1, min_len)
        hist = np.bincount(degrees, minlength=hist_len)
        return hist.astype(np.int64)

    @staticmethod
    def _pad_hist(hist: np.ndarray, target_len: int) -> np.ndarray:
        """Pad histogram with zeros to target length (no-op if already long enough)."""
        if len(hist) >= target_len:
            return hist
        return np.pad(hist, (0, target_len - len(hist)))

    @staticmethod
    def _hist_mse(current_hist: np.ndarray, target_hist: np.ndarray) -> float:
        """Compute MSE between two histograms, padding the shorter one."""
        max_len = max(len(current_hist), len(target_hist))
        if max_len == 0:
            return 0.0
        cur = TreeStateManager._pad_hist(current_hist, max_len).astype(np.float64)
        tgt = TreeStateManager._pad_hist(target_hist, max_len).astype(np.float64)
        diff = tgt - cur
        return float(np.mean(diff * diff))

    def _track_attempt_impl(self, move_type):
        """Track move attempt (enabled when verbosity > 0)."""
        self.move_attempts[move_type] += 1
        self.total_attempts += 1

    def _track_success_impl(self, move_type):
        """Track successful move (enabled when verbosity > 0)."""
        self.move_success[move_type] += 1
        self.total_accepted += 1

    def _track_rejection_impl(self, reason):
        """Track rejection reason (enabled when verbosity > 0)."""
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def _diversity_from_communities(self, community_labels):
        """Compute diversity metric from a list of community IDs."""
        if not community_labels:
            return 0.0
        counts = Counter(community_labels)
        max_count = max(counts.values())
        total = len(community_labels)
        return 1.0 - (max_count / total)

    def compute_diversity(self, node):
        """Get neighbor diversity for a node."""
        # Optimization: Use cached adjacency (5x faster than tree.neighbors())
        community_labels = [self.communities[nbr] for nbr in self._adj_cache[node]]
        return self._diversity_from_communities(community_labels)

    def get_current_loss(self):
        return sum(self.losses.values())

    def get_current_tp(self):
        return self.tp_score

    def nni_move(self):
        self._track_attempt("nni")
        edge = self.tree_edges.sample()
        u, v = tuple(edge)
        if self._degree_cache[u] == 1 or self._degree_cache[v] == 1:
            self._track_rejection("nni_low_degree")
            return False, None
        # Use cached adjacency for 5x speedup
        u_neigh = [n for n in self._adj_cache[u] if n != v]
        v_neigh = [n for n in self._adj_cache[v] if n != u]
        a = random.choice(u_neigh)
        b = random.choice(v_neigh)
        # detect no-op: if the old and new edge sets are equal, reject
        old_set = {frozenset((u, a)), frozenset((v, b))}
        new_set = {frozenset((u, b)), frozenset((v, a))}
        if old_set == new_set:
            self._track_rejection("no_op")
            return False, None
        return True, {
            "type": "nni",
            "old": [frozenset((u, a)), frozenset((v, b))],
            "new": [frozenset((u, b)), frozenset((v, a))],
        }

    def spr_move(self):
        self._track_attempt("spr")
        edge = self.tree_edges.sample()
        a, b = tuple(edge)
        if random.random() < 0.5:
            u, v = a, b
        else:
            v, u = a, b
        # Optimization: Use BFS to find subtree containing v after removing (u,v)
        # 112x faster than nx.connected_components() - no graph copy needed!
        subtree_nodes = {v}
        queue = deque([v])
        while queue:
            node = queue.popleft()
            for nbr in self._adj_cache[node]:
                if (node == u and nbr == v) or (node == v and nbr == u):
                    continue  # Skip the edge we're removing
                if nbr not in subtree_nodes:
                    subtree_nodes.add(nbr)
                    queue.append(nbr)
        candidates = [n for n in self._nodes_list if n not in subtree_nodes]
        new_parent = random.choice(candidates)
        # detect no-op: moving v under the same parent or producing same edge
        old_set = {frozenset((u, v))}
        new_set = {frozenset((new_parent, v))}
        if old_set == new_set:
            self._track_rejection("no_op")
            return False, None
        move = {
            "type": "spr",
            "old": [frozenset((u, v))],
            "new": [frozenset((new_parent, v))],
        }
        return True, move

    def tbr_move(self):
        self._track_attempt("tbr")
        edge = self.tree_edges.sample()
        u, v = tuple(edge)
        # Optimization: Use BFS to find components after removing (u,v)
        # 112x faster than nx.connected_components() - no graph copy needed!
        comp1 = {u}
        queue = deque([u])
        while queue:
            node = queue.popleft()
            for nbr in self._adj_cache[node]:
                if (node == u and nbr == v) or (node == v and nbr == u):
                    continue  # Skip the edge we're removing
                if nbr not in comp1:
                    comp1.add(nbr)
                    queue.append(nbr)

        # comp2 is all nodes not in comp1
        comp2 = set(self._nodes_list) - comp1

        a = random.choice(list(comp1))
        b = random.choice(list(comp2))
        # detect no-op: adding an edge that is identical to the removed one
        old_set = {frozenset((u, v))}
        new_set = {frozenset((a, b))}
        if old_set == new_set:
            self._track_rejection("no_op")
            return False, None
        return True, {
            "type": "tbr",
            "old": [frozenset((u, v))],
            "new": [frozenset((a, b))],
        }

    def delta_losses(self, move):
        delta_losses = {}

        delta_losses["community"] = self.delta_community_loss(move)
        delta_losses["diversity"] = self.delta_diversity_loss(move)
        delta_losses["degree"] = self.delta_degree_loss(move)
        delta_losses["shortcut"] = self.delta_shorcut_loss(move)
        delta_losses["score"] = self.delta_score_loss(move)
        return delta_losses

    def delta_community_loss(self, move):
        delta = 0
        for e in move["old"]:
            u, v = tuple(e)
            if self.communities[u] != self.communities[v]:
                delta -= 1
        for e in move["new"]:
            u, v = tuple(e)
            if self.communities[u] != self.communities[v]:
                delta += 1
        return delta * self.weights["community"]

    def delta_diversity_loss(self, move):
        # LOCAL OPTIMIZATION: Only compute changes to affected nodes' diversity
        # (1.16x faster) instead of recomputing full diversity before & after
        affected_nodes = set()
        for e in move["old"] + move["new"]:
            affected_nodes.update(e)

        # Map edges to their endpoints for quick lookup
        old_edges = {tuple(e) for e in move["old"]}
        new_edges = {tuple(e) for e in move["new"]}

        delta_loss = 0.0

        for node in affected_nodes:
            # Get current neighbor communities
            neighbor_communities = [
                self.communities[nbr] for nbr in self._adj_cache[node]
            ]

            if not neighbor_communities:
                # Edge case: node has no neighbors currently
                # After move, it will have new neighbors
                new_communities = []
                for u, v in new_edges:
                    if u == node:
                        new_communities.append(self.communities[v])
                    elif v == node:
                        new_communities.append(self.communities[u])

                if new_communities:
                    old_diversity = 0.0
                    new_diversity = self._diversity_from_communities(new_communities)
                    delta_loss += new_diversity - old_diversity
                continue

            # Current diversity: track community counts
            counts = Counter(neighbor_communities)
            total = len(neighbor_communities)
            old_max = max(counts.values())
            old_diversity = 1.0 - (old_max / total)

            # Simulate edge removals
            for nbr in self._adj_cache[node]:
                edge_key = (node, nbr) if (node, nbr) in old_edges else (nbr, node)
                if edge_key in old_edges:
                    comm = self.communities[nbr]
                    counts[comm] -= 1
                    if counts[comm] == 0:
                        del counts[comm]
                    total -= 1

            # Simulate edge additions
            for u, v in new_edges:
                if u == node:
                    comm = self.communities[v]
                    counts[comm] = counts.get(comm, 0) + 1
                    total += 1
                elif v == node:
                    comm = self.communities[u]
                    counts[comm] = counts.get(comm, 0) + 1
                    total += 1

            # New diversity after changes
            if total > 0 and counts:
                new_max = max(counts.values())
                new_diversity = 1.0 - (new_max / total)
            else:
                new_diversity = 0.0

            delta_loss += new_diversity - old_diversity

        return delta_loss * self.weights["diversity"]

    def delta_degree_loss(self, move):
        if move["type"] == "nni":
            return 0
        if self._target_degree_hist is None:
            return 0

        u_old, v_old = tuple(move["old"][0])
        u_new, v_new = tuple(move["new"][0])

        nodes = [u_old, v_old, u_new, v_new]
        current_degrees = np.array(
            [self._degree_cache[n] for n in nodes], dtype=np.int64
        )
        degree_deltas = np.array([-1, -1, 1, 1], dtype=np.int64)

        # Ensure histograms can accommodate possible new max degree
        max_new_degree = int(np.max(current_degrees + degree_deltas))
        max_len = max(
            len(self._degree_hist_current),
            len(self._target_degree_hist),
            max_new_degree + 1,
        )

        current_hist = self._pad_hist(self._degree_hist_current, max_len)
        target_hist = self._pad_hist(self._target_degree_hist, max_len)

        # LOCAL APPROACH: Only compute changes to affected bins
        # Build degree change map: degree -> net change in count
        degree_delta = {}
        for deg, delta in zip(current_degrees, degree_deltas):
            old_deg = int(deg)
            new_deg = int(deg + delta)
            degree_delta[old_deg] = degree_delta.get(old_deg, 0) - 1
            degree_delta[new_deg] = degree_delta.get(new_deg, 0) + 1

        # Compute local delta in MSE (only affected bins)
        delta_mse = 0.0
        for deg, delta_count in degree_delta.items():
            if delta_count == 0:
                continue

            curr_count = int(current_hist[deg]) if deg < len(current_hist) else 0
            targ_count = int(target_hist[deg]) if deg < len(target_hist) else 0
            new_count = curr_count + delta_count

            # Change in squared error for this bin
            old_error = (curr_count - targ_count) ** 2
            new_error = (new_count - targ_count) ** 2

            delta_mse += new_error - old_error

        # Convert to mean
        delta_mse /= max_len
        return delta_mse * self.weights["degree"]

    def delta_shorcut_loss(self, move):
        delta = 0
        for e in move["old"]:
            if e not in self.graph_edges:
                delta -= 1
        for e in move["new"]:
            if e not in self.graph_edges:
                delta += 1
        return delta * self.weights["shortcut"]

    def delta_score_loss(self, move):
        delta = 0.0
        for e in move["old"]:
            delta -= self.edge_scores[e]
        for e in move["new"]:
            delta += self.edge_scores[e]
        return delta * self.weights["score"]

    def delta_tp(self, move):
        if not self.has_ground_truth:
            return 0
        delta_tp = 0
        for e in move["old"]:
            if e in self.true_edges:
                delta_tp -= 1
        for e in move["new"]:
            if e in self.true_edges:
                delta_tp += 1
        return delta_tp

    def apply_move(self, move, delta_losses):
        # Update tree edges (only tree_edges pool, defer NetworkX tree update)
        old_edges_tuples = [tuple(e) for e in move["old"]]
        new_edges_tuples = [tuple(e) for e in move["new"]]

        # Only update the edge pool (fast), NOT the NetworkX tree object
        self.tree_edges.remove(*move["old"])
        self.tree_edges.add(*move["new"])

        # Update cached adjacency list and degrees (our source of truth during search)
        affected_nodes = set()
        degree_changes = defaultdict(int)

        for u, v in old_edges_tuples:
            affected_nodes.update([u, v])
            self._adj_cache[u].remove(v)
            self._adj_cache[v].remove(u)
            degree_changes[u] -= 1
            degree_changes[v] -= 1

        for u, v in new_edges_tuples:
            affected_nodes.update([u, v])
            self._adj_cache[u].append(v)
            self._adj_cache[v].append(u)
            degree_changes[u] += 1
            degree_changes[v] += 1

        if degree_changes:
            if self._target_degree_hist is not None:
                # Ensure histogram can hold any new degree
                max_new_degree = max(
                    self._degree_cache[n] + delta for n, delta in degree_changes.items()
                )
                if max_new_degree >= len(self._degree_hist_current):
                    self._degree_hist_current = self._pad_hist(
                        self._degree_hist_current, max_new_degree + 1
                    )
                    self._target_degree_hist = self._pad_hist(
                        self._target_degree_hist, max_new_degree + 1
                    )

                for n, delta in degree_changes.items():
                    old_deg = self._degree_cache[n]
                    new_deg = old_deg + delta
                    self._degree_hist_current[old_deg] -= 1
                    self._degree_hist_current[new_deg] += 1
                    self._degree_cache[n] = new_deg
            else:
                for n, delta in degree_changes.items():
                    self._degree_cache[n] += delta

        # Update losses
        for key, delta in delta_losses.items():
            self.losses[key] += delta
        self.tp_score += self.delta_tp(move)

        # Update diagnostics (no-op when verbosity=0)
        mtype = move.get("type")
        if mtype:
            self._track_success(mtype)
        return

    def sync_tree_from_cache(self):
        """Rebuild NetworkX tree from cached edges (called at end of search)."""
        # Clear all edges and rebuild from tree_edges pool
        self.tree.clear_edges()
        self.tree.add_edges_from(tuple(e) for e in self.tree_edges.item_list)


def simulated_annealing(
    graph: nx.Graph,
    tree: nx.Graph,
    true_tree: nx.Graph,
    edge_scores: dict,
    config,
    return_stats: bool = False,
):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Starting simulated annealing...")
    # Seeding for reproducibility: optional base seed passed via config['seed'].
    # Use a single global seed (no per-trial offset).
    try:
        base_seed = int(config.get("seed", 42)) if isinstance(config, dict) else 42
    except Exception:
        base_seed = 42
    if base_seed is not None:
        try:
            random.seed(base_seed)
            np.random.seed(base_seed)
            logger.debug(f"Seeded RNGs with seed={base_seed}")
        except Exception:
            logger.exception("Failed to seed RNGs for simulated_annealing")
    if not nx.is_tree(tree):
        raise ValueError("Input graph is not a tree.")
    # allow configuring stagnation early-stop threshold from config
    stagnation_threshold = int(config.get("stagnation_threshold", 10000))
    # bold_stagnation_threshold: after stagnation_threshold iters without improvement,
    # we increase bold_prob; if still no improvement after this many more iters, we break
    bold_stagnation_threshold = int(config.get("bold_stagnation_threshold", 10000))

    # pass seed and n_iterations into Leiden for deterministic community detection
    leiden_n_iterations = int(config.get("leiden_n_iterations", 2))
    communities = perform_leiden(
        graph, config["resolution"], seed=base_seed, n_iterations=leiden_n_iterations
    )
    verbosity = config.get("verbosity", 0)
    manager = TreeStateManager(
        graph,
        true_tree,
        tree.copy(),
        edge_scores,
        communities,
        config,
        verbosity=verbosity,
    )
    best_loss = manager.get_current_loss()
    last_improvement_iter = 0
    T0 = T = config["initial_temp"]
    alpha = config["cooling_rate"]
    max_iter = config["max_iter"]
    report_interval = config.get("report_interval", 1000)
    verbosity = config.get("verbosity", 1)
    tpr_history = []
    history_steps = []
    loss_history = {key: [] for key in manager.losses}
    valid_moves = 0
    i = 0
    time_start = time.time()
    # Hoist config lookups outside loop to avoid redundant dict access (50M times)
    min_bold_prob = float(config.get("min_bold_prob", 0.05))
    bold_prob_scale = float(config.get("bold_prob_scale", 1.0 / 3.0))
    bold_multiplier = float(config.get("bold_multiplier", 3.0))
    for i in range(max_iter):
        # Periodic reporting: record diagnostics at fixed iteration steps
        if i % report_interval == 0:
            tpr_val = (
                manager.get_current_tp() / len(manager.true_edges)
                if manager.has_ground_truth and manager.true_edges
                else 0.0
            )
            logger.debug(
                f"Iter {i}, Loss: {manager.get_current_loss()}, TPR: {tpr_val:.4f}, Temp: {T:.4f}, Valid Moves: {valid_moves}"
            )
            # record step for time-series alignment
            history_steps.append(i)
            for key in loss_history:
                loss_history[key].append(manager.losses[key])
            # record TPR history for diagnostics
            tpr_history.append(
                manager.get_current_tp() / len(manager.true_edges)
                if manager.has_ground_truth and manager.true_edges
                else 0.0
            )

        stagnation_iters = i - last_improvement_iter
        # Use hoisted bold probability parameters
        bold_prob = max(min_bold_prob, (T / T0) * bold_prob_scale)

        # Two-tier stagnation handling:
        # 1. After stagnation_threshold iters without improvement, increase bold_prob
        if stagnation_iters > stagnation_threshold:
            bold_prob = min(1.0, bold_prob * bold_multiplier)

        # 2. If bold moves also don't help after bold_stagnation_threshold more iters, break
        if stagnation_iters > (stagnation_threshold + bold_stagnation_threshold):
            logger.info(
                f"No improvement after {stagnation_iters} iters (bold phase included); stopping early at iter {i}"
            )
            break
        r = random.random()
        if r < bold_prob / 2:
            valid, move = manager.tbr_move()
        elif r < bold_prob:
            valid, move = manager.spr_move()
        else:
            valid, move = manager.nni_move()
        if not valid:
            continue
        delta_losses = manager.delta_losses(move)
        delta = sum(delta_losses.values())
        if T > 1e-5:
            accept = delta < 0 or random.random() < np.exp(-delta / T)
        else:
            accept = delta < 0
        if accept:
            manager.apply_move(move, delta_losses)
            valid_moves += 1
            if manager.get_current_loss() < best_loss:
                best_loss = manager.get_current_loss()
                last_improvement_iter = i
        T *= alpha
        # end for

    # Sync NetworkX tree from cache (we only updated cache during loop for speed)
    manager.sync_tree_from_cache()

    if not nx.is_tree(manager.tree):
        raise RuntimeError("The resulting graph is not a tree.")
    time_end = time.time()
    # log final summary according to verbosity
    final_tpr = (
        manager.get_current_tp() / len(manager.true_edges)
        if manager.has_ground_truth and manager.true_edges
        else 0.0
    )
    logger.warning(f"Time taken: {time_end - time_start:.2f} seconds")
    logger.warning(f"Final TPR: {final_tpr:.4f}")
    logger.warning(f"Final temperature: {T}")
    logger.warning(f"Number of iterations: {i}")
    logger.warning(f"Number of valid moves: {valid_moves}")
    logger.warning("Simulated annealing completed")
    # Build run_stats if requested
    if return_stats:
        # Compute confusion matrix (only if ground truth available)
        if manager.has_ground_truth:
            confusion = utils.compute_confusion_from_trees(
                manager.true_edges, manager.tree.edges(), manager.graph_edges
            )
            tp, fp, fn, tn = (
                confusion["tp"],
                confusion["fp"],
                confusion["fn"],
                confusion["tn"],
            )
            total_true = confusion["total_true"]
        else:
            # Eval mode: no ground truth metrics
            tp = fp = fn = tn = 0
            total_true = 0

        # Base run_stats: compact scalars only (NO huge arrays)
        run_stats = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_true": total_true,
            "move_attempts": manager.move_attempts,
            "move_success": manager.move_success,
            "rejection_reasons": manager.rejection_reasons,
            "total_attempts": manager.total_attempts,
            "total_accepted": manager.total_accepted,
            "loss_summary": manager.losses,
            "duration_s": time_end - time_start,
            "iterations": i,
            # Graph size info
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_tree_edges": manager.tree.number_of_edges(),
        }

        # Derive additional human-friendly metrics (post-run)
        try:
            derived_metrics = (
                utils.compute_metrics_from_confusion(
                    tp, fp, fn, total_pos=total_true, tn=tn
                )
                if total_true > 0
                else {"tpr": 0, "fpr": 0, "precision": 0, "recall": 0, "f1": 0}
            )
            run_stats.update(derived_metrics)

            # degree MSE (raw, unweighted) using histogram distribution
            if manager._target_degree_hist is not None:
                degree_mse = manager._hist_mse(
                    manager._degree_hist_current, manager._target_degree_hist
                )
            else:
                degree_mse = None

            # illegal edges in final tree (edges not present in original graph)
            illegal_edges = sum(
                1
                for u, v in manager.tree.edges()
                if frozenset((u, v)) not in manager.graph_edges
            )

            # community mismatch (raw count)
            community_mismatch = sum(
                1
                for u, v in manager.tree.edges()
                if manager.communities.get(u) != manager.communities.get(v)
            )

            # raw (unweighted) component losses by reversing weights
            raw_losses = {}
            for k, v in manager.losses.items():
                w = manager.weights.get(k, 1.0)
                try:
                    raw_losses[k] = float(v / w) if w != 0 else float(v)
                except Exception:
                    raw_losses[k] = float(v)

            # raw score sum (before weight multiplication)
            score_sum = float(
                sum(
                    manager.edge_scores.get(frozenset((u, v)), 0.0)
                    for u, v in manager.tree.edges()
                )
            )

            # attach derived metrics
            run_stats.update(
                {
                    "degree_mse": degree_mse,
                    "illegal_edges": int(illegal_edges),
                    "community_mismatch": int(community_mismatch),
                    "raw_losses": raw_losses,
                    "score_sum": score_sum,
                }
            )
        except Exception:
            logging.getLogger(LOGGER_NAME).exception(
                "Failed to compute derived post-run metrics"
            )

        # Store timeseries data for later use (will be saved based on verbosity)
        run_stats["_timeseries_data"] = {
            "steps": history_steps,
            "tpr_history": tpr_history,
            "loss_history": loss_history,
        }

        return {
            "tree": manager.tree,
            "loss_history": loss_history,
            "tp": manager.get_current_tp(),
            "total_true": total_true,
            "run_stats": run_stats,
        }
    return manager.tree


def build_mst_tree(G: nx.Graph, edge_scores: dict) -> nx.Graph:
    """Build minimum spanning tree from edge scores (-log(s) format).

    Args:
        G: Undirected graph
        edge_scores: Dict mapping frozenset((u,v)) -> -log(s) weight (lower = better edge, i.e., higher s)

    Returns:
        MST as undirected NetworkX graph
    """
    G_weighted = G.copy()

    for u, v in G.edges():
        key = frozenset((u, v))
        # edge_scores in -log(s) format (lower = better edge, i.e., higher probability)
        weight = edge_scores.get(key, 0.0)
        G_weighted[u][v]["weight"] = weight

    mst = nx.minimum_spanning_tree(G_weighted, weight="weight")
    return mst


def build_initial_tree(G, positive_edges: set) -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_nodes_from(G.nodes())
    tree.add_edges_from(positive_edges)

    # Initialize in-degrees
    in_degree = {node: 0 for node in G.nodes()}
    for _, v in positive_edges:
        in_degree[v] += 1

    # Handle both directed and undirected edges properly:
    # Convert to frozensets for comparison to handle undirected G
    positive_edges_normalized = {frozenset((u, v)) for u, v in positive_edges}
    remaining_edges = [
        (u, v)
        for u, v in G.edges()
        if frozenset((u, v)) not in positive_edges_normalized
    ]
    uf = utils.UnionFind(G.nodes())

    # Union for positive edges
    for u, v in positive_edges:
        uf.union(u, v)

    is_undirected = not G.is_directed()

    while len(tree.edges()) < len(G.nodes()) - 1:
        added = False

        for i, (u, v) in enumerate(remaining_edges):
            # For undirected graphs, try BOTH directions to avoid directionality bias
            # The arbitrary order of edges from G.edges() shouldn't determine tree structure
            if is_undirected:
                # Try (u, v): v as child
                if in_degree[v] == 0 and uf.union(u, v):
                    tree.add_edge(u, v)
                    in_degree[v] += 1
                    remaining_edges.pop(i)
                    added = True
                    break
                # Try (v, u): u as child
                elif in_degree[u] == 0 and uf.union(v, u):
                    tree.add_edge(v, u)
                    in_degree[u] += 1
                    remaining_edges.pop(i)
                    added = True
                    break
            else:
                # Directed graph: only check the given direction
                if in_degree[v] == 0 and uf.union(u, v):
                    tree.add_edge(u, v)
                    in_degree[v] += 1
                    remaining_edges.pop(i)
                    added = True
                    break

        if not added:
            # Try adding from complete graph if G is exhausted
            for u in G.nodes():
                for v in G.nodes():
                    if (
                        u != v
                        and (u, v) not in tree.edges()
                        and in_degree[v] == 0
                        and uf.union(u, v)
                    ):
                        tree.add_edge(u, v)
                        in_degree[v] += 1
                        added = True
                        break
                if added:
                    break

        if not added:
            raise RuntimeError(
                "Unable to build initial tree. Tree is not arborescence."
            )

    # Ensure the tree is arborescence
    if not nx.is_arborescence(tree):
        raise RuntimeError("The resulting tree is not arborescence.")
    logging.getLogger(LOGGER_NAME).info("Initial tree built successfully.")
    return tree


def precompute_for_manifest_entry(
    G_path,
    T_path,
    score_path,
    positive_edges_path=None,
    mode="train",
    init_method="positive_edges",
    skip_score_loading=False,
):
    """Precompute objects for simulated annealing from manifest entry.

    Args:
        G_path, T_path, score_path, positive_edges_path: Paths to graph/truth/scores/edges
        mode: "train" or "eval"
        init_method: "positive_edges", "mst", or "empty"
        skip_score_loading: If True, do not load edge_scores (for no_scores variant of SA)
                           This avoids I/O and score computation entirely.

    Returns dict: G, T (can be None), initial_tree, edge_scores (undirected log-loss), positive_edges (can be empty).
    """
    G = utils.load_graph(G_path)
    G_ud = G.to_undirected() if hasattr(G, "to_undirected") else G

    T_ud = None
    if T_path:
        T = utils.load_graph(T_path)
        T_ud = T.to_undirected() if hasattr(T, "to_undirected") else T

    # Load and convert scores to undirected log-loss map
    # If skip_score_loading=True (for pure SA without scores), edge_scores remains empty dict
    edge_scores = {}
    if not skip_score_loading and score_path and os.path.exists(score_path):
        try:
            raw_scores = utils.load_edge_scores(score_path)
            seen = set()
            for (u, v), s in raw_scores.items():
                key = frozenset((u, v))
                if key not in seen:
                    s_rev = raw_scores.get((v, u), 0.0)
                    edge_scores[key] = float(-np.log(max(s, s_rev) + 1e-9))
                    seen.add(key)
        except Exception:
            logging.getLogger(LOGGER_NAME).exception(
                f"Failed to read score_path {score_path}"
            )

    # Load positive edges (only needed for positive_edges init method)
    positive_edges = set()
    if (
        init_method == "positive_edges"
        and positive_edges_path
        and os.path.exists(positive_edges_path)
    ):
        try:
            positive_edges = set(utils.load_graph(positive_edges_path).edges())
        except Exception:
            logging.getLogger(LOGGER_NAME).exception(
                f"Failed to read positive_edges_path {positive_edges_path}"
            )

    # Build initial tree based on init_method
    if init_method == "mst":
        initial_tree = build_mst_tree(G_ud, edge_scores)
    elif init_method == "empty":
        initial_tree = build_initial_tree(G, set())  # No seed edges
    else:  # "positive_edges" (default)
        initial_tree = build_initial_tree(G, positive_edges)
    initial_tree = (
        initial_tree.to_undirected()
        if hasattr(initial_tree, "to_undirected")
        else initial_tree
    )

    return {
        "G": G_ud,
        "T": T_ud,
        "initial_tree": initial_tree,
        "edge_scores": edge_scores,
        "positive_edges": positive_edges,
    }


def plot_loss_history(loss_history):
    """Generates a plot for loss components over iterations."""
    plt.figure(figsize=(10, 6))
    for key, values in loss_history.items():
        plt.plot(range(0, len(values) * 1000, 1000), values, label=key)
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.title("Loss Components Over Simulated Annealing")
    plt.legend()
    plt.grid(True)
    out_dir = getattr(plot_loss_history, "out_dir", None)
    out_path = os.path.join(out_dir, "loss_plot.png") if out_dir else "loss_plot.png"
    plt.savefig(out_path)
    try:
        plt.close()
    except Exception:
        pass
    logging.getLogger(LOGGER_NAME).info(f"Loss plot saved as {out_path}")


# ================================================================================
# Manifest processing functions (like edge_features and edge_scores modules)
# ================================================================================


def validate_manifest_entry(
    entry: Dict[str, Any], mode: str = "train", skip_score_validation: bool = False
) -> Dict[str, Any]:
    """
    Validate manifest entry has required fields based on mode.

    Args:
        entry: Manifest entry dict
        mode: "train" or "eval"
        skip_score_validation: If True, do not require score_path (for no_scores variant)

    TRAIN mode requirements: graph_id, G_path, T_path, score_path (unless skip_score_validation=True), positive_edges_path
    EVAL mode requirements: graph_id, G_path, score_path (unless skip_score_validation=True) (T_path optional, positive_edges_path not used)
    """
    # Always required
    required = ["graph_id", "G_path"]
    if not skip_score_validation:
        required.append("score_path")

    if mode == "train":
        # Train mode also requires ground truth tree and positive edges
        required.extend(["T_path", "positive_edges_path"])
    # Eval mode: only the base requirements (G_path, score_path)

    missing = [f for f in required if f not in entry or not entry[f]]
    if missing:
        raise ValueError(
            f"Manifest entry (graph_id={entry.get('graph_id')}) missing required fields for {mode} mode: {missing}"
        )
    return entry


def process_manifest_entry(
    entry: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
    mode: str = "train",
    force: bool = False,
    init_method: str = "positive_edges",
    output_suffix: str = "search",
) -> Dict[str, Any]:
    """
    Process a single manifest entry (train or eval mode).

    Args:
        entry: Manifest entry with graph_id, G_path, score_path, positive_edges_path,
               and optional T_path (required for train mode)
        config: SA configuration dict
        output_dir: Base output directory
        mode: "train" (requires T_path, computes TPR) or "eval" (T_path optional)
        force: If True, recompute even if output exists

    Returns:
        Dict with results: graph_id, tree_path, metrics (tpr, f1, etc), run_stats
    """
    logger = logging.getLogger(LOGGER_NAME)
    gid = entry["graph_id"]
    collection = entry.get("collection", "")

    # Setup output paths: outputs/[collection/]graph_id/{output_suffix}/
    base_dir = (
        os.path.join(output_dir, collection, gid)
        if collection
        else os.path.join(output_dir, gid)
    )
    graph_out_dir = os.path.join(base_dir, output_suffix)
    os.makedirs(graph_out_dir, exist_ok=True)
    S_path = os.path.join(graph_out_dir, "tree.pkl")  # Output tree (solution)
    metrics_json_path = os.path.join(graph_out_dir, "metrics.json")

    # Skip if exists and not forcing
    if os.path.exists(S_path) and not force:
        logger.info(f"Tree for {gid} already exists, skipping")
        result = {"graph_id": gid, "tree_path": S_path, "skipped": True}
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path, "r") as f:
                result["metrics"] = json.load(f)
        return result

    logger.info(f"Processing {gid} in {mode} mode")

    # Load precomputed data
    precomputed = precompute_for_manifest_entry(
        G_path=entry["G_path"],
        T_path=entry.get("T_path"),  # Can be None in eval mode
        score_path=entry["score_path"],
        positive_edges_path=entry.get(
            "positive_edges_path"
        ),  # Can be None in eval mode
        mode=mode,
        init_method=init_method,
    )

    G = precomputed["G"]
    T = precomputed.get("T")  # None if not provided
    initial_tree = precomputed["initial_tree"]
    edge_scores = precomputed["edge_scores"]
    degree_dist_path = entry.get("degree_dist_path")

    # Mode-specific configuration
    local_config = dict(config)
    local_config["trial_dir"] = graph_out_dir

    if mode == "train":
        if T is None:
            raise ValueError(f"Train mode requires T_path for graph {gid}")

    elif mode == "eval":
        # Eval mode: optional ground truth for degree distribution
        if T is not None and local_config.get("use_T_degree_dist", False):
            target_hist = TreeStateManager._build_degree_hist(
                (deg for _, deg in T.degree()), min_len=1
            ).tolist()
            local_config["target_degree_distribution"] = target_hist
            logger.info(
                f"Using T degree distribution (histogram) for {gid} (eval mode)"
            )

        # Explicit degree distribution path overrides config
        if degree_dist_path:
            try:
                local_config["target_degree_distribution"] = (
                    utils.load_degree_distribution(degree_dist_path)
                )
                logger.info(
                    f"Loaded degree distribution from {degree_dist_path} for {gid}"
                )
            except Exception:
                logger.exception(
                    f"Failed to load degree distribution from {degree_dist_path}; skipping degree loss"
                )
        elif T is None and "target_degree_distribution" not in local_config:
            logger.warning(
                f"No ground truth and no target_degree_distribution for {gid}, skipping degree loss"
            )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")

    # Run SA (always returns dict with return_stats=True)
    res = simulated_annealing(
        G, initial_tree, T, edge_scores, local_config, return_stats=True
    )

    # Extract results from dict
    S = res["tree"]
    run_stats = res.get("run_stats", {})
    loss_history = res.get("loss_history")

    # Extract timeseries data (temporary, not saved in metrics.json)
    timeseries_data = run_stats.pop("_timeseries_data", None)

    # Save outputs to graph output directory
    # Structure: output_dir/[collection/]graph_id/search/{tree.pkl, metrics.json, [loss_plot.png], [trial_timeseries.csv]}

    # Save tree (S = solution) to tree.pkl
    utils.save_graph(S, S_path)
    logger.info(f"Saved tree for {gid} to {S_path}")

    # Compute and save metrics (compact: scalars and small dicts only, NO huge arrays)
    metrics = utils.compute_tree_metrics(T, S) if T is not None else {}
    metrics.update(run_stats)
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_json_path}")

    # Save visualizations and detailed timeseries based on verbosity
    verbosity = local_config.get("verbosity", 1)

    # Verbosity >= 1: Save loss plot
    if verbosity >= 1 and loss_history:
        setattr(plot_loss_history, "out_dir", graph_out_dir)
        plot_loss_history(loss_history)
        logger.info(f"Saved loss plot to {graph_out_dir}/loss_plot.png")

    # Verbosity >= 2: Save detailed timeseries CSV
    if verbosity >= 2 and timeseries_data:
        try:
            steps = timeseries_data["steps"]
            tpr_hist = timeseries_data["tpr_history"]
            loss_hist = timeseries_data["loss_history"]

            n_points = len(steps)
            total_losses = [
                sum(
                    (
                        loss_hist.get(key, [])[idx]
                        if idx < len(loss_hist.get(key, []))
                        else 0.0
                    )
                    for key in loss_hist
                )
                for idx in range(n_points)
            ]

            df = pd.DataFrame(
                {
                    "step": steps,
                    "tpr": tpr_hist,
                    "total_loss": total_losses,
                    **{key: pd.Series(vals) for key, vals in loss_hist.items()},
                }
            )

            csv_path = os.path.join(graph_out_dir, "trial_timeseries.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved timeseries CSV to {csv_path}")
        except Exception:
            logger.exception(f"Failed to write timeseries CSV for {gid}")

    return {
        "graph_id": gid,
        "tree_path": S_path,
        "metrics": metrics,
        "skipped": False,
    }


def process_manifest(
    manifest_path: str,
    config: Dict[str, Any],
    output_dir: str,
    mode: str = "train",
    workers: int = 1,
    force: bool = False,
    init_method: str = "positive_edges",
    output_suffix: str = "search",
) -> pd.DataFrame:
    """Process all graphs in a manifest (train or eval mode).

    TRAIN mode: Every entry MUST have T_path
    EVAL mode: T_path is optional per entry
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Load and validate manifest
    with open(manifest_path, "r") as f:
        entries = [validate_manifest_entry(e, mode=mode) for e in json.load(f)]
    logger.info(f"Processing {len(entries)} graphs in {mode} mode")

    # Process graphs (sequential or parallel)
    if workers == 1:
        results = [
            process_manifest_entry(
                e, config, output_dir, mode, force, init_method, output_suffix
            )
            for e in entries
        ]
    else:
        logger.info(f"Using {workers} parallel workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = [
                f.result()
                for f in [
                    executor.submit(
                        process_manifest_entry,
                        e,
                        config,
                        output_dir,
                        mode,
                        force,
                        init_method,
                        output_suffix,
                    )
                    for e in entries
                ]
            ]

    # Create summary dataframe with essential metrics only
    summary_data = []
    for result in results:
        row = {
            "graph_id": result["graph_id"],
            "tree_path": result.get("tree_path"),
            "skipped": result.get("skipped", False),
        }

        # Extract only essential scalar metrics (exclude huge arrays and nested dicts)
        if "metrics" in result:
            metrics = result["metrics"]
            essential_keys = [
                # Graph size
                "n_nodes",
                "n_edges",
                "n_tree_edges",
                # Performance metrics
                "tpr",
                "fpr",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "tn",
                "total_true",
                # Quality metrics
                "degree_mse",
                "illegal_edges",
                "community_mismatch",
                "score_sum",
                # Run statistics
                "total_attempts",
                "total_accepted",
                "duration_s",
                "iterations",
            ]
            for key in essential_keys:
                if key in metrics:
                    row[key] = metrics[key]
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Determine collection from manifest entries
    collection = entries[0].get("collection", "") if entries else ""

    # Save summary to collection-specific directory with descriptive name
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if collection:
        summary_dir = os.path.join(output_dir, collection)
        os.makedirs(summary_dir, exist_ok=True)
        summary_filename = f"tree_search_{mode}_{timestamp}.csv"
    else:
        summary_dir = output_dir
        summary_filename = f"tree_search_{mode}_{timestamp}.csv"

    summary_path = os.path.join(summary_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Print statistics
    for metric in ["tpr", "f1"]:
        if metric in summary_df.columns:
            vals = summary_df[metric].dropna()
            logger.info(f"Mean {metric.upper()}: {vals.mean():.4f} ± {vals.std():.4f}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Simulated annealing tree search with manifest or single-graph input (train/eval modes)"
    )

    # Manifest or single-graph input
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest JSON (for processing multiple graphs)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        help="Path to graph pkl (single-graph mode; provide with --gid to create temp manifest)",
    )
    parser.add_argument(
        "--gid",
        type=str,
        help="Graph ID for single-graph mode",
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Collection name for output path (optional)",
    )
    parser.add_argument(
        "--score-path",
        type=str,
        help="Path to edge scores CSV (required for both modes)",
    )
    parser.add_argument(
        "--positive-edges-path",
        type=str,
        help="Path to positive edges pkl (required in train mode only, not used in eval mode)",
    )
    parser.add_argument(
        "--degree-dist-path",
        type=str,
        help="Path to degree distribution file (json/csv/txt) for eval mode without T",
    )

    # Tree (ground truth) - required in train mode, optional in eval
    parser.add_argument(
        "--tree",
        type=str,
        help="Path to ground truth tree pkl (required in train mode, optional in eval)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Mode: 'train' (requires T_path in all manifest entries) or 'eval' (T_path optional)",
    )

    # Output and processing
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (manifest mode only)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to SA config JSON (optional; uses Optuna-optimized defaults)",
    )
    parser.add_argument(
        "--init-method",
        type=str,
        choices=["positive_edges", "mst", "empty"],
        default="positive_edges",
        help="Initial tree construction method: positive_edges (use positive.pkl, default), "
        "mst (MST with -log(score) weights, maximum-likelihood tree), empty (arbitrary union find tree)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        help="Override SA max_iter",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Verbosity: 0=errors+summary, 1=info+periodic, 2=debug+detailed",
    )

    # Eval mode options
    parser.add_argument(
        "--use-T-degree-dist",
        action="store_true",
        help="In eval mode with T_path, use T's degree distribution as degree loss target",
    )

    # Control
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if outputs exist",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="search",
        help="Output directory suffix: search (default), search_mst, search_sa, search_mst_sa",
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = utils.load_config(args.config)
    else:
        # Default config (Optuna-optimized, compatible with optuna_tree_search.py)
        config = {
            "verbosity": 0,
            "initial_temp": 0.059975464498114105,
            "cooling_rate": 0.9999916174766893,
            "max_iter": 5000000,
            "loss_weights": {
                "community": 1.05289320232193785,
                "diversity": 3.9434261916737015,
                "degree": 1.308403018937822,
                "shortcut": 1.79904543576019,
                "score": 5.24011451979451394,
            },
            "resolution": 1.19409453492559967,
            "stagnation_threshold": 10000,
            "bold_stagnation_threshold": 10000,
            "leiden_n_iterations": 2,
            "seed": 42,
            "min_bold_prob": 0.05,
            "bold_prob_scale": 0.3333333333333333,
            "bold_multiplier": 3.0,
        }

    # Apply CLI overrides to config
    if args.max_iter is not None:
        config["max_iter"] = args.max_iter
    config["verbosity"] = args.verbosity
    config["use_T_degree_dist"] = args.use_T_degree_dist

    # Convert single-graph to temp manifest if needed (like edge_features/edge_scores)
    manifest_path = args.manifest
    temp_manifest = None

    if not manifest_path:
        # Single-graph mode: convert to temp manifest
        if not args.graph or not args.gid or not args.score_path:
            raise ValueError(
                "Provide either --manifest or (--graph, --gid, --score-path)"
            )

        # Mode-specific validation
        if args.mode == "train":
            if not args.tree:
                raise ValueError("Train mode requires --tree")
            if not args.positive_edges_path:
                raise ValueError("Train mode requires --positive-edges-path")
        # Eval mode: --tree and --positive-edges-path optional

        # Build single-entry manifest
        manifest_entry = {
            "graph_id": args.gid,
            "collection": args.collection or "",
            "G_path": args.graph,
            "score_path": args.score_path,
        }
        if args.tree:
            manifest_entry["T_path"] = args.tree
        if args.positive_edges_path:
            manifest_entry["positive_edges_path"] = args.positive_edges_path
        if args.degree_dist_path:
            manifest_entry["degree_dist_path"] = args.degree_dist_path

        # Create temp manifest file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([manifest_entry], f, indent=2)
            temp_manifest = f.name
        manifest_path = temp_manifest

    try:
        # Setup logger
        logger = utils.setup_logger(
            args.output_dir, name=LOGGER_NAME, level=config["verbosity"]
        )

        # Process manifest (single-graph was converted above)
        logger.info(
            f"Processing manifest in {args.mode} mode with {args.workers} workers (init_method={args.init_method})"
        )
        process_manifest(
            manifest_path=manifest_path,
            config=config,
            output_dir=args.output_dir,
            mode=args.mode,
            workers=args.workers,
            force=args.force,
            init_method=args.init_method,
            output_suffix=args.output_suffix,
        )
    finally:
        # Cleanup temp manifest if created
        if temp_manifest:
            try:
                os.remove(temp_manifest)
            except Exception:
                pass


if __name__ == "__main__":
    main()
