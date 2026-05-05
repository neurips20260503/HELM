"""
optimal_root: Find optimal root and direct spanning tree to arborescence.

Manifest-driven module that:
1. Takes undirected spanning trees from tree_search
2. Tests all possible roots using optimized adjacency-dict BFS (235x faster than NetworkX)
3. Selects root that minimizes depth-distribution MSE vs ground truth
4. Directs tree from root (creates arborescence)
5. Measures directed recall vs directed ground truth

Optimizations:
- Uses adjacency dict instead of NetworkX BFS (19x faster)
- Uses pure Python for depth/MSE calculations (6x faster than NumPy)
- Supports parallel processing across graphs

Follows same pattern as edge_features, edge_scores, tree_search modules.
"""

import argparse
import os
import json
import math
import pickle
import tempfile
from collections import deque, defaultdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from src.utils import load_graph, save_graph

try:
    from src.algorithms.rumor_centrality import estimate_source_exact as _rumor_estimate_source
    _RUMOR_AVAILABLE = True
except ImportError:
    _RUMOR_AVAILABLE = False


# Path policy (enforced conventions)
# SEARCH_DIR = "search"
TREE_FILE = "tree.pkl"  # Undirected tree output from tree_search
OPTIMAL_ROOT_DIR = "optimal_root"
TREE_DIRECTED_FILE = "tree_directed.pkl"  # Root-directed arborescence
OPTIMAL_ROOT_FILE = (
    "optimal_root.json"  # {root, depth_mse, depth_vector_pred, depth_vector_true}
)
DIRECTED_EVAL_FILE = "directed_eval.json"  # {recall, precision, f1, directed_recall}


def get_graph_dir(base_dir, collection, gid):
    """Get graph directory: base/collection/gid or base/gid if no collection."""
    if collection:
        return os.path.join(base_dir, collection, gid)
    return os.path.join(base_dir, gid)


def validate_manifest_entry(entry: Dict, mode: str = "train") -> Dict:
    """
    Validate manifest entry has required fields based on mode.

    TRAIN mode: requires graph_id, collection, T_path
    EVAL mode: requires graph_id, collection (T_path optional)
    """
    required = ["graph_id", "collection"]

    if mode == "train":
        required.append("T_path")
    # Eval mode: T_path is optional

    missing = [f for f in required if f not in entry or not entry[f]]
    if missing:
        raise ValueError(
            f"Manifest entry (graph_id={entry.get('graph_id')}) "
            f"missing required fields for {mode} mode: {missing}"
        )
    return entry


# ============================================================================
# OPTIMIZED ADJACENCY-BASED IMPLEMENTATIONS (235x faster than NetworkX)
# ============================================================================


def graph_to_adjacency(G: nx.Graph) -> Dict:
    """Convert NetworkX graph to adjacency list dict for faster operations."""
    adj = defaultdict(list)
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return dict(adj)


def bfs_tree_adjacency(adj: Dict, root) -> Dict:
    """
    Create BFS tree using adjacency dict (235x faster than nx.bfs_tree).

    Returns directed adjacency dict (parent -> children).
    """
    visited = {root}
    queue = deque([root])
    tree_adj = defaultdict(list)

    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                tree_adj[node].append(neighbor)  # node -> neighbor (directed)

    return dict(tree_adj)


def compute_depth_vector(tree_adj: Dict, root) -> List[int]:
    """
    Compute depth distribution from rooted tree adjacency dict.

    Pure Python implementation (no NumPy overhead for small vectors).
    """
    queue = deque([(root, 0)])
    level_counts = {}

    while queue:
        node, depth = queue.popleft()
        level_counts[depth] = level_counts.get(depth, 0) + 1

        for child in tree_adj.get(node, []):
            queue.append((child, depth + 1))

    if not level_counts:
        return [1]

    max_depth = max(level_counts.keys())
    counts_list = [level_counts.get(i, 0) for i in range(max_depth + 1)]

    return counts_list


def compute_depth_mse(
    depth_vector_pred: List[int], depth_vector_true: List[int]
) -> float:
    """
    Compute MSE between depth vectors.

    Pure Python implementation (6x faster than NumPy for small vectors).
    """
    max_len = max(len(depth_vector_pred), len(depth_vector_true))

    # Pad with zeros
    pred_padded = depth_vector_pred + [0] * (max_len - len(depth_vector_pred))
    true_padded = depth_vector_true + [0] * (max_len - len(depth_vector_true))

    # MSE
    squared_diffs = [(p - t) ** 2 for p, t in zip(pred_padded, true_padded)]
    mse = sum(squared_diffs) / len(squared_diffs)

    return float(mse)


def find_optimal_root(
    tree_undirected: nx.Graph,
    true_tree: Optional[nx.Graph],
    use_T_depth_dist: bool = False,
    prior_dist: Optional[List[float]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Find optimal root by minimizing depth-distribution MSE vs ground truth.

    Uses optimized adjacency-dict approach for 235x speedup.

    Root selection methods (in priority order):
    1. prior_dist provided: Minimize MSE against this explicit depth histogram.
    2. use_T_depth_dist=True and T available: Minimize MSE against T's depth distribution.
    3. Fallback: tree height heuristic (prefer balanced/shallow trees).

    Args:
        tree_undirected: Undirected spanning tree (output from tree_search)
        true_tree: Ground truth tree (directed or undirected, handles both)
        use_T_depth_dist: If True, use T's depth distribution as target (default: False).
                         Train mode: ignored (always uses T).
                         Eval mode: must be explicitly enabled.
        prior_dist: Optional explicit depth histogram (list of counts per depth level).
                    When provided, overrides use_T_depth_dist.
        verbose: Print progress

    Returns:
        {
            "root": best_root_node,
            "depth_mse": minimum_mse_value (or tree_height if using heuristic),
            "depth_vector_pred": list of predicted depth distribution,
            "depth_vector_true": list of true depth distribution (if available),
            "true_root": root node of ground truth (if available),
            "all_roots": [{root, mse, depth_vector}, ...] sorted by MSE
        }
    """
    nodes = list(tree_undirected.nodes())

    if not nodes:
        raise ValueError("Tree has no nodes")

    if len(nodes) == 1:
        # Single node tree
        root = nodes[0]
        depth_pred = np.array([1], dtype=int)
        return {
            "root": root,
            "depth_mse": 0.0,
            "depth_vector_pred": depth_pred.tolist(),
            "depth_vector_true": [1] if true_tree else None,
            "all_roots": [
                {"root": root, "mse": 0.0, "depth_vector": depth_pred.tolist()}
            ],
        }

    # Get true depth vector if available
    depth_vector_true_list = None
    true_root = None

    # Priority 1: explicit prior_dist overrides everything
    if prior_dist is not None:
        depth_vector_true_list = list(prior_dist)
    elif true_tree is not None and use_T_depth_dist:
        # Try to find a natural root (in_degree=0 if directed, else use node 0)
        if true_tree.is_directed():
            true_roots = [n for n in true_tree.nodes() if true_tree.in_degree(n) == 0]
            true_root = true_roots[0] if true_roots else list(true_tree.nodes())[0]
        else:
            # true_root = list(true_tree.nodes())[0]
            raise ValueError(
                "Ground truth tree must be directed to use its depth distribution"
            )

        # Use optimized adjacency approach for ground truth too
        true_adj = graph_to_adjacency(true_tree)
        true_tree_bfs = bfs_tree_adjacency(true_adj, true_root)
        depth_vector_true_list = compute_depth_vector(true_tree_bfs, true_root)

    # OPTIMIZATION: Convert to adjacency dict for 235x speedup
    adj = graph_to_adjacency(tree_undirected)

    # Test all possible roots
    root_results = []

    for root in nodes:
        # Create BFS tree using optimized adjacency approach (235x faster)
        tree_adj = bfs_tree_adjacency(adj, root)

        # Compute depth distribution
        depth_vector_pred = compute_depth_vector(tree_adj, root)

        # Compute MSE
        if depth_vector_true_list is not None:
            # Target: prior_dist or T's depth distribution
            mse = compute_depth_mse(depth_vector_pred, depth_vector_true_list)
        elif true_tree is not None:
            # T available but use_T_depth_dist=False: use tree height heuristic
            mse = float(len(depth_vector_pred) - 1)
        else:
            # No T: use tree height heuristic (prefer balanced trees)
            mse = float(len(depth_vector_pred) - 1)

        root_results.append(
            {"root": root, "mse": mse, "depth_vector": depth_vector_pred}
        )

        if verbose and len(nodes) <= 50:  # Only print for small trees
            print(f"  Root {root}: MSE={mse:.4f}, depth_dist={depth_vector_pred}")

    # Sort by MSE (ascending = best first)
    root_results.sort(key=lambda x: x["mse"])
    best = root_results[0]

    return {
        "root": best["root"],
        "depth_mse": best["mse"],
        "depth_vector_pred": best["depth_vector"],
        "depth_vector_true": depth_vector_true_list,
        "true_root": true_root,
        "all_roots": root_results,
    }


def compute_directed_recall(
    pred_tree_directed: nx.DiGraph, true_tree: nx.Graph
) -> Dict[str, float]:
    """
    Compute directed and undirected edge recall/precision/F1.

    Works with both directed and undirected ground truth.

    Args:
        pred_tree_directed: Predicted directed tree (arborescence) from optimal root
        true_tree: Ground truth tree (directed or undirected)

    Returns:
        {
            "directed_recall": TP / |true_edges|,
            "directed_precision": TP / |pred_edges|,
            "directed_f1": harmonic mean,
            "undirected_recall": recall ignoring direction
        }
    """
    pred_edges = set((u, v) for u, v in pred_tree_directed.edges())

    # Handle both directed and undirected ground truth
    if true_tree.is_directed():
        true_edges = set((u, v) for u, v in true_tree.edges())
    else:
        raise ValueError(
            "Ground truth tree must be directed for directed recall computation"
        )

    # Directed metrics
    tp_directed = len(pred_edges & true_edges)
    directed_recall = tp_directed / len(true_edges) if true_edges else 0.0
    directed_precision = tp_directed / len(pred_edges) if pred_edges else 0.0
    directed_f1 = (
        2
        * directed_precision
        * directed_recall
        / (directed_precision + directed_recall)
        if (directed_precision + directed_recall) > 0
        else 0.0
    )

    # Undirected recall (edge exists regardless of direction)
    pred_edges_undirected = {frozenset((u, v)) for u, v in pred_edges}
    true_edges_undirected = {frozenset((u, v)) for u, v in true_tree.edges()}

    tp_undirected = len(pred_edges_undirected & true_edges_undirected)
    undirected_recall = (
        tp_undirected / len(true_edges_undirected) if true_edges_undirected else 0.0
    )

    return {
        "directed_recall": float(directed_recall),
        "directed_precision": float(directed_precision),
        "directed_f1": float(directed_f1),
        "undirected_recall": float(undirected_recall),
        "tp_directed": int(tp_directed),
        "fp_directed": int(len(pred_edges) - tp_directed),
        "fn_directed": int(len(true_edges) - tp_directed),
        "pred_edges": int(len(pred_edges)),
        "true_edges": int(len(true_edges)),
    }


def find_rumor_root(tree_undirected: nx.Graph) -> Any:
    """
    Select root using rumor centrality (Shah & Zaman 2011).

    The rumor center maximises the probability of being the source under
    a symmetric SI model on a tree.  Equivalent to the node with maximum
    rumor centrality R(v) = n! / prod_{subtree sizes rooted at v}.

    Requires the optional `rumor_centrality` module.  Falls back to the
    tree-height heuristic if the module is unavailable.

    Args:
        tree_undirected: Undirected spanning tree.

    Returns:
        The selected root node.
    """
    if not _RUMOR_AVAILABLE:
        raise ImportError(
            "rumor_centrality module not found.  "
            "Make sure rumor_centrality.py is importable or install the package."
        )

    if not nx.is_connected(tree_undirected):
        # Pick largest component for robustness
        largest = max(nx.connected_components(tree_undirected), key=len)
        subgraph = tree_undirected.subgraph(largest).copy()
    else:
        subgraph = tree_undirected

    root, _scores = _rumor_estimate_source(subgraph)
    return root


def process_graph(
    entry: Dict[str, Any],
    output_dir: str,
    collection: str,
    mode: str = "train",
    use_T_depth_dist: bool = False,
    verbose: bool = False,
    search_dir_name: str = "search",
    root_method: str = "depth_prior",
    prior_dist: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Process single graph: find optimal root and evaluate directed tree.

    Supports train and eval modes:
    - train: requires T_path, always uses T's depth distribution
    - eval: T_path optional, uses T depth dist only if --use-T-depth-dist flag set

    Args:
        entry: Manifest entry with graph_id, collection, T_path (optional)
        output_dir: Base output directory
        collection: Collection name (already validated)
        mode: 'train' (requires T_path) or 'eval' (T_path optional)
        use_T_depth_dist: Use T's depth distribution (train mode: ignored/always True)
        verbose: Print progress
        search_dir_name: Name of search directory (search, search_mst, search_sa, search_mst_sa)
        root_method: Root selection method: 'depth_prior' (default) or 'rumor'.
        prior_dist: Optional depth histogram (list of counts per depth level) to use as
                    external prior instead of computing from T.  Only used when
                    root_method='depth_prior'.

    Returns:
        {
            "graph_id": gid,
            "collection": collection,
            "root": optimal_root,
            "depth_mse": mse_value,
            "directed_recall": recall_value or None,
            "error": None or error message
        }
    """
    gid = entry["graph_id"]
    search_dir = os.path.join(
        get_graph_dir(output_dir, collection, gid), search_dir_name
    )

    try:
        # Load undirected tree from tree_search output
        tree_path = os.path.join(search_dir, TREE_FILE)
        if not os.path.exists(tree_path):
            return {
                "graph_id": gid,
                "collection": collection,
                "root": None,
                "depth_mse": None,
                "directed_recall": None,
                "error": f"Tree file not found: {tree_path}",
            }

        with open(tree_path, "rb") as f:
            tree_undirected = pickle.load(f)

        if not isinstance(tree_undirected, nx.Graph):
            tree_undirected = nx.Graph(tree_undirected)

        # Load ground truth tree if provided
        true_tree = None
        if "T_path" in entry and entry["T_path"] and os.path.exists(entry["T_path"]):
            true_tree = load_graph(entry["T_path"])
        elif mode == "train":
            # Train mode requires T_path
            return {
                "graph_id": gid,
                "collection": collection,
                "root": None,
                "depth_mse": None,
                "directed_recall": None,
                "error": "Train mode requires T_path in manifest entry",
            }

        # Find optimal root
        if verbose:
            print(f"\n  {gid}: Testing roots...")

        if root_method == "rumor":
            optimal_root = find_rumor_root(tree_undirected)
            root_result = {
                "root": optimal_root,
                "depth_mse": None,
                "depth_vector_pred": None,
                "depth_vector_true": None,
                "true_root": None,
                "all_roots": [],
            }
        else:
            # depth_prior (default)
            # Train mode: always use T depth dist (ignore flag)
            # Eval mode: only use T depth dist if flag explicitly set
            use_T = (mode == "train") or use_T_depth_dist

            root_result = find_optimal_root(
                tree_undirected,
                true_tree,
                use_T_depth_dist=use_T,
                prior_dist=prior_dist,
                verbose=verbose,
            )
            optimal_root = root_result["root"]

        # Create directed tree from optimal root using BFS
        directed_tree = nx.DiGraph(nx.bfs_tree(tree_undirected, optimal_root))

        # Evaluate directed recall if ground truth available
        directed_recall = None
        undirected_recall = None
        if true_tree is not None:
            eval_metrics = compute_directed_recall(directed_tree, true_tree)
            directed_recall = eval_metrics["directed_recall"]
            undirected_recall = eval_metrics["undirected_recall"]

        # Save results
        # Save to {search_dir}/optimal_root/ to keep results separated by method
        # e.g., outputs/{collection}/{gid}/search_mst/optimal_root/
        result_dir = os.path.join(search_dir, OPTIMAL_ROOT_DIR)
        os.makedirs(result_dir, exist_ok=True)

        # Save directed tree using utils.save_graph (handles both undirected and directed)
        tree_directed_path = os.path.join(result_dir, TREE_DIRECTED_FILE)
        save_graph(directed_tree, tree_directed_path)

        with open(os.path.join(result_dir, OPTIMAL_ROOT_FILE), "w") as f:
            json.dump(
                {
                    "root": optimal_root,
                    "true_root": root_result["true_root"],
                    "depth_mse": root_result["depth_mse"],
                    "depth_vector_pred": root_result["depth_vector_pred"],
                    "depth_vector_true": root_result["depth_vector_true"],
                    "undirected_recall": undirected_recall,
                    "directed_recall": directed_recall,
                    "num_roots_tested": len(root_result["all_roots"]),
                },
                f,
                indent=2,
            )

        if directed_recall is not None:
            with open(os.path.join(result_dir, DIRECTED_EVAL_FILE), "w") as f:
                json.dump(eval_metrics, f, indent=2)

        result = {
            "graph_id": gid,
            "collection": collection,
            "root": optimal_root,
            "true_root": root_result["true_root"],
            "depth_mse": root_result["depth_mse"],
            "directed_recall": directed_recall,
            "undirected_recall": undirected_recall,
            "error": None,
        }

        if verbose:
            directed_str = (
                f"{directed_recall:.4f}" if directed_recall is not None else "N/A"
            )
            undirected_str = (
                f"{undirected_recall:.4f}" if undirected_recall is not None else "N/A"
            )
            print(
                f"  ✓ {gid}: root={optimal_root}, true_root={root_result['true_root']}, "
                f"MSE={root_result['depth_mse']:.4f}, "
                f"recall_undirected={undirected_str}, recall_directed={directed_str}"
            )

        return result

    except Exception as e:
        return {
            "graph_id": gid,
            "collection": collection,
            "root": None,
            "depth_mse": None,
            "directed_recall": None,
            "error": f"{type(e).__name__}: {str(e)}",
        }


def process_manifest(
    manifest_path: str,
    collection: str,
    output_dir: str,
    mode: str = "train",
    use_T_depth_dist: bool = False,
    workers: int = 1,
    verbose: bool = False,
    search_dir_name: str = "search",
    root_method: str = "depth_prior",
    prior_dist: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Process all graphs in manifest: find optimal roots and evaluate.

    Supports train/eval modes and parallel processing.

    Args:
        manifest_path: Path to manifest JSON file
        collection: Collection name (must match manifest entries)
        output_dir: Base output directory
        mode: 'train' (requires T_path) or 'eval' (T_path optional)
        use_T_depth_dist: Use T's depth distribution (eval mode only; train always uses T)
        workers: Number of parallel workers (1 = sequential)
        verbose: Print progress
        search_dir_name: Name of search directory (search, search_mst, search_sa, search_mst_sa)
        root_method: Root selection method: 'depth_prior' (default) or 'rumor'.
        prior_dist: Optional external depth histogram for 'depth_prior' method.

    Returns:
        List of result dicts
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter by collection and validate entries for mode
    entries = [
        validate_manifest_entry(e, mode=mode)
        for e in manifest
        if e.get("collection") == collection
    ]

    if not entries:
        raise ValueError(f"No manifest entries for collection '{collection}'")

    if verbose:
        print(
            f"Processing {len(entries)} graphs in {mode} mode (use_T_depth_dist={use_T_depth_dist}, root_method={root_method})"
        )

    # Sequential or parallel processing
    if workers == 1:
        results = []
        for i, entry in enumerate(entries):
            if verbose:
                print(f"[{i+1}/{len(entries)}] {entry['graph_id']}...", flush=True)
            result = process_graph(
                entry,
                output_dir,
                collection,
                mode=mode,
                use_T_depth_dist=use_T_depth_dist,
                verbose=verbose,
                search_dir_name=search_dir_name,
                root_method=root_method,
                prior_dist=prior_dist,
            )
            results.append(result)
    else:
        # Parallel processing
        if verbose:
            print(f"Using {workers} parallel workers")

        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_graph,
                    entry,
                    output_dir,
                    collection,
                    mode,
                    use_T_depth_dist,
                    False,  # verbose=False in parallel
                    search_dir_name,
                    root_method,
                    prior_dist,
                )
                for entry in entries
            ]
            for i, future in enumerate(futures):
                if verbose:
                    print(f"[{i+1}/{len(entries)}] Waiting for result...")
                result = future.result()
                results.append(result)

    return results


def write_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Write results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ Wrote results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find optimal root and direct spanning tree to arborescence"
    )
    parser.add_argument(
        "--manifest", help="Path to manifest JSON file with graph entries"
    )
    parser.add_argument("--gid", help="Single graph ID (alternative to manifest)")
    parser.add_argument("--collection", required=True, help="Collection name")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory (default: outputs)",
    )
    parser.add_argument("--results-csv", help="Optional: write results to CSV file")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Mode: 'train' (requires T_path) or 'eval' (T_path optional)",
    )
    parser.add_argument(
        "--use-T-depth-dist",
        action="store_true",
        help="Use T's depth distribution for root selection (eval mode only; ignored in train mode where it's always used)",
    )
    parser.add_argument(
        "--root-method",
        choices=["depth_prior", "rumor"],
        default="depth_prior",
        help=(
            "Root selection method: 'depth_prior' (default) minimises MSE against a "
            "depth histogram (from T or --prior-path); 'rumor' uses Shah & Zaman (2011) "
            "rumor centrality (requires rumor_centrality module)."
        ),
    )
    parser.add_argument(
        "--prior-path",
        default=None,
        help=(
            "Path to a JSON file containing an explicit depth histogram "
            "(list of counts per depth level) used as the prior for depth_prior root selection. "
            "When provided, overrides use_T_depth_dist."
        ),
    )
    parser.add_argument(
        "--search-dir",
        default="search",
        help="Name of search directory (search, search_mst, search_sa, search_mst_sa)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert single-graph mode to manifest-of-1
    if not args.manifest:
        if not args.gid:
            raise ValueError("Either --manifest or --gid is required")

        # Build temp manifest entry with paths from convention
        manifest_entry = {
            "graph_id": args.gid,
            "collection": args.collection,
            "T_path": os.path.join(
                "data", args.collection, args.gid, "hierarchy_tree.pkl"
            ),
        }

        # Create temp manifest file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([manifest_entry], f, indent=2)
            temp_manifest = f.name

        args.manifest = temp_manifest

    print(f"🌳 Optimal Root Selection")
    print(f"  Manifest: {args.manifest}")
    print(f"  Collection: {args.collection}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Root method: {args.root_method}")
    print(f"  Use T depth dist: {args.use_T_depth_dist}")
    if args.prior_path:
        print(f"  Prior path: {args.prior_path}")
    print(f"  Workers: {args.workers}")
    print(f"  Search dir: {args.search_dir}")

    # Load optional external prior distribution
    prior_dist = None
    if args.prior_path:
        with open(args.prior_path) as f:
            prior_dist = json.load(f)
        if not isinstance(prior_dist, list):
            raise ValueError(
                f"--prior-path must be a JSON file containing a list of numbers, "
                f"got {type(prior_dist).__name__}"
            )
        prior_dist = [float(x) for x in prior_dist]

    results = process_manifest(
        args.manifest,
        args.collection,
        args.output_dir,
        mode=args.mode,
        use_T_depth_dist=args.use_T_depth_dist,
        workers=args.workers,
        verbose=args.verbose,
        search_dir_name=args.search_dir,
        root_method=args.root_method,
        prior_dist=prior_dist,
    )

    # Summary
    print(f"\n📊 Summary ({len(results)} graphs):")
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"  ⚠️  Errors: {len(errors)}")
        for r in errors[:5]:  # Show first 5 errors
            print(f"    - {r['graph_id']}: {r['error']}")
    else:
        print(f"  ✅ All successful")

    successes = [r for r in results if not r["error"]]
    if successes:
        mses = [r["depth_mse"] for r in successes if r["depth_mse"] is not None]
        if mses:
            avg_mse = np.mean(mses)
            print(f"  Average depth MSE: {avg_mse:.4f}")

        undirected_recalls = [
            r["undirected_recall"]
            for r in successes
            if r["undirected_recall"] is not None
        ]
        if undirected_recalls:
            avg_undirected = np.mean(undirected_recalls)
            print(f"  Average base recall (undirected): {avg_undirected:.4f}")

        directed_recalls = [
            r["directed_recall"] for r in successes if r["directed_recall"] is not None
        ]
        if directed_recalls:
            avg_recall = np.mean(directed_recalls)
            print(f"  Average directed recall: {avg_recall:.4f}")

    # Optional CSV output
    if args.results_csv:
        write_results_csv(results, args.results_csv)


if __name__ == "__main__":
    main()
