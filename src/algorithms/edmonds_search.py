"""
edmonds_search: Edmonds' minimum spanning arborescence as a single-step
hierarchy reconstruction method.

Replaces the two-step MST + root-finding pipeline with a single call to
Edmonds' algorithm (Chu-Liu / Edmonds').  The key difference:

    MST pipeline
        undirected graph → MST (ignores edge direction) → BFS-root with a
        separately chosen root node → directed arborescence

    Edmonds pipeline
        directed graph → minimum spanning arborescence in one step →
        structure AND root are jointly optimised

For wiki the entity_graph is a DiGraph, so edge directionality (hyperlink
direction) is preserved and scores are genuinely asymmetric (diff features
are signed, not absolute-valued).

Scores are loaded as directed (u, v) tuples — NOT converted to frozensets —
so score(u→v) ≠ score(v→u) when the model learned asymmetric features.

Edge weights: w(u,v) = -log(score(u,v) + 1e-9)  →  Edmonds minimises total
weight  →  equivalent to maximising the product of scores  (MLE arborescence).

Output per graph
    outputs/{collection}/{graph_id}/edmonds/arborescence.pkl  (nx.DiGraph)
    outputs/{collection}/{graph_id}/edmonds/edmonds_results.json

Isolation guarantee: this module imports nothing from the rest of src/
except src.utils.load_graph / save_graph.  It does not touch any existing
output directories.
"""

import argparse
import json
import math
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.utils import load_graph, save_graph

# (deque removed — depth metrics dropped)

# ---------------------------------------------------------------------------
# Path conventions (isolated — no overlap with search*, optimal_root dirs)
# ---------------------------------------------------------------------------

EDMONDS_DIR = "edmonds"
ARBORESCENCE_FILE = "arborescence.pkl"
RESULTS_FILE = "edmonds_results.json"


def _graph_dir(output_dir: str, collection: str, gid: str) -> str:
    return os.path.join(output_dir, collection, gid)


# ---------------------------------------------------------------------------
# Score loading — preserves direction
# ---------------------------------------------------------------------------


def load_directed_scores(score_path: str) -> Dict[Tuple, float]:
    """
    Load edge scores from CSV as directed (u, v) → float mapping.

    Unlike the MST pipeline this does NOT convert to frozensets.
    Columns expected: source, target, score.
    """
    df = pd.read_csv(score_path)
    required = {"source", "target", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"score CSV {score_path} missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return {(row.source, row.target): float(row.score) for row in df.itertuples()}


# ---------------------------------------------------------------------------
# Build directed weighted graph
# ---------------------------------------------------------------------------


def build_directed_weighted_graph(
    G_directed: nx.DiGraph,
    directed_scores: Dict[Tuple, float],
) -> Tuple[nx.DiGraph, int]:
    """
    Build a weighted DiGraph for Edmonds' algorithm.

    Weight = -log(score + 1e-9) so lower weight = better edge (Edmonds
    minimises total weight, equivalent to MLE arborescence).

    Returns
    -------
    D_weighted : nx.DiGraph with 'weight' attribute on edges
    n_missing  : number of edges in G that had no score (assigned default 0)
    """
    D = nx.DiGraph()
    D.add_nodes_from(G_directed.nodes())

    n_missing = 0
    for u, v in G_directed.edges():
        score = directed_scores.get((u, v))
        if score is None:
            # No score for this directed edge — assign weight 0 (score=1.0)
            n_missing += 1
            weight = 0.0
        else:
            weight = -math.log(max(score, 1e-9))
        D.add_edge(u, v, weight=weight)

    return D, n_missing


# ---------------------------------------------------------------------------
# Edmonds' algorithm wrapper
# ---------------------------------------------------------------------------


def run_edmonds(
    D_weighted: nx.DiGraph,
) -> Tuple[Optional[nx.DiGraph], Optional[str]]:
    """
    Run Edmonds' minimum spanning arborescence on D_weighted.

    Returns
    -------
    arborescence : nx.DiGraph that is a valid arborescence, or None on failure
    error        : None on success, error message string on failure
    """
    try:
        arborescence = nx.minimum_spanning_arborescence(D_weighted, attr="weight")
    except nx.exception.NetworkXException as exc:
        return None, f"NetworkXException: {exc}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"

    return arborescence, None


# ---------------------------------------------------------------------------
# Metric helpers (self-contained, no imports from optimal_root)
# ---------------------------------------------------------------------------


def compute_root_distance(
    pred_root,
    true_tree: nx.DiGraph,
    true_root,
) -> Optional[int]:
    """
    Shortest-path hop distance from pred_root to true_root in the
    undirected version of true_tree.  Returns None if pred_root is not
    a node in true_tree.
    """
    if pred_root not in true_tree.nodes():
        return None
    T_ud = true_tree.to_undirected()
    try:
        return nx.shortest_path_length(T_ud, pred_root, true_root)
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_arborescence(
    arborescence: nx.DiGraph,
    true_tree: nx.DiGraph,
) -> Dict[str, Any]:
    """
    Compute all quality metrics for a predicted arborescence vs ground truth.

    Metrics
    -------
    directed_recall      TP / |true_edges|
    directed_precision   TP / |pred_edges|
    directed_f1
    undirected_recall    edge-set recall ignoring direction
    tp / fp / fn         counts
    root_pred            node id with in_degree 0 in arborescence
    root_true            node id with in_degree 0 in true_tree
    root_accuracy        1 if root_pred == root_true else 0
    root_distance        hop distance in T (undirected) between pred and true root
    """
    # ---- roots --------------------------------------------------------
    pred_roots = [n for n in arborescence.nodes() if arborescence.in_degree(n) == 0]
    true_roots = [n for n in true_tree.nodes() if true_tree.in_degree(n) == 0]

    root_pred = pred_roots[0] if pred_roots else None
    root_true = true_roots[0] if true_roots else None

    root_accuracy = int(root_pred is not None and root_pred == root_true)
    root_distance = (
        compute_root_distance(root_pred, true_tree, root_true)
        if (root_pred is not None and root_true is not None)
        else None
    )

    # ---- edge metrics -------------------------------------------------
    pred_edges = set(arborescence.edges())
    true_edges = set(true_tree.edges())

    tp = len(pred_edges & true_edges)
    fp = len(pred_edges) - tp
    fn = len(true_edges) - tp

    directed_recall = tp / len(true_edges) if true_edges else 0.0
    directed_precision = tp / len(pred_edges) if pred_edges else 0.0
    directed_f1 = (
        2
        * directed_precision
        * directed_recall
        / (directed_precision + directed_recall)
        if (directed_precision + directed_recall) > 0
        else 0.0
    )

    pred_ud = {frozenset(e) for e in pred_edges}
    true_ud = {frozenset(e) for e in true_edges}
    tp_ud = len(pred_ud & true_ud)
    undirected_recall = tp_ud / len(true_ud) if true_ud else 0.0

    return {
        "directed_recall": float(directed_recall),
        "directed_precision": float(directed_precision),
        "directed_f1": float(directed_f1),
        "undirected_recall": float(undirected_recall),
        "tp_directed": int(tp),
        "fp_directed": int(fp),
        "fn_directed": int(fn),
        "pred_edges": int(len(pred_edges)),
        "true_edges": int(len(true_edges)),
        "root_pred": root_pred,
        "root_true": root_true,
        "root_accuracy": int(root_accuracy),
        "root_distance": int(root_distance) if root_distance is not None else None,
    }


# ---------------------------------------------------------------------------
# Per-graph processing (top-level function — must be picklable for workers)
# ---------------------------------------------------------------------------


def process_graph(
    entry: Dict[str, Any],
    output_dir: str,
    collection: str,
    verbose: bool = False,
    directed_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run Edmonds' arborescence pipeline for a single graph.

    Saves
    -----
    outputs/{collection}/{gid}/edmonds/arborescence.pkl
    outputs/{collection}/{gid}/edmonds/edmonds_results.json

    Returns
    -------
    Result dict with all metrics + error field (None on success).
    """
    gid = entry["graph_id"]

    base_result = {
        "graph_id": gid,
        "collection": collection,
        "split": entry.get("split", "unknown"),
        "directed_recall": None,
        "directed_precision": None,
        "directed_f1": None,
        "undirected_recall": None,
        "tp_directed": None,
        "fp_directed": None,
        "fn_directed": None,
        "pred_edges": None,
        "true_edges": None,
        "root_pred": None,
        "root_true": None,
        "root_accuracy": None,
        "root_distance": None,
        "n_nodes": None,
        "n_scored_edges": None,
        "n_missing_scores": None,
        "error": None,
    }

    try:
        # ---- load inputs -----------------------------------------------
        g_path = entry.get("G_path", "")
        t_path = entry.get("T_path", "")
        score_path = entry.get("score_path", "")

        for label, path in [
            ("G_path", g_path),
            ("T_path", t_path),
            ("score_path", score_path),
        ]:
            if not path or not os.path.exists(path):
                base_result["error"] = f"Missing {label}: {path!r}"
                return base_result

        G = load_graph(g_path)
        T = load_graph(t_path)

        if not G.is_directed():
            if directed_mode:
                # Bidirected mode: score CSV already has both directions;
                # convert G so build_directed_weighted_graph sees all edges.
                G = G.to_directed()
            else:
                base_result["error"] = (
                    f"Entity graph is undirected (type={type(G).__name__}). "
                    "Edmonds' directed-only strategy requires a DiGraph. "
                    "Pass --directed-mode for undirected datasets (microbiome, memetracker)."
                )
                return base_result

        directed_scores = load_directed_scores(score_path)

        # ---- build weighted digraph ------------------------------------
        D_weighted, n_missing = build_directed_weighted_graph(G, directed_scores)

        base_result["n_nodes"] = G.number_of_nodes()
        base_result["n_scored_edges"] = len(directed_scores)
        base_result["n_missing_scores"] = n_missing

        if verbose:
            print(
                f"  {gid}: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges, "
                f"{n_missing} missing scores"
            )

        # ---- Edmonds' --------------------------------------------------
        arborescence, err = run_edmonds(D_weighted)

        if err is not None:
            base_result["error"] = err
            return base_result

        # ---- evaluate --------------------------------------------------
        if not T.is_directed():
            base_result["error"] = (
                "Ground-truth tree T is undirected — cannot compute directed recall."
            )
            return base_result

        metrics = evaluate_arborescence(arborescence, T)
        base_result.update(metrics)

        if verbose:
            print(
                f"  {gid}: directed_recall={metrics['directed_recall']:.3f} "
                f"root_acc={metrics['root_accuracy']} "
                f"root_dist={metrics['root_distance']}"
            )

        # ---- save results ----------------------------------------------
        out_dir = os.path.join(_graph_dir(output_dir, collection, gid), EDMONDS_DIR)
        os.makedirs(out_dir, exist_ok=True)

        save_graph(arborescence, os.path.join(out_dir, ARBORESCENCE_FILE))

        with open(os.path.join(out_dir, RESULTS_FILE), "w") as f:
            json.dump(base_result, f, indent=2, default=str)

    except Exception as exc:
        base_result["error"] = f"{type(exc).__name__}: {exc}"

    return base_result


# ---------------------------------------------------------------------------
# Manifest-level processing
# ---------------------------------------------------------------------------


def process_manifest(
    manifest_path: str,
    collection: str,
    output_dir: str,
    workers: int = 1,
    verbose: bool = False,
    directed_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process all graphs in manifest with optional parallelism.

    Returns list of result dicts (one per graph).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [e for e in manifest if e.get("collection") == collection]
    if not entries:
        raise ValueError(f"No entries for collection '{collection}' in {manifest_path}")

    if verbose:
        print(
            f"Processing {len(entries)} graphs "
            f"(collection={collection}, workers={workers})"
        )

    if workers == 1:
        results = []
        for i, entry in enumerate(entries):
            if verbose:
                print(f"[{i+1}/{len(entries)}] {entry['graph_id']}", flush=True)
            results.append(process_graph(entry, output_dir, collection, verbose, directed_mode))
        return results

    results_map: Dict[str, Dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_graph, entry, output_dir, collection, False, directed_mode): entry
            for entry in entries
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            entry = futures[future]
            gid = entry["graph_id"]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "graph_id": gid,
                    "collection": collection,
                    "error": f"Worker exception: {exc}",
                }
            results_map[gid] = result
            if verbose:
                err = result.get("error")
                tag = (
                    f"ERROR: {err}"
                    if err
                    else f"dr={result.get('directed_recall', '?'):.3f}"
                )
                print(f"  [{done}/{len(entries)}] {gid} — {tag}", flush=True)

    # Preserve original manifest order
    gid_order = [e["graph_id"] for e in entries]
    return [results_map[gid] for gid in gid_order if gid in results_map]


def write_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Results written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Edmonds' minimum spanning arborescence — single-step hierarchy "
            "reconstruction using directed edge scores."
        )
    )
    p.add_argument(
        "--manifest",
        help="Path to manifest JSON (list of graph entries with G_path, T_path, score_path)",
    )
    p.add_argument("--gid", help="Single graph_id (alternative to --manifest)")
    p.add_argument("--collection", required=True, help="Dataset collection name")
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory [default: outputs]",
    )
    p.add_argument(
        "--results-csv",
        default=None,
        help="Optional path to write aggregated results CSV",
    )
    p.add_argument(
        "-w", "--workers", type=int, default=1, help="Parallel workers [default: 1]"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--directed-mode",
        dest="directed_mode",
        action="store_true",
        default=False,
        help=(
            "Convert undirected G to bidirected before running Edmonds. "
            "Use for datasets whose score CSV was produced with --directed-mode "
            "(microbiome, memetracker). The score CSV must contain both (u,v) "
            "and (v,u) rows so Edmonds can pick the correct direction."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if not args.manifest and not args.gid:
        print("ERROR: provide --manifest or --gid", file=sys.stderr)
        sys.exit(1)

    if args.gid and not args.manifest:
        # Build single-entry temp manifest
        entry = {
            "graph_id": args.gid,
            "collection": args.collection,
            "G_path": os.path.join(
                "data", args.collection, args.gid, "entity_graph.pkl"
            ),
            "T_path": os.path.join(
                "data", args.collection, args.gid, "hierarchy_tree.pkl"
            ),
            "score_path": os.path.join(
                "outputs", args.collection, args.gid, "scores", "edge_scores.csv"
            ),
            "split": "eval",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump([entry], tmp, indent=2)
            args.manifest = tmp.name

    print("🌿 Edmonds Arborescence Search")
    print(f"  Manifest   : {args.manifest}")
    print(f"  Collection : {args.collection}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Workers    : {args.workers}")

    results = process_manifest(
        args.manifest,
        args.collection,
        args.output_dir,
        workers=args.workers,
        verbose=args.verbose,
        directed_mode=args.directed_mode,
    )

    # Summary
    errors = [r for r in results if r.get("error")]
    successes = [r for r in results if not r.get("error")]

    print(f"\n📊 Summary ({len(results)} graphs):")
    if errors:
        print(f"  ⚠️  Errors ({len(errors)}):")
        for r in errors:
            print(f"     {r['graph_id']}: {r['error']}")

    if successes:
        dr = [
            r["directed_recall"] for r in successes if r["directed_recall"] is not None
        ]
        ur = [
            r["undirected_recall"]
            for r in successes
            if r["undirected_recall"] is not None
        ]
        ra = [r["root_accuracy"] for r in successes if r["root_accuracy"] is not None]
        rd = [r["root_distance"] for r in successes if r["root_distance"] is not None]

        def _fmt(vals, fmt=".3f"):
            if not vals:
                return "n/a"
            return f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}"

        print(f"  ✅ Successful: {len(successes)}")
        print(f"  directed_recall    : {_fmt(dr)}")
        print(f"  undirected_recall  : {_fmt(ur)}")
        print(f"  root_accuracy      : {_fmt(ra, '.3f')}")
        print(f"  root_distance      : {_fmt(rd, '.2f')}")

    if args.results_csv:
        write_results_csv(results, args.results_csv)


if __name__ == "__main__":
    main()
