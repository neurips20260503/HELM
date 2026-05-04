import pickle
import os
import sys
import multiprocessing
import argparse
import json
import copy
import tempfile
import optuna
import numpy as np
import logging
import optuna.storages.journal
import matplotlib.pyplot as plt
import src.utils as utils
import src.algorithms.tree_search as tree_search_mod
from datetime import datetime
import concurrent.futures


# Setup Logging is handled per-run via utils.setup_logger (do not use global basicConfig)

# Global storage for graph, node scores, and edge weights
TRIAL_TREES_DIR = None
LOGGER_NAME = "optuna_tree_search"
GLOBAL_MAX_ITER = None
GLOBAL_VERBOSITY = 1
GLOBAL_SA_VERBOSITY = 1  # separate control for SA module logging
PRECOMPUTED_DIR = None
PRECOMPUTED_CACHE = {}
STUDY_PATIENCE = None
objective = None
GRAPH_WORKERS = 10  # Parallel graph processes per trial
EARLY_PRUNE_THRESHOLD = 0.6  # If mean TPR < this after min_graphs_for_signal, prune
EARLY_PRUNE_MIN_GRAPHS = (
    2  # Wait for this many graphs before checking pruning threshold
)


def _run_graph_in_process(entry, trial_number, config):
    """Run SA for one graph in a separate process (called by ProcessPoolExecutor).

    Args:
        entry: Manifest entry with graph_id
        trial_number: Optuna trial number
        config: SA configuration dict

    Returns:
        tuple: (tpr, gid) for identification

    Note: Imports are at module level to be available in spawned process
    """
    gid = entry.get("graph_id")

    # Load precomputed data from disk
    precomp_path = os.path.join(globals().get("PRECOMPUTED_DIR", "."), f"{gid}.pkl")
    if not os.path.exists(precomp_path):
        raise RuntimeError(f"Precomputed file not found: {precomp_path}")
    with open(precomp_path, "rb") as pf:
        pre = pickle.load(pf)

    # Deep copy to avoid interference (these ARE needed for each process)
    G = copy.deepcopy(pre.get("G"))
    T = copy.deepcopy(pre.get("T"))
    S_0 = copy.deepcopy(pre.get("initial_tree"))
    edge_scores = copy.deepcopy(pre.get("edge_scores"))

    # Setup trial directory
    trial_dir = os.path.join(
        globals().get("TRIAL_TREES_DIR", "trial_trees"), f"trial_{trial_number}", gid
    )
    os.makedirs(trial_dir, exist_ok=True)

    local_config = dict(config)
    local_config["trial_dir"] = trial_dir
    local_config["verbosity"] = globals().get("GLOBAL_SA_VERBOSITY", 1)

    # Run SA (CPU-intensive part)
    res = tree_search_mod.simulated_annealing(
        G, S_0, T, edge_scores, local_config, return_stats=True
    )

    # Extract TPR
    if isinstance(res, dict):
        tpr = res.get("run_stats", {}).get("tpr", None)
        S = res.get("tree")
    else:
        S = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else None
        tpr = None

    # Fallback TPR calculation
    if tpr is None:
        if S is not None and T is not None:
            final_TP = sum(
                1
                for u, v in S.edges()
                if frozenset((u, v)) in set(frozenset((a, b)) for a, b in T.edges())
            )
            total_true = len(set(frozenset((u, v)) for u, v in T.edges()))
            tpr = final_TP / total_true if total_true > 0 else 0.0
        else:
            tpr = 0.0

    # Save result
    rec = {"tree": S, "tpr": tpr, "params": local_config}
    result_path = os.path.join(trial_dir, f"trial_{trial_number}_{gid}.pkl")
    with open(result_path, "wb") as fh:
        pickle.dump(rec, fh)

    return float(tpr), gid


def _precompute_one(args_tuple):
    """Worker-friendly precompute helper for ProcessPoolExecutor.

    args_tuple: (entry, precomp_dir)
    Returns precomputed dict p on success.
    """
    entry, precomp_dir = args_tuple
    gid = entry.get("graph_id")
    G_path = entry.get("G_path")
    T_path = entry.get("T_path")
    score_path = entry.get("score_path")
    positive_edges_path = entry.get("positive_edges_path")
    # perform the actual precompute (uses module tree_search_mod)
    # CRITICAL: Pass init_method and mode='train' (Optuna runs on train manifest)
    p = tree_search_mod.precompute_for_manifest_entry(
        G_path,
        T_path,
        score_path,
        positive_edges_path,
        mode="train",  # Optuna runs on TRAIN manifest
        init_method=globals().get("GLOBAL_INIT_METHOD", "positive_edges"),
        skip_score_loading=globals().get("GLOBAL_SKIP_SCORE_LOADING", False),
    )
    p["graph_id"] = gid

    # atomic write using a unique temp file in the target dir
    target_path = os.path.join(precomp_dir, f"{gid}.pkl")
    fd, tmp_path = tempfile.mkstemp(dir=precomp_dir, suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as pf:
            pickle.dump(p, pf)
        os.replace(tmp_path, target_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    return p


def objective_manifest_factory(manifest_entries, precomputed_map):
    """Return an objective function that evaluates a trial across all manifest graphs.

    The returned objective(trial) will run simulated_annealing on each graph for the
    trial's parameters and return the mean TPR across manifest entries.
    """

    def objective_manifest(trial):
        # construct SA config from trial
        # NOTE: degree loss was changed - use wider range to explore new optimum

        # Build loss_weights dict conditionally based on skip_score_loading flag
        loss_weights = {
            "community": trial.suggest_float("community", 1, 7),
            "diversity": trial.suggest_float("diversity", 1, 7),
            "degree": trial.suggest_float("degree", 1, 7),
            "shortcut": trial.suggest_float("shortcut", 1, 7),
        }
        # Only suggest score weight if we're actually using edge scores
        if not globals().get("GLOBAL_SKIP_SCORE_LOADING", False):
            loss_weights["score"] = trial.suggest_float("score", 1, 7)
        else:
            loss_weights["score"] = 0.0  # Explicitly set to 0 for no_scores variant

        config = {
            "initial_temp": trial.suggest_float("initial_temp", 0.01, 0.5),
            "cooling_rate": trial.suggest_float("cooling_rate", 0.9999, 0.9999999),
            "max_iter": int(GLOBAL_MAX_ITER) if GLOBAL_MAX_ITER else 100000000,
            "init_method": globals().get("GLOBAL_INIT_METHOD", "positive_edges"),
            "loss_weights": loss_weights,
            "resolution": trial.suggest_float("resolution", 0.1, 3),
            "stagnation_threshold": trial.suggest_int(
                "stagnation_threshold", 5000, 75000, step=5000
            ),
            "bold_stagnation_threshold": trial.suggest_int(
                "bold_stagnation_threshold", 5000, 75000, step=5000
            ),
            "leiden_n_iterations": trial.suggest_int("leiden_n_iterations", 2, 10),
            "min_bold_prob": trial.suggest_float("min_bold_prob", 0.01, 0.2),
            "bold_prob_scale": trial.suggest_float("bold_prob_scale", 0.05, 1.0),
            "bold_multiplier": trial.suggest_float("bold_multiplier", 1.0, 5.0),
            "seed": globals().get("GLOBAL_SEED", None),
        }

        logger = logging.getLogger(LOGGER_NAME)

        # Run all graphs in parallel using ProcessPoolExecutor
        max_workers = min(len(manifest_entries), GRAPH_WORKERS)
        logger.info(
            f"Trial {trial.number}: Processing {len(manifest_entries)} graphs in parallel "
            f"(max_workers={max_workers})"
        )

        tprs = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all graphs at once
            futures = {
                executor.submit(
                    _run_graph_in_process, entry, trial.number, config
                ): entry
                for entry in manifest_entries
            }

            # Collect results as they complete (allows early pruning)
            for future in concurrent.futures.as_completed(futures):
                try:
                    tpr, gid = future.result()
                    tprs.append(tpr)
                    logger.debug(
                        f"Trial {trial.number}: Graph {gid} completed with TPR={tpr:.4f}"
                    )

                    # Report intermediate value to Optuna (enables learning from pruned trials)
                    running_mean = np.mean(tprs)
                    trial.report(running_mean, step=len(tprs))

                    # Early pruning: if we have enough results and mean is catastrophic, cancel remaining
                    if len(tprs) >= EARLY_PRUNE_MIN_GRAPHS:
                        if running_mean < EARLY_PRUNE_THRESHOLD:
                            logger.warning(
                                f"Trial {trial.number}: Early pruning triggered after {len(tprs)} graphs. "
                                f"Mean TPR={running_mean:.4f} < threshold={EARLY_PRUNE_THRESHOLD}. "
                                f"Cancelling remaining {len(futures)-len(tprs)} graphs."
                            )
                            # Cancel all pending futures
                            cancelled = 0
                            for f in futures:
                                if f.cancel():
                                    cancelled += 1
                            logger.warning(
                                f"Trial {trial.number}: Cancelled {cancelled} remaining graphs"
                            )
                            # Raise pruned exception
                            raise optuna.exceptions.TrialPruned()
                except concurrent.futures.CancelledError:
                    # This future was cancelled, skip it
                    logger.debug(
                        f"Trial {trial.number}: Graph processing was cancelled"
                    )
                    pass
                except optuna.exceptions.TrialPruned:
                    # Re-raise prune exception
                    raise
                except Exception as e:
                    entry = futures[future]
                    logger.exception(
                        f"Trial {trial.number}: Graph {entry.get('graph_id')} failed"
                    )
                    raise

        mean_score = float(np.mean(tprs)) if tprs else 0.0
        logger.warning(
            f"Trial {trial.number}: Completed {len(tprs)}/{len(manifest_entries)} graphs, "
            f"mean TPR = {mean_score:.4f}"
        )

        # Log trial hyperparameters and result
        try:
            logger.info(
                f"Trial {trial.number}: params={trial.params}, mean TPR = {mean_score}"
            )
        except Exception:
            logger.info(f"Trial {trial.number}: mean TPR = {mean_score}")

        return mean_score

    return objective_manifest


def run_worker(
    worker_id: int,
    trials_per_worker: int,
    storage_file: str,
    study_name: str,
    logs_dir: str,
    precomputed_dir: str = None,
):
    # Setup per-worker logger instead of dup2'ing stdout/stderr
    os.makedirs(logs_dir, exist_ok=True)
    # trial dir is expected to be provided via module-level TRIAL_TREES_DIR

    # Do not assume `objective` exists in worker (spawn start method).
    # We will reconstruct it from the precomputed_dir's manifest.json below if needed.

    # pick up verbosity if set in module global
    worker_level = logging.INFO
    try:
        worker_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}[
            GLOBAL_VERBOSITY
        ]
    except Exception:
        worker_level = logging.INFO
    worker_log = utils.setup_worker_logger(
        worker_id, logs_dir, name=LOGGER_NAME, level=worker_level
    )
    # Ensure module logger also writes to this worker's log file so calls to
    # `logging.getLogger("optuna_tree_search").info(...)` inside the objective
    # are captured in the per-worker log.
    worker_log.info(f"Worker {worker_id} starting (trials={trials_per_worker})")
    base_logger = logging.getLogger(LOGGER_NAME)
    worker_log_path = os.path.join(logs_dir, f"worker_{worker_id}.log")
    # add a file handler to the base logger if not already present
    existing_paths = [getattr(h, "baseFilename", None) for h in base_logger.handlers]
    if worker_log_path not in existing_paths:
        fh = logging.FileHandler(worker_log_path)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        base_logger.addHandler(fh)
    try:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage_file)
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
        # if a precomputed_dir is given, set module-level PRECOMPUTED_DIR so
        # the objective can lazy-load per-graph pickles in this worker process
        if precomputed_dir:
            globals()["PRECOMPUTED_DIR"] = precomputed_dir
            worker_log.info(f"Worker {worker_id} set PRECOMPUTED_DIR={precomputed_dir}")
        # Reconstruct the manifest-aware objective in spawn-mode workers if missing.
        if globals().get("objective") is None:
            try:
                manifest_path = os.path.join(precomputed_dir, "manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r") as mf:
                        entries = json.load(mf)
                else:
                    # Fallback: derive entries from available pickles
                    entries = []
                    for fn in os.listdir(precomputed_dir):
                        if fn.endswith(".pkl"):
                            gid = os.path.splitext(fn)[0]
                            entries.append({"graph_id": gid})
                # Build objective using lazy on-disk loading (precomputed_map=None)
                obj = objective_manifest_factory(entries, None)
                globals()["objective"] = obj
                worker_log.info(
                    "Reconstructed objective in worker from precomputed_dir"
                )
            except Exception:
                worker_log.exception("Failed to reconstruct objective in worker")
                raise
        # rely on module-level TRIAL_TREES_DIR configured in main
        # Optionally attach a study-level early stopping callback (patience in trials)
        callbacks = None
        try:
            patience = int(globals().get("STUDY_PATIENCE", 0) or 0)
        except Exception:
            patience = 0
        if patience and patience > 0:
            # closure maintaining best value and no-improve counter per worker
            state = {"best": None, "no_improve": 0}

            def _stop_callback(study_obj, trial_obj):
                try:
                    val = trial_obj.value
                    if val is None:
                        return
                    if state["best"] is None or val > state["best"]:
                        state["best"] = val
                        state["no_improve"] = 0
                    else:
                        state["no_improve"] += 1
                        if state["no_improve"] >= patience:
                            worker_log.info(
                                f"Study patience {patience} reached (no improvement); stopping study."
                            )
                            study_obj.stop()
                except Exception:
                    worker_log.exception("Error in study early-stop callback")

            callbacks = [_stop_callback]

        if callbacks:
            study.optimize(
                objective, n_trials=trials_per_worker, n_jobs=1, callbacks=callbacks
            )
        else:
            study.optimize(objective, n_trials=trials_per_worker, n_jobs=1)
        worker_log.info(f"Worker {worker_id} finished successfully.")
    except Exception:
        worker_log.exception(f"Worker {worker_id} failed during optimization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna hyperparameter search for the tree-search simulated annealing. "
            "Verbosity controls logging and diagnostics: 0=errors+summary, "
            "1=info+periodic summaries, 2=debug+detailed (writes per-trial CSV timeseries)."
        )
    )
    parser.add_argument("--G", required=False, help="Path to entity graph pkl")
    parser.add_argument("--T", required=False, help="Path to true tree pkl")
    parser.add_argument("--scores", required=False, help="Path to edge_scores.csv")
    parser.add_argument(
        "--positive",
        required=False,
        help="Path to positive_edges.pkl (graph)",
    )
    parser.add_argument(
        "--gid",
        required=False,
        help="Graph ID for single-graph mode (used in temp manifest)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest JSON to run collection-mode optimization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for reproducibility (per-trial seed = base + trial.number)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection id to filter manifest entries (if manifest provided)",
    )
    parser.add_argument(
        "--out",
        required=False,
        default=None,
        help="Output directory for trial artifacts (defaults to outputs/{collection}/optuna/{timestamp})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing --out directory if present (use with care)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1, help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--trials-per-worker", type=int, default=50, help="Trials per worker"
    )
    parser.add_argument(
        "--graph-workers",
        type=int,
        default=10,
        help="Number of parallel processes for running graphs within each trial (default: 8)",
    )
    parser.add_argument(
        "--early-prune-threshold",
        type=float,
        default=0.15,
        help="If mean TPR < this after early_prune_min_graphs, prune trial early (default: 0.6)",
    )
    parser.add_argument(
        "--early-prune-min-graphs",
        type=int,
        default=1,
        help="Wait for this many graphs before checking early pruning threshold (default: 2)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna journal DB file path (optional)",
    )
    parser.add_argument("--study-name", type=str, default=LOGGER_NAME)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override SA max_iter (useful for quick smoke runs)",
    )
    parser.add_argument(
        "--init-method",
        type=str,
        choices=["positive_edges", "mst", "empty"],
        default="positive_edges",
        help="Initial tree construction method: positive_edges (default), mst (MST with -log(score) weights, maximum-likelihood tree), empty (arbitrary union find tree)",
    )
    parser.add_argument(
        "--skip-score-loading",
        action="store_true",
        help="Skip loading edge scores entirely (for pure SA without scores). Avoids I/O and score computation overhead.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help=(
            "Optuna verbosity level. 0=errors+summary; "
            "1=info+trial status; 2=debug+detailed. "
            "Note: Controls Optuna logging only, use --sa-verbosity for SA module overhead."
        ),
    )
    parser.add_argument(
        "--sa-verbosity",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help=(
            "SA module verbosity (tree_search overhead tracking). Defaults to --verbosity if not set. "
            "0=warning (no overhead), 1=info (standard), 2=debug (detailed)."
        ),
    )
    parser.add_argument(
        "--study-patience",
        type=int,
        default=0,
        help="Stop the whole Optuna study if best value hasn't improved for this many trials (0 = disabled)",
    )
    parser.add_argument(
        "--absolute-min",
        type=float,
        default=None,
        help="Catastrophic absolute minimum mean score below which trials are immediately pruned (optional)",
    )
    args = parser.parse_args()

    # If no manifest provided, support single-graph inputs by creating a
    # temporary manifest-of-1 early so downstream path derivation (e.g. collection)
    # can pick up values from the manifest.
    if not args.manifest:
        if args.G and args.T and args.scores and args.positive and args.gid:
            manifest_entry = {
                "graph_id": args.gid,
                "collection": args.collection or "",
                "G_path": args.G,
                "T_path": args.T,
                "score_path": args.scores,
                "positive_edges_path": args.positive,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump([manifest_entry], f, indent=2)
                temp_manifest = f.name
            args.manifest = temp_manifest

    # Determine output directory. If the user didn't provide --out, derive it as
    # outputs/{collection}/optuna/{timestamp}_{study_name} where collection is taken from
    # --collection or from the manifest's first entry if available.
    if args.out:
        out_dir = args.out
    else:
        collection = args.collection
        # if manifest is set (possibly created above for single-graph), try to read collection
        if not collection and args.manifest and os.path.exists(args.manifest):
            try:
                with open(args.manifest, "r") as mf:
                    entries = json.load(mf)
                    if isinstance(entries, list) and len(entries) > 0:
                        collection = entries[0].get("collection") or collection
            except Exception:
                pass
        if not collection:
            collection = "ungrouped"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include study name in path to distinguish between methods (e.g., sa vs mst_sa)
        out_dir = os.path.join(
            "outputs", collection, "optuna", f"{ts}_{args.study_name}"
        )

    # Ensure we have a manifest (either provided or created above)
    if not args.manifest:
        sys.stderr.write(
            "Provide either --manifest or single-graph args (--G, --T, --scores, --positive, --gid).\n"
        )
        sys.exit(1)

    # Guard against accidental overwrite of previous run outputs
    if os.path.exists(out_dir) and os.listdir(out_dir) and not args.force:
        sys.stderr.write(
            f"Refusing to write into existing non-empty out directory {out_dir}. Use --force to overwrite.\n"
        )
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "logs")
    trials_dir = os.path.join(out_dir, "trial_trees")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(trials_dir, exist_ok=True)

    # Map verbosity to logging level
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = level_map.get(args.verbosity, logging.INFO)

    # Configure top-level logger early so initialization messages are captured
    logger = utils.setup_logger(out_dir, level=log_level)

    # Ensure SA module logs (`tree_search`) are routed into the same run logger
    try:
        tree_logger = logging.getLogger("tree_search")
        if not getattr(tree_logger, "handlers", None):
            for h in logger.handlers:
                tree_logger.addHandler(h)
            tree_logger.setLevel(logger.level)
            tree_logger.propagate = False
    except Exception:
        logger.exception("Failed to attach tree_search logger to run logger")

    # allow overriding the SA max_iter from CLI for quick runs
    globals()["GLOBAL_MAX_ITER"] = (
        int(args.max_iter) if args.max_iter is not None else None
    )
    globals()["GLOBAL_INIT_METHOD"] = args.init_method
    # pass verbosity into SA config via module-global for workers
    globals()["GLOBAL_VERBOSITY"] = int(args.verbosity)
    # SA-specific verbosity: defaults to --verbosity if --sa-verbosity not provided
    sa_verb = (
        int(args.sa_verbosity) if args.sa_verbosity is not None else int(args.verbosity)
    )
    globals()["GLOBAL_SA_VERBOSITY"] = sa_verb
    # Set graph parallelism for within-trial graph processing
    globals()["GRAPH_WORKERS"] = int(args.graph_workers)
    # Set early pruning thresholds
    globals()["EARLY_PRUNE_THRESHOLD"] = float(args.early_prune_threshold)
    globals()["EARLY_PRUNE_MIN_GRAPHS"] = int(args.early_prune_min_graphs)
    # Set score loading flag for no_scores variant
    globals()["GLOBAL_SKIP_SCORE_LOADING"] = bool(args.skip_score_loading)
    logger.info(
        f"SA Config: max_iter={GLOBAL_MAX_ITER}, init_method={args.init_method}, skip_score_loading={args.skip_score_loading}"
    )
    logger.info(
        f"Graph parallelism: {args.graph_workers} parallel processes per trial, "
        f"early prune if mean TPR < {args.early_prune_threshold} after {args.early_prune_min_graphs} graphs"
    )
    # If manifest mode: prepare pooled model and set up manifest-aware objective
    if args.manifest:
        with open(args.manifest, "r") as mf:
            manifest_entries = json.load(mf)
        entries = [
            e
            for e in manifest_entries
            if args.collection is None or e.get("collection") == args.collection
        ]
        if not entries:
            raise SystemExit("No manifest entries found for collection in manifest")

        # Validate manifest entries: require G_path, T_path, positive_edges_path
        # score_path is optional if skip_score_loading is True (no_scores variant)
        missing_entries = []
        for e in entries:
            gid = e.get("graph_id")
            missing = []
            if not e.get("G_path"):
                missing.append("G_path")
            if not e.get("T_path"):
                missing.append("T_path")
            if not e.get("positive_edges_path"):
                missing.append("positive_edges_path")
            # Only require score_path if not in skip_score_loading mode
            if not args.skip_score_loading and not e.get("score_path"):
                missing.append("score_path")
            if missing:
                missing_entries.append((gid or str(e), missing))
        if missing_entries:
            fields_str = "; ".join(
                [f"{gid}: {','.join(m)}" for gid, m in missing_entries]
            )
            raise SystemExit(f"Manifest entries missing required paths: {fields_str}")

        # Persist precomputed pickles (one per graph) so workers can reload them.
        precomp_dir = os.path.join(out_dir, "precomputed")
        os.makedirs(precomp_dir, exist_ok=True)

        # Parallel precompute across manifest entries (atomic per-graph pickles)
        precomputed = []
        max_workers = min(len(entries), (os.cpu_count() or 1))
        tasks = [(e, precomp_dir) for e in entries]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_precompute_one, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futs):
                t = futs[fut]
                try:
                    p = fut.result()
                    precomputed.append(p)
                except Exception:
                    logging.getLogger(LOGGER_NAME).exception(
                        f"Precompute failed for {t[0].get('graph_id')}"
                    )
                    raise

        # write a copy of the manifest subset next to precomputed pickles so
        # workers started with spawn can reconstruct the objective locally
        manifest_json_path = os.path.join(precomp_dir, "manifest.json")
        try:
            with open(manifest_json_path, "w") as mf:
                json.dump(entries, mf, indent=2)
        except Exception:
            logging.getLogger(LOGGER_NAME).exception("Failed writing manifest.json")

        precomputed_map = {p["graph_id"]: p for p in precomputed}

        # expose base seed to objective factory via module-global
        globals()["GLOBAL_SEED"] = int(args.seed)
        # optional catastrophic absolute-min threshold
        globals()["GLOBAL_ABS_MIN"] = (
            float(args.absolute_min) if args.absolute_min is not None else None
        )
        # study-level early stopping patience (trials without improvement)
        globals()["STUDY_PATIENCE"] = int(args.study_patience)

        # create manifest-aware objective and replace the global objective
        obj = objective_manifest_factory(entries, precomputed_map)
        globals()["objective"] = obj
        objective = obj

        # manifest processing is complete; we will run Optuna using the manifest

    if objective is None:
        raise SystemExit("Objective not initialized; manifest processing failed")

    # configure storage: default to a journal file in the out dir so parallel
    # workers can be used by default. If the user passed --storage, use that
    # path instead.
    storage_file = (
        args.storage if args.storage else os.path.join(out_dir, "optuna.journal")
    )
    os.makedirs(os.path.dirname(storage_file) or ".", exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(storage_file)
    )
    # Use a sampler seeded for reproducibility
    sampler = optuna.samplers.TPESampler(seed=globals().get("GLOBAL_SEED", None))
    # Create or load study in the chosen storage
    try:
        optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=storage,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, interval_steps=1),
        )
    except Exception:
        # study may already exist in storage
        pass
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    # Enqueue a baseline trial using best hyperparameters from Phase 1
    # Try to load from collection-specific best_hyperparameters.json (Phase 1 results)
    baseline_trial = None
    collection_name = args.collection
    if not collection_name and args.manifest:
        try:
            with open(args.manifest, "r") as mf:
                manifest_entries = json.load(mf)
                if isinstance(manifest_entries, list) and len(manifest_entries) > 0:
                    collection_name = manifest_entries[0].get("collection")
        except Exception:
            pass

    # Try to load best params from Phase 1
    if collection_name:
        phase1_params_path = os.path.join(
            "outputs", collection_name, "best_hyperparameters.json"
        )
        if os.path.exists(phase1_params_path):
            try:
                with open(phase1_params_path, "r") as f:
                    phase1_params = json.load(f)
                    baseline_trial = {
                        "initial_temp": phase1_params.get("initial_temp"),
                        "cooling_rate": phase1_params.get("cooling_rate"),
                        "community": phase1_params.get("loss_weights", {}).get(
                            "community"
                        ),
                        "diversity": phase1_params.get("loss_weights", {}).get(
                            "diversity"
                        ),
                        "degree": phase1_params.get("loss_weights", {}).get("degree"),
                        "shortcut": phase1_params.get("loss_weights", {}).get(
                            "shortcut"
                        ),
                        "score": phase1_params.get("loss_weights", {}).get("score"),
                        "resolution": phase1_params.get("resolution"),
                        "stagnation_threshold": phase1_params.get(
                            "stagnation_threshold", 10000
                        ),
                        "bold_stagnation_threshold": phase1_params.get(
                            "bold_stagnation_threshold", 10000
                        ),
                        "leiden_n_iterations": phase1_params.get(
                            "leiden_n_iterations", 2
                        ),
                        "min_bold_prob": phase1_params.get("min_bold_prob", 0.05),
                        "bold_prob_scale": phase1_params.get(
                            "bold_prob_scale", 0.3333333333333333
                        ),
                        "bold_multiplier": phase1_params.get("bold_multiplier", 3.0),
                    }
                    logging.getLogger(LOGGER_NAME).info(
                        f"✓ Loaded baseline trial from Phase 1: {phase1_params_path}"
                    )
            except Exception as e:
                logging.getLogger(LOGGER_NAME).warning(
                    f"Failed to load Phase 1 best params from {phase1_params_path}: {e}"
                )

    # Fallback to hardcoded defaults if Phase 1 params not found
    if baseline_trial is None:
        baseline_trial = {
            "initial_temp": 0.31836604989664075,
            "cooling_rate": 0.9999041607310027,
            "community": 2.485198852774305,
            "diversity": 2.4184655797768237,
            "degree": 4.803247912123328,
            "shortcut": 1.9814046201499087,
            "score": 4.999681963765728,
            "resolution": 0.12466809353674525,
            "stagnation_threshold": 35000,
            "bold_stagnation_threshold": 45000,
            "leiden_n_iterations": 2,
            "min_bold_prob": 0.18829950979088872,
            "bold_prob_scale": 0.208103495032827,
            "bold_multiplier": 4.796859739743831,
        }
        logging.getLogger(LOGGER_NAME).warning(
            "Using hardcoded baseline trial (Phase 1 params not found)"
        )

    # Enqueue baseline trial so it's evaluated first
    try:
        study.enqueue_trial(baseline_trial)
        logging.getLogger(LOGGER_NAME).info(
            f"✓ Enqueued baseline trial with {collection_name or 'default'} best params"
        )
    except Exception:
        logging.getLogger(LOGGER_NAME).exception("Failed to enqueue baseline trial")

    # === Parallel worker launch ===
    N_WORKERS = args.n_workers
    TRIALS_PER_WORKER = args.trials_per_worker

    # set module-level trial dir so objective writes to configured location
    TRIAL_TREES_DIR = trials_dir

    processes = []
    for i in range(N_WORKERS):
        p = multiprocessing.Process(
            target=run_worker,
            args=(
                i,
                TRIALS_PER_WORKER,
                storage_file,
                args.study_name,
                logs_dir,
                precomp_dir,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("✅ All workers finished.")
    msg = "Optimization finished. Best parameters and results saved."
    logger.info(msg)

    # reload study from storage
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    best_params = study.best_params
    best_TPR = study.best_value

    # Build loss_weights conditionally
    loss_weights = {
        "community": best_params.get("community"),
        "diversity": best_params.get("diversity"),
        "degree": best_params.get("degree"),
        "shortcut": best_params.get("shortcut"),
    }
    # Only include score if it was tuned (not in skip_score_loading mode)
    if "score" in best_params:
        loss_weights["score"] = best_params.get("score")
    else:
        loss_weights["score"] = 0.0  # Explicitly 0 for no_scores variant

    best_config = {
        "initial_temp": best_params.get("initial_temp"),
        "cooling_rate": best_params.get("cooling_rate"),
        "max_iter": int(GLOBAL_MAX_ITER) if GLOBAL_MAX_ITER else 100000000,
        "loss_weights": loss_weights,
        "resolution": best_params.get("resolution"),
        "stagnation_threshold": best_params.get("stagnation_threshold"),
        "bold_stagnation_threshold": best_params.get("bold_stagnation_threshold"),
        "leiden_n_iterations": best_params.get("leiden_n_iterations"),
        "min_bold_prob": best_params.get("min_bold_prob"),
        "bold_prob_scale": best_params.get("bold_prob_scale"),
        "bold_multiplier": best_params.get("bold_multiplier"),
        "seed": globals().get("GLOBAL_SEED", None),
    }

    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_config, f, indent=4)

    logger.info(f"Best parameters: {best_params}, Best TPR: {best_TPR}")
    logger.info("Optimization finished. Best hyperparameters saved.")

    # Save Optuna visualizations (best-effort; do not fail the run)
    viz_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    try:
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(viz_dir, "param_importance.png"))
        plt.close(ax.figure)
    except Exception:
        logger.exception("Failed to save param_importance plot")

    try:
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(viz_dir, "optimization_history.png"))
        plt.close(ax.figure)
    except Exception:
        logger.exception("Failed to save optimization_history plot")

    try:
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(viz_dir, "parallel_coordinates.png"))
        plt.close(ax.figure)
    except Exception:
        logger.exception("Failed to save parallel_coordinates plot")

    # cleanup temporary manifest-of-1 if we created one
    try:
        if (
            "temp_manifest" in locals()
            and temp_manifest
            and os.path.exists(temp_manifest)
        ):
            os.remove(temp_manifest)
            logger.info(f"Removed temporary manifest {temp_manifest}")
    except Exception:
        logger.exception("Failed to remove temporary manifest file")
