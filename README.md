# HELM: Hierarchy via Edge Learning and MST

**Supplementary Code and Data**

This package contains the source code and sample data for:

> **HELM: Hierarchy via Edge Learning and MST**  

## Dependencies

**Core (required):**

```bash
pip install networkx numpy pandas scikit-learn xgboost lightgbm optuna optuna-integration[xgboost,lightgbm,pytorch-lightning]
pip install graphMeasures
pip install --upgrade networkx  # Important: upgrade after graphMeasures
```

**GNN support (optional):**

```bash
pip install torch pytorch-lightning torch-geometric
```

**Wikipedia extraction (optional):**

```bash
pip install wikipediaapi
```

**Optuna for hyperparameter tuning (optional):**

```bash
pip install optuna
```

## HELM Pipeline (Complete Workflow)

The HELM algorithm follows a 4-step pipeline. Tested hyperparameters are provided in `configs/` for each dataset.

### Step 1: Extract Topological Features

```bash
# Extract features for all graphs in manifest
python -m src.algorithms.edge_features \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-base . \
  --workers 4
```

**Output:** `outputs/{collection}/{graph_id}/features/node_features.csv` and `edge_features.csv`

### Step 2: Train Edge Scoring Model

```bash
# Train XGBoost model with tested hyperparameters
python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --model xgb \
  --xgb-params configs/xgb_wiki.json
```

**Tested configs available:**

- `configs/xgb_wiki.json` (Wikipedia)
- `configs/xgb_microbiome.json` (Microbiome)  
- `configs/xgb_memetracker.json` (MemeTracker)

**Output:** `outputs/{collection}/model/best_model.pkl` and edge scores per graph

### Step 3: Hierarchy Reconstruction (Tree Search)

```bash
# Run tree search with tested hyperparameters
python -m src.algorithms.tree_search \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --config configs/sa_wiki.json \
  --max-iter 5000000
```

**Tested configs available:**

- `configs/sa_wiki.json` (Wikipedia)
- `configs/sa_microbiome.json` (Microbiome)
- `configs/sa_memetracker.json` (MemeTracker)

**Key parameters:**

- `--max-iter`: Number of iterations (default: 5M, use 0 for MST-only without SA search)
- `--init-method`: Tree initialization (`mst`, `positive_edges`, `empty`)

**Output:** `outputs/{collection}/{graph_id}/search/tree.pkl`

### Step 4: Root Selection and Directionality

```bash
# Find optimal root using depth distribution matching
python -m src.algorithms.optimal_root \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --mode eval
```

**Output:** `outputs/{collection}/{graph_id}/search/optimal_root/tree_directed.pkl`

---

## Configuration Files

### Edge Scoring (XGBoost)

Tested hyperparameters for each dataset in `configs/xgb_{collection}.json`:

### Tree Search (Simulated Annealing)

Tested hyperparameters for each dataset in `configs/sa_{collection}.json`:

---

## Optional: Hyperparameter Tuning with Optuna

If you want to discover new hyperparameters for your own datasets:

### Tune Edge Scoring Model

```bash
pip install optuna  # If not already installed

python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --model xgb \
  --n-trials 50
```

### Tune Tree Search Parameters

```bash
python -m src.scripts.optuna_tree_search \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --n-workers 4 \
  --trials-per-worker 25
```

This will search for optimal hyperparameters and save results to `outputs/{collection}/optuna/`.

---

## Core Modules

### `src/algorithms/edge_features.py`

Extracts structural features from graphs:

- Node features (centrality, clustering, k-core, Fiedler vector, etc.)
- Edge features (betweenness, connectivity, shortest path)
- Combined node-pair features (avg and diff vectors)

**Parallelization:** ProcessPoolExecutor for speedup

**Usage:**

```bash
# Single graph
python -m src.algorithms.edge_features --gid ID --collection NAME --output-base .

# Batch (manifest mode)
python -m src.algorithms.edge_features --manifest manifests/manifest.json --collection wiki
```

### `src/algorithms/edge_scores.py`

Edge scoring with GBDT models:

**Models:** XGBoost, LightGBM  
**Optimization:** Optuna with TPESampler  
**Early stopping:** MedianPruner (trial-level), PlateauCallback (study-level)  
**Multi-graph training:** Pool features across graphs, train single model  
**Evaluation:** AUC, F1, precision, recall on train/val/test splits

**Usage:**

```bash
# Train on 10 graphs
python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --model xgb \
  --n-trials 50 \
  --plateau-patience 20

# Score test graphs
python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --score-only \
  --model-path outputs/wiki/model
```

### `src/algorithms/edge_scores_gnn.py`

Graph Neural Network edge scoring:

**Architecture:** GINEConv with 4 layers, BatchNorm  
**Features:** Node embeddings (from node_features.csv) + edge features  
**Training:** PyTorch Lightning with early stopping  
**Optimization:** Optuna hyperparameter search  
**Evaluation:** ROC curves, per-graph and aggregate metrics

**Key features:**

- Scalable node feature encoding
- Edge dropout for regularization (20-60%)
- Aggressive hyperparameter tuning (hidden_dim, dropout_rate, learning_rate)
- Multi-graph training with train/val/test splits
- GPU acceleration support

**Usage:**

```bash
# Train GNN with hyperparameter search
python -m src.scripts.optuna_tree_search \
  --mode train \
  --train_manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output_dir . \
  --max_trials 30 \
  --gpu_id 0
```

### `src/algorithms/tree_search.py`

Simulated Annealing for hierarchy reconstruction:

**Features:**

- Initial tree from MST based on scores or union find tree based on set of known edges
- 3 move types: NNI, SPR, TBR
- Adaptive boldness (TBR/SPR ratio increases during stagnation)
- Early stopping (2-tier: stagnation + bold stagnation)
**Configuration:** 12 hyperparameters (optimizable via Optuna)

**Usage:**

```bash
# Optimize hyperparameters (Optuna)
python -m src.scripts.optuna_tree_search \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --n-workers 4 \
  --trials-per-worker 25

# Use best parameters for test set
python -m src.algorithms.tree_search \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --config outputs/wiki/optuna/.../best_hyperparameters.json
```

### `src/algorithms/optimal_root.py`

Optimal root selection for tree directionality:

**Root selection methods:**

- `depth_prior` (default): BFS-based, tests all candidate roots, minimises MSE against a depth histogram
- `rumor`: Shah & Zaman (2011) rumor centrality — prior-free (implemented in `src/algorithms/rumor_centrality.py`)

**Depth histogram sources (depth_prior method, in priority order):**

1. `--prior-path <file.json>` — explicit external prior (JSON list of counts per depth level)
2. `--use-T-depth-dist` — inferred from the ground-truth tree `T` at eval time
3. Fallback: tree-height heuristic (prefers balanced/shallow trees)

**Outputs per graph:**

- Optimal root node ID
- Depth distribution vectors
- Directed arborescence (tree rooted at optimal_root)
- Directed recall metrics (if ground-truth available)

**Usage:**

```bash
# Default depth-prior method (single graph)
python -m src.algorithms.optimal_root \
  --gid ID --collection wiki --output-dir . --mode train

# Batch with external prior
python -m src.algorithms.optimal_root \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki --output-dir . --mode eval \
  --prior-path configs/depth_prior_wiki.json

# Rumor centrality root selection
python -m src.algorithms.optimal_root \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki --output-dir . --mode eval \
  --root-method rumor
```

### `src/algorithms/rumor_centrality.py`

Rumor centrality computation (Shah & Zaman 2011):

- `estimate_source_exact(G)` — returns `(best_node, scores_dict)` for a connected undirected tree
- Uses adjacency arrays + BFS for exact log-score per candidate root
- Used internally by `optimal_root.py` when `--root-method rumor` is set

### `src/utils.py`

Utility functions:

| Function | Purpose |
|----------|---------|
| `load_graph(path)` | Load NetworkX graph from pickle |
| `save_graph(graph, path)` | Save NetworkX graph to pickle |
| `compute_tree_metrics(true_tree, pred_tree)` | Compute TPR, FPR, F1, etc. |
| `compute_confusion_from_trees(...)` | Get TP/FP/FN/TN counts |
| `setup_logger(out_dir, level)` | Configure logging |
| `Pool`, `UnionFind` | Data structures |

## Data Format Reference

### Manifest File Format

```json
[
  {
    "graph_id": "Algorithms",
    "collection": "wiki",
    "G_path": "data/wiki/Algorithms/entity_graph.pkl",
    "T_path": "data/wiki/Algorithms/hierarchy_tree.pkl",
    "node_features_path": "outputs/wiki/Algorithms/features/node_features.csv",
    "edge_features_path": "outputs/wiki/Algorithms/features/edge_features.csv",
    "score_path": "outputs/wiki/Algorithms/scores/edge_scores.csv",
    "positive_edges_path": "outputs/wiki/Algorithms/positive_edges/positive.pkl"
  }
]
```
