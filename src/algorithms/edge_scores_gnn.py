"""
GNN-based Edge Scoring for Hierarchy Prediction

CLEAR VARIABLE NAMING:
- node_features: Node embeddings from node_features.csv (used by GNN)
- edge_features: Structural edge features (betweenness, etc.)
- node_pair_features: Computed avg/diff vectors from node pairs
- final_edge_features: Concatenation of edge_features + node_pair_features
- node_embeddings: Learned representations after GNN message passing
- target_edges_tensor: Indices of edges to score

ARCHITECTURE:
1. Load node features -> Use as node_embeddings input
2. Load edge features + compute node_pair features -> final_edge_features
3. GNN message passing with node_embeddings and final_edge_features
4. Score target edges by concatenating src/dst node embeddings
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    # Use Tensor Cores with reduced precision accumulation for speed/memory
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINEConv
    from torch_geometric.utils import dropout_edge
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback

    _gnn_deps_available = True
    _gnn_deps_error = None
except ImportError as _e:
    _gnn_deps_available = False
    _gnn_deps_error = str(_e)

# GNN class definitions use torch/pl at module level — fail early with a helpful message
if not _gnn_deps_available:
    raise ImportError(
        f"GNN dependencies not available ({_gnn_deps_error}).\n"
        "Install with: pip install torch pytorch-lightning torch-geometric optuna\n"
        "Skip this module entirely if you only need the XGBoost/LightGBM pipeline."
    )

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from src.utils import load_graph
from datetime import datetime

DEFAULT_SEED = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def prepare_features(node_df, edge_df, use_abs_diff: bool = False):
    """
    Prepare combined edge features: structural + node-pair features.
    If use_abs_diff is True, make diff sign-invariant (for undirected graphs).

    Input:
      node_df: DataFrame with node features (indexed by node ID)
      edge_df: DataFrame with edge features (indexed by (source, target) tuples)

    Returns:
      edge_df: Enhanced with node-pair avg/diff vectors
      feature_cols: List of all feature column names

    Example:
      If node_features = [x1, x2, x3] for nodes u and v:
        avg_features = [(u.x1 + v.x1)/2, (u.x2 + v.x2)/2, ...]
        diff_features = [(u.x1 - v.x1)/2, (u.x2 - v.x2)/2, ...]
    """
    node_df.index.name = "node"

    # VECTORIZED: Extract sources and targets
    sources = [u for u, v in edge_df.index]
    targets = [v for u, v in edge_df.index]

    # VECTORIZED: Lookup all node features at once
    # Handle missing nodes by reindexing with fill_value=0
    node_u_matrix = node_df.reindex(sources, fill_value=0.0).values
    node_v_matrix = node_df.reindex(targets, fill_value=0.0).values

    # VECTORIZED: Compute avg/diff for all edges at once
    avg_vectors = (node_u_matrix + node_v_matrix) / 2
    diff_vectors = (node_u_matrix - node_v_matrix) / 2
    if use_abs_diff:
        diff_vectors = np.abs(diff_vectors)

    # Combine avg and diff features
    combined_vectors = np.concatenate([avg_vectors, diff_vectors], axis=1)

    # Create dataframe with avg/diff features
    avg_diff_df = pd.DataFrame(
        combined_vectors,
        index=edge_df.index,
        columns=[f"avg_{col}" for col in node_df.columns]
        + [f"diff_{col}" for col in node_df.columns],
    )

    # Combine with original edge features
    edge_df = pd.concat([edge_df, avg_diff_df], axis=1)
    feature_cols = list(edge_df.columns.difference(["source", "target"]))

    return edge_df, feature_cols


def load_features(node_path, edge_path):
    """Load and validate node/edge features from CSV files."""
    if not os.path.exists(node_path):
        raise FileNotFoundError(f"Node features not found: {node_path}")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge features not found: {edge_path}")

    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)

    # Set node index
    if "node" in node_df.columns:
        node_df = node_df.set_index("node")

    # Set edge index
    if "source" in edge_df.columns and "target" in edge_df.columns:
        edge_df["edge"] = list(zip(edge_df["source"], edge_df["target"]))
        edge_df = edge_df.set_index("edge")

    # Validate all edge nodes exist in node_df
    edge_nodes = set(u for edge in edge_df.index for u in edge)
    missing_nodes = edge_nodes - set(node_df.index)
    if missing_nodes:
        raise ValueError(f"Missing nodes in node_features: {list(missing_nodes)[:5]}")

    # Drop columns with NaN
    node_df.dropna(axis=1, inplace=True)
    edge_cols = [c for c in edge_df.columns if c not in ("source", "target")]
    for col in edge_cols:
        if edge_df[col].isna().any():
            edge_df.drop(columns=[col], inplace=True)

    return node_df, edge_df


def pool_features_and_samples(
    manifest_path, collection, base_dir, skip_shortest_path=False, seed=DEFAULT_SEED
):
    """
    Pool features from all graphs in manifest for multi-graph training.

    Process:
    1. Load node features + edge features for each graph
    2. Compute node-pair avg/diff features for each edge
    3. Combine all edge features across graphs
    4. Optionally skip all_pairs_shortest_path_* features
    5. Fit single StandardScaler on pooled features
    6. Return as MultiIndex DataFrame (graph_id, edge)

    Args:
        skip_shortest_path: If True, remove all_pairs_shortest_path_* columns

    Returns:
      pooled_df: All features with MultiIndex (graph_id, edge_tuple)
      feature_cols: Column names (consistent across graphs)
      scaler: Fitted StandardScaler
      graphs: List of (graph_id, G, T, node_df) tuples
    """
    set_seed(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter for specific collection
    entries = [e for e in manifest if e.get("collection") == collection]
    if not entries:
        raise RuntimeError(f"No entries found for collection '{collection}'")

    dfs = []
    feature_cols = None
    graphs = []

    print(f"Pooling features from {len(entries)} graphs...")
    for entry in entries:
        graph_id = entry["graph_id"]
        print(f"  Loading {graph_id}...")

        # Load features
        node_df, edge_df = load_features(
            entry["node_features_path"], entry["edge_features_path"]
        )

        # Load graphs
        G = load_graph(entry["G_path"]) if entry.get("G_path") else None
        T = load_graph(entry["T_path"]) if entry.get("T_path") else None

        # Determine if frozenset needed (for undirected graphs)
        G_is_undirected = not G.is_directed() if G else False
        T_is_undirected = not T.is_directed() if T else False

        # Reject G directed but T undirected (inconsistent state)
        if not G_is_undirected and T_is_undirected:
            raise ValueError(
                f"{graph_id}: G directed but T undirected - Not supported."
            )

        # Use frozensets if either graph is undirected
        use_abs_diff = G_is_undirected or T_is_undirected

        # Filter node_df BEFORE prepare_features if skip_shortest_path is set
        if skip_shortest_path:
            apsp_node_cols = [
                c for c in node_df.columns if c.startswith("all_pairs_shortest_path")
            ]
            if apsp_node_cols:
                node_df = node_df.drop(columns=apsp_node_cols)

        # Prepare features (edge + node-pair) with abs_diff for undirected
        processed_df, cols = prepare_features(
            node_df, edge_df, use_abs_diff=use_abs_diff
        )

        # Validate feature consistency across graphs
        if feature_cols is None:
            feature_cols = list(cols)
        elif list(cols) != feature_cols:
            raise RuntimeError(f"Feature mismatch in {graph_id}")

        # Add graph_id and collect
        processed_df = processed_df.reset_index()
        processed_df["graph_id"] = graph_id
        dfs.append(processed_df)
        graphs.append((graph_id, G, T, node_df))  # Save node_df for later

    # Pool all features (UNSCALED - scaling will happen after train/val/test split)
    print("Pooling features (scaling will be applied after train/val/test split)...")
    pooled = pd.concat(dfs, ignore_index=True)

    # Optionally skip all_pairs_shortest_path_* features
    if skip_shortest_path:
        # Get APSP columns from edge features (node_df already filtered above)
        apsp_edge_cols = [
            c for c in feature_cols if c.startswith("all_pairs_shortest_path")
        ]
        if apsp_edge_cols:
            print(
                f"   Reduced edge feature set: {len(feature_cols)} → {len(feature_cols) - len(apsp_edge_cols)} features (removed {len(apsp_edge_cols)} APSP)"
            )
            feature_cols = [c for c in feature_cols if c not in apsp_edge_cols]
            # Also remove from pooled data
            pooled = pooled.drop(columns=apsp_edge_cols)

    # Return None for scaler - it will be created in train_with_optuna after split
    scaler = None

    # Set MultiIndex
    pooled["edge"] = list(zip(pooled["source"], pooled["target"]))
    pooled = pooled.set_index(["graph_id", "edge"])

    # Convert feature columns to float64 to avoid dtype mismatch when scaling
    pooled[feature_cols] = pooled[feature_cols].astype("float64")

    print(f"Pooled {len(pooled)} edges from {len(graphs)} graphs")

    return pooled, feature_cols, scaler, graphs


def sample_pooled_edges(
    pooled_df, graphs, feature_cols, fraction_pos=0.1, min_pos=10, seed=DEFAULT_SEED
):
    """
    Sample positive (in tree T) and negative (in graph G but not T) edges.

    DIRECTIONALITY FIX: Uses frozensets for undirected graphs to avoid
    matching issues (e.g., where G is undirected but we want direction-invariant edges).

    Returns:
      X: Feature matrix [num_edges, num_features]
      y: Labels [1=positive, 0=negative]
      sampled_edges: Dict mapping graph_id -> list of (edge_tuple, label) tuples
    """
    set_seed(seed)

    sampled_list = []
    labels = []
    sampled_dict = {graph_id: [] for graph_id, _, _, _ in graphs}

    for graph_id, G, T, node_df in graphs:
        if not G or not T:
            continue

        # Get edges that have features in pooled_df
        graph_edges = pooled_df.loc[graph_id].index.tolist()

        # Check directionality
        G_undirected = not G.is_directed()
        T_undirected = not T.is_directed()

        if not G_undirected and T_undirected:
            raise ValueError(f"{graph_id}: G directed but T undirected - Not supported")

        # Use frozensets if either is undirected
        use_frozenset = G_undirected or T_undirected

        # Convert all edges to keys for consistent comparison
        feature_edge_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in graph_edges
        }
        T_edge_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in T.edges()
        }
        G_edge_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in G.edges()
        }

        # Find edges that appear in both feature set and graph/tree
        positive_edges = T_edge_keys & feature_edge_keys  # In tree AND have features
        negative_candidates = (
            G_edge_keys & feature_edge_keys
        )  # In graph AND have features

        if not positive_edges:
            raise RuntimeError(f"No positive edges found for {graph_id}")

        # Sample edges
        num_pos = int(max(min_pos, fraction_pos * len(positive_edges)))
        sampled_pos = random.sample(
            list(positive_edges), min(num_pos, len(positive_edges))
        )
        sampled_neg = random.sample(
            list(negative_candidates - set(sampled_pos)),
            min(4 * len(sampled_pos), len(negative_candidates) - len(sampled_pos)),
        )

        # Build lookup: edge_key -> actual_edge_tuple
        feature_lookup = {
            (frozenset({u, v}) if use_frozenset else (u, v)): (u, v)
            for u, v in graph_edges
        }

        # Collect sampled edges
        for edge_key in sampled_pos:
            feature_tuple = feature_lookup[edge_key]
            sampled_list.append((graph_id, feature_tuple))
            labels.append(1)
            sampled_dict[graph_id].append((feature_tuple, 1))

        for edge_key in sampled_neg:
            feature_tuple = feature_lookup[edge_key]
            sampled_list.append((graph_id, feature_tuple))
            labels.append(0)
            sampled_dict[graph_id].append((feature_tuple, 0))

    # Create feature matrix
    X = (
        pooled_df.loc[sampled_list, feature_cols].values
        if sampled_list
        else np.empty((0, len(feature_cols)))
    )

    return X, labels, sampled_dict


class EdgeScoringDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for edge scoring.

    Batches edges PER GRAPH (not graphs per edge) to avoid duplicating full graphs.
    Returns: (graph_data, chunk_of_target_edges[K,2], chunk_of_labels[K])

    This avoids the OOM catastrophe of returning (full_graph, single_edge) × batch_size.
    """

    def __init__(
        self,
        edge_list,
        graphs_dict,
        pooled_df,
        feature_cols,
        node_features_dict,
        graph_cache=None,
        node_to_idx_cache=None,
        chunk_size=256,
    ):
        """
        Args:
          edge_list: List of (graph_id, edge_tuple, label)
          graphs_dict: Dict[graph_id -> (G, T)]
          pooled_df: Feature DataFrame with MultiIndex
          feature_cols: Feature column names
          node_features_dict: Dict[graph_id -> node_df]
          graph_cache: Pre-built PyG Data cache (optional, builds if None)
          node_to_idx_cache: Pre-built node index mappings (optional)
          chunk_size: Number of edges per sample (group by graph, then chunk)
        """
        self.chunk_size = chunk_size
        self.feature_cols = feature_cols
        self.graphs_dict = graphs_dict
        self.pooled_df = pooled_df
        self.node_features_dict = node_features_dict

        # Group edges by graph_id
        self.by_gid = {}
        for gid, edge, label in edge_list:
            self.by_gid.setdefault(gid, []).append((edge, label))

        self.gids = list(self.by_gid.keys())

        # Use pre-built cache if provided, otherwise build now
        if graph_cache is not None and node_to_idx_cache is not None:
            self.graph_cache = graph_cache
            self.node_to_idx_cache = node_to_idx_cache
        else:
            # Build and cache PyG graphs once for each graph_id
            self.graph_cache = {}
            self.node_to_idx_cache = {}
            for gid in self.gids:
                data, node_to_idx = self._build_graph_data(gid)
                self.graph_cache[gid] = data
                self.node_to_idx_cache[gid] = node_to_idx

        # Build index of (graph_id, chunk_start) for efficient iteration
        self.index = []
        for gid in self.gids:
            n_edges = len(self.by_gid[gid])
            for s in range(0, n_edges, self.chunk_size):
                self.index.append((gid, s))

    def _build_graph_data(self, graph_id):
        """
        Build PyG Data object for a graph.

        Returns:
          data: PyG Data with x=node_features, edge_index, edge_attr
          node_to_idx: Mapping from node ID to tensor index
        """
        G, T = self.graphs_dict[graph_id]
        node_df = self.node_features_dict[graph_id]

        # Node indices
        node_ids = list(G.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Node features: Use actual node embeddings from node_features.csv
        node_features_list = []
        for node_id in node_ids:
            if node_id in node_df.index:
                node_features_list.append(node_df.loc[node_id].values)
            else:
                # Missing node: use zeros
                node_features_list.append(np.zeros(node_df.shape[1]))

        node_embeddings = torch.tensor(
            np.array(node_features_list), dtype=torch.float32
        )

        # Edge indices
        edge_list = list(G.edges())
        edge_index = (
            torch.tensor(
                [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        # Edge features: Use pooled features (VECTORIZED for speed)
        # Build MultiIndex for all edges at once
        edge_multiindex = pd.MultiIndex.from_tuples(
            [(graph_id, (u, v)) for u, v in edge_list], names=self.pooled_df.index.names
        )

        # Vectorized lookup: get all edge features at once
        try:
            edge_features = self.pooled_df.loc[
                edge_multiindex, self.feature_cols
            ].values
        except KeyError:
            # Fallback: some edges might be missing, handle individually
            edge_features_list = []
            for u, v in edge_list:
                try:
                    feat = self.pooled_df.loc[
                        (graph_id, (u, v)), self.feature_cols
                    ].values
                except KeyError:
                    feat = np.zeros(len(self.feature_cols))
                edge_features_list.append(feat)
            edge_features = np.array(edge_features_list)

        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        # Create PyG Data object (do NOT clone here)
        data = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

        return data, node_to_idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        gid, chunk_start = self.index[idx]

        # Get chunk of edges for this graph
        chunk = self.by_gid[gid][chunk_start : chunk_start + self.chunk_size]

        # Get graph data and node mapping
        graph_data = self.graph_cache[gid]
        node_to_idx = self.node_to_idx_cache[gid]

        # Convert chunk edges to node indices and labels
        target_edges = []
        labels = []
        for (u, v), y in chunk:
            target_edges.append([node_to_idx[u], node_to_idx[v]])
            labels.append(y)

        # Return: graph (once), edges (many), labels (many)
        # Do NOT clone graph_data; Batch.from_data_list will handle batching
        return (
            graph_data,
            torch.tensor(target_edges, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )


def custom_collate_fn(batch):
    """Collate edges from SAME GRAPH only.

    With batch_size=1, each batch contains exactly 1 sample from 1 graph.
    This ensures we never mix edge chunks from different graphs.

    Input: batch = [(graph_data, target_edges[K,2], labels[K])]
    Output: (graph_data, target_edges[K,2], labels[K])
    """
    assert len(batch) == 1, "batch_size must be 1 to ensure single-graph batches"
    return batch[0]


class GINEEdgeScorerModel(nn.Module):
    """
    Graph Neural Network for edge scoring using GINEConv.

    Architecture:
    1. Node encoder: node_features -> hidden_dim
    2. Edge encoder: edge_features -> hidden_dim
    3. Message passing: GINEConv layers with edge features
    4. Edge scorer: MLP on [src_embedding || dst_embedding]
    """

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_rate=0.1,
        edge_dropout_rate=0.0,
    ):
        super().__init__()

        self.edge_dropout_rate = edge_dropout_rate

        # Encoders
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)

        # GINEConv layers with BatchNorm for better generalization
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()  # Add batch normalization

        for _ in range(num_layers):
            # MLP for GINEConv
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_layers.append(GINEConv(mlp, edge_dim=hidden_dim))
            # LayerNorm for message passing stabilization
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
            # BatchNorm to normalize activations and reduce covariate shift across cascades
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout_rate)

        # Edge scorer head: takes [src || dst] embeddings
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_embeddings,
        edge_index,
        edge_attr,
        target_edges_tensor,
    ):
        """
        Forward pass for edge scoring.

        Args:
          node_embeddings: [num_nodes, node_feature_dim]
          edge_index: [2, num_edges] - source and target node indices
          edge_attr: [num_edges, edge_feature_dim]
          target_edges_tensor: [batch_size, 2] - indices of edges to score

        Returns:
          logits: [batch_size] - raw scores for target edges
        """
        # Encode
        node_embeddings = self.node_encoder(node_embeddings)
        edge_attr = self.edge_encoder(edge_attr)
        # Edge dropout during training using PyG's built-in dropout_edge
        if self.training and self.edge_dropout_rate > 0:
            edge_index, edge_mask = dropout_edge(
                edge_index,
                p=self.edge_dropout_rate,
                training=self.training,
            )
            # Apply mask to edge_attr (keep only non-dropped edges)
            edge_attr = edge_attr[edge_mask]

        # Message passing with aggressive regularization
        for conv, norm, batch_norm in zip(
            self.conv_layers, self.norm_layers, self.batch_norm_layers
        ):
            node_embeddings = conv(node_embeddings, edge_index, edge_attr)
            node_embeddings = norm(node_embeddings)  # LayerNorm per node
            node_embeddings = batch_norm(
                node_embeddings
            )  # BatchNorm to reduce cascade-level covariate shift
            node_embeddings = F.relu(node_embeddings)
            node_embeddings = self.dropout(node_embeddings)

        # Score target edges
        # Get embeddings for source and target nodes
        source_embeddings = node_embeddings[target_edges_tensor[:, 0]]
        target_embeddings = node_embeddings[target_edges_tensor[:, 1]]

        # Concatenate and score
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)
        # DEBUG: print shapes
        if edge_embeddings.shape[0] != target_edges_tensor.shape[0]:
            print(
                f"WARNING: edge_embeddings shape {edge_embeddings.shape} != target_edges_tensor shape {target_edges_tensor.shape}"
            )
        logits = self.score_head(edge_embeddings).squeeze(-1)

        return logits


class EdgeScorerLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for GNN edge scorer."""

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_rate=0.1,
        edge_dropout_rate=0.0,
        learning_rate=1e-3,
        weight_decay=1e-5,
        pos_weight=1.0,
        gradient_clip_val=1.0,  # NEW: Gradient clipping parameter
    ):
        super().__init__()
        self.save_hyperparameters()
        self.gradient_clip_val = gradient_clip_val  # NEW: Store for optimizer_step

        self.model = GINEEdgeScorerModel(
            node_feature_dim,
            edge_feature_dim,
            hidden_dim,
            num_layers,
            dropout_rate,
            edge_dropout_rate,
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Store validation outputs
        self.validation_outputs = []

    def forward(
        self,
        node_embeddings,
        edge_index,
        edge_attr,
        target_edges_tensor,
    ):
        return self.model(
            node_embeddings,
            edge_index,
            edge_attr,
            target_edges_tensor,
        )

    def training_step(self, batch, batch_idx):
        graph_data, target_edges_tensor, labels = batch

        logits = self.forward(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
            target_edges_tensor,
        )

        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(labels))

        return loss

    def validation_step(self, batch, batch_idx):
        graph_data, target_edges_tensor, labels = batch

        logits = self.forward(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
            target_edges_tensor,
        )

        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        self.log("val_loss", loss, prog_bar=True, batch_size=len(labels))

        # Collect for epoch-end metrics
        self.validation_outputs.append(
            {"preds": probs.detach().cpu(), "labels": labels.detach().cpu()}
        )

        return loss

    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch."""
        if not self.validation_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.validation_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_outputs])

        # Compute metrics
        auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        preds_binary = (all_preds > 0.5).long()
        f1 = f1_score(all_labels.numpy(), preds_binary.numpy(), zero_division=0)
        precision = precision_score(
            all_labels.numpy(), preds_binary.numpy(), zero_division=0
        )
        recall = recall_score(all_labels.numpy(), preds_binary.numpy(), zero_division=0)

        self.log("val_auc", auc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_precision", precision)
        self.log("val_recall", recall)

        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


class EdgeScoringDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for edge scoring.

    IMPORTANT: batch_size is FIXED to 1 (architectural constraint for single-graph batching).
    Each batch contains edges from only one graph to avoid mixing graphs.
    """

    def __init__(
        self,
        train_edge_list,
        val_edge_list,
        test_edge_list,
        graphs_dict,
        pooled_df,
        feature_cols,
        node_features_dict,
        graph_cache=None,
        node_to_idx_cache=None,
        num_workers=4,
    ):
        """Initialize EdgeScoringDataModule.

        Args:
            graph_cache: Pre-built PyG Data cache to avoid rebuilding graphs 3x
            node_to_idx_cache: Pre-built node index mappings
        """
        super().__init__()
        self.train_edge_list = train_edge_list
        self.val_edge_list = val_edge_list
        self.test_edge_list = test_edge_list
        self.graphs_dict = graphs_dict
        self.pooled_df = pooled_df
        self.feature_cols = feature_cols
        self.node_features_dict = node_features_dict
        self.graph_cache = graph_cache
        self.node_to_idx_cache = node_to_idx_cache
        self.num_workers = num_workers
        # batch_size is FIXED to 1 (not passed as param)

    def setup(self, stage=None):
        """Create train/val/test datasets.

        OPTIMIZATION: If pre-built graph_cache provided, all three datasets share it.
        This avoids rebuilding the same graphs 3 times (was the bottleneck!).
        """
        self.train_dataset = EdgeScoringDataset(
            self.train_edge_list,
            self.graphs_dict,
            self.pooled_df,
            self.feature_cols,
            self.node_features_dict,
            graph_cache=self.graph_cache,
            node_to_idx_cache=self.node_to_idx_cache,
        )
        self.val_dataset = EdgeScoringDataset(
            self.val_edge_list,
            self.graphs_dict,
            self.pooled_df,
            self.feature_cols,
            self.node_features_dict,
            graph_cache=self.graph_cache,
            node_to_idx_cache=self.node_to_idx_cache,
        )
        self.test_dataset = EdgeScoringDataset(
            self.test_edge_list,
            self.graphs_dict,
            self.pooled_df,
            self.feature_cols,
            self.node_features_dict,
            graph_cache=self.graph_cache,
            node_to_idx_cache=self.node_to_idx_cache,
        )

    def train_dataloader(self):
        # batch_size=1 ensures each batch is from a single graph only
        # This prevents mixing edge chunks from different graphs in one batch
        # Throughput is maintained by chunk_size in EdgeScoringDataset
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 1),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 1),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 1),
        )


def train_with_optuna(
    manifest_path,
    collection,
    output_dir,
    device,
    max_trials=30,
    max_epochs=200,
    patience=20,
    num_workers=4,
    plateau_patience=20,
    plateau_min_trials=50,
    plateau_min_delta=0.0001,
    skip_shortest_path_features=False,
    hidden_dim_range=None,
    seed=DEFAULT_SEED,
    optuna_enabled=True,
    hyperparams_config=None,
):
    """Train GNN with optional Optuna hyperparameter optimization.

    If optuna_enabled=False, requires hyperparams_config with explicit params.
    """
    set_seed(seed)

    # Use defaults if not provided
    # batch_size is FIXED to 1 (architectural constraint)
    # chunk_size is FIXED to 256 (doesn't affect learning, only efficiency)
    if hidden_dim_range is None:
        hidden_dim_range = [64, 96, 128, 192]

    print("\n" + "=" * 70)
    print("STEP 1: POOLING FEATURES FROM ALL GRAPHS")
    print("=" * 70)

    pooled_df, feature_cols, scaler, graphs_list = pool_features_and_samples(
        manifest_path,
        collection,
        output_dir,
        skip_shortest_path=skip_shortest_path_features,
        seed=seed,
    )

    # Prepare graphs dict and node features dict
    graphs_dict = {graph_id: (G, T) for graph_id, G, T, _ in graphs_list}
    node_features_dict = {graph_id: node_df for graph_id, _, _, node_df in graphs_list}

    print("\n" + "=" * 70)
    print("STEP 2: SAMPLING POSITIVE/NEGATIVE EDGES")
    print("=" * 70)

    X, y, sampled_edges = sample_pooled_edges(
        pooled_df, graphs_list, feature_cols, seed=seed
    )

    print(f"Total sampled edges: {len(y)}")
    print(f"  Positive: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"  Negative: {len(y) - sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN/VAL/TEST SPLIT")
    print("=" * 70)

    # Train/val split (80/20) from sampled edges - matches XGB
    indices = list(range(len(y)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=seed
    )

    # Build train/val edge lists from sampled edges
    train_edges, val_edges = [], []
    train_X_indices, val_X_indices = [], []
    idx = 0
    for graph_id, edges_labels in sampled_edges.items():
        for edge, label in edges_labels:
            if idx in train_idx:
                train_edges.append((graph_id, edge, label))
                train_X_indices.append(idx)
            elif idx in val_idx:
                val_edges.append((graph_id, edge, label))
                val_X_indices.append(idx)
            idx += 1

    # Build test set: ALL edges in each graph not in train/val (matches XGB)
    # This is CRITICAL: test = all_unsampled_edges, NOT 10% of sampled edges
    print(f"\nBuilding test set from unsampled edges (like XGB)...")
    test_edges = []
    test_X_indices = []
    for graph_id, G, T, _ in graphs_list:
        if T is None or G is None:
            continue

        # Determine if frozenset needed (for undirected graphs)
        G_is_undirected = not G.is_directed()
        T_is_undirected = not T.is_directed()

        if not G_is_undirected and T_is_undirected:
            raise ValueError(
                f"Graph {graph_id}: G directed but T undirected. Not supported."
            )

        use_frozenset = G_is_undirected or T_is_undirected

        # Get all feature edges from pooled_df (ground truth edge list)
        feature_edges = pooled_df.loc[graph_id].index.tolist()

        # Get sampled edges (train + val) as tuples
        sampled_tuples = [e for e, _ in sampled_edges[graph_id]]

        # Find unsampled edges with proper direction handling
        if use_frozenset:
            # Map frozenset keys to original tuples
            feature_key_to_tuple = {frozenset({u, v}): (u, v) for u, v in feature_edges}
            sampled_keys = {frozenset({u, v}) for u, v in sampled_tuples}

            # Unsampled edges = feature edges not in sampled
            unused_keys = feature_key_to_tuple.keys() - sampled_keys
            unused_edges = [feature_key_to_tuple[k] for k in unused_keys]
        else:
            # Both directed: direct set difference
            unused_edges = list(set(feature_edges) - set(sampled_tuples))

        if not unused_edges:
            continue

        # Build T edge keys for labeling
        T_edge_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in T.edges()
        }

        # Assign labels: 1 if in tree T, else 0
        for edge in unused_edges:
            key = frozenset({edge[0], edge[1]}) if use_frozenset else edge
            label = 1 if key in T_edge_keys else 0
            test_edges.append((graph_id, edge, label))

    print(
        f"\nTrain:      {len(train_edges)} edges ({100*len(train_edges)/len(y):.1f}% of sampled)"
    )
    print(
        f"Validation: {len(val_edges)} edges ({100*len(val_edges)/len(y):.1f}% of sampled)"
    )
    print(f"Test:       {len(test_edges)} edges (ALL unsampled from train graphs)")
    n_pos_test = sum(1 for _, _, label in test_edges if label == 1)
    print(f"  Test positive: {n_pos_test} ({100*n_pos_test/len(test_edges):.1f}%)")
    print(
        f"  Test negative: {len(test_edges)-n_pos_test} ({100*(len(test_edges)-n_pos_test)/len(test_edges):.1f}%)"
    )

    print("\n" + "=" * 70)
    print("STEP 4: FIT SCALER ON TRAIN FEATURES ONLY")
    print("=" * 70)

    # Extract train, val, test features from pooled_df (UNSCALED) - like XGB
    # Build MultiIndex for each split
    train_idx = pd.MultiIndex.from_tuples(
        [(gid, edge) for gid, edge, _ in train_edges], names=pooled_df.index.names
    )
    val_idx = pd.MultiIndex.from_tuples(
        [(gid, edge) for gid, edge, _ in val_edges], names=pooled_df.index.names
    )
    test_idx = pd.MultiIndex.from_tuples(
        [(gid, edge) for gid, edge, _ in test_edges], names=pooled_df.index.names
    )

    train_X_unscaled = pooled_df.loc[train_idx, feature_cols].values
    val_X_unscaled = (
        pooled_df.loc[val_idx, feature_cols].values
        if len(val_edges) > 0
        else np.array([])
    )
    test_X_unscaled = pooled_df.loc[test_idx, feature_cols].values

    # Fit scaler on train features ONLY (this fixes the leakage!)
    print(f"Fitting scaler on {len(train_X_unscaled)} train samples only...")

    # Create model directory with timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(output_dir) / collection / f"model_gnn_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Model directory: {model_dir}")

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X_unscaled)

    # Transform val and test with train-fitted scaler
    val_X_scaled = (
        scaler.transform(val_X_unscaled) if len(val_X_unscaled) > 0 else np.array([])
    )
    test_X_scaled = scaler.transform(test_X_unscaled)

    print(f"  Train features: {train_X_scaled.shape}")
    print(f"  Val features: {val_X_scaled.shape}")
    print(f"  Test features: {test_X_scaled.shape}")
    print(f"  Scaler fitted on train data only ✅")

    # Update pooled_df with scaled features directly (like XGB)
    print(f"\nApplying scaled features to pooled_df...")
    pooled_df.loc[train_idx, feature_cols] = train_X_scaled
    if len(val_edges) > 0:
        pooled_df.loc[val_idx, feature_cols] = val_X_scaled
    pooled_df.loc[test_idx, feature_cols] = test_X_scaled

    # Scale node features: fit scaler on train graphs only
    print(f"\nScaling node features...")
    train_node_features = []
    for graph_id, edge, label in train_edges:
        if graph_id not in {gid for gid, _, _ in train_edges if True}:
            continue
        node_df = node_features_dict[graph_id]
        train_node_features.append(node_df.values)

    # Get unique graph IDs in training set
    train_graph_ids = set(gid for gid, _, _ in train_edges)
    train_node_features = []
    for gid in train_graph_ids:
        train_node_features.append(node_features_dict[gid].values)

    train_node_features = np.vstack(train_node_features)

    # Fit scaler on train node features
    node_feature_scaler = StandardScaler()
    node_feature_scaler.fit(train_node_features)

    # Scale all node features (train, val, and test graphs)
    for gid in node_features_dict:
        scaled_nodes = node_feature_scaler.transform(node_features_dict[gid].values)
        node_features_dict[gid] = pd.DataFrame(
            scaled_nodes,
            index=node_features_dict[gid].index,
            columns=node_features_dict[gid].columns,
        )

    print(
        f"  Node features scaled (scaler fitted on {len(train_graph_ids)} train graphs)"
    )
    sample_graph_id = list(graphs_dict.keys())[0]
    node_feature_dim = node_features_dict[sample_graph_id].shape[1]
    edge_feature_dim = len(feature_cols)
    node_feature_cols = list(node_features_dict[sample_graph_id].columns)

    # Compute class weights
    n_pos_train = sum(1 for _, _, label in train_edges if label == 1)
    n_neg_train = len(train_edges) - n_pos_train
    pos_weight = n_neg_train / max(1, n_pos_train)

    print(f"\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Node feature dimension: {node_feature_dim}")
    print(f"Edge feature dimension: {edge_feature_dim}")
    print(f"Positive weight: {pos_weight:.2f}")

    # Save training artifacts (model_dir already created earlier)
    with open(model_dir / "edge_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(model_dir / "node_scaler.pkl", "wb") as f:
        pickle.dump(node_feature_scaler, f)
    print(f"✅ Saved scalers:")
    print(f"   - Edge scaler to {model_dir / 'edge_scaler.pkl'}")
    print(f"   - Node scaler to {model_dir / 'node_scaler.pkl'}")
    with open(model_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(model_dir / "node_feature_cols.json", "w") as f:
        json.dump(node_feature_cols, f)
    with open(model_dir / "dims.json", "w") as f:
        json.dump(
            {
                "node_feature_dim": node_feature_dim,
                "edge_feature_dim": edge_feature_dim,
                "pos_weight": float(pos_weight),
            },
            f,
        )

    # Save datasets for analysis (Issue #8: Save train/val/test edge lists)
    datasets_dir = model_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    def save_edge_list(edge_list, filename):
        """Save edge list as JSON for later analysis."""
        data = [
            {"graph_id": gid, "edge": list(edge), "label": int(label)}
            for gid, edge, label in edge_list
        ]
        with open(datasets_dir / filename, "w") as f:
            json.dump(data, f, indent=2)

    save_edge_list(train_edges, "train_edges.json")
    save_edge_list(val_edges, "val_edges.json")
    save_edge_list(test_edges, "test_edges.json")
    print(f"\n💾 Saved datasets to {datasets_dir}")
    print(f"   - train_edges.json: {len(train_edges)} edges")
    print(f"   - val_edges.json: {len(val_edges)} edges")
    print(f"   - test_edges.json: {len(test_edges)} edges")

    # OPTIMIZATION: Pre-build graphs ONCE to avoid rebuilding 3x in setup()
    # This is a major bottleneck when DataModule is instantiated multiple times in Optuna loops
    print(f"\n" + "=" * 70)
    print("PRE-BUILDING PyG GRAPHS (Optimization: avoid 3x rebuild in setup())")
    print("=" * 70)

    graph_cache = {}
    node_to_idx_cache = {}

    # Build a temporary dataset just to access _build_graph_data method
    temp_dataset = EdgeScoringDataset(
        train_edges,  # Only used for gid extraction
        graphs_dict,
        pooled_df,
        feature_cols,
        node_features_dict,
    )

    for gid in graphs_dict.keys():
        data, node_to_idx = temp_dataset._build_graph_data(gid)
        graph_cache[gid] = data
        node_to_idx_cache[gid] = node_to_idx

    print(f"   ✅ Pre-built graphs cached ({len(graph_cache)} graphs)")
    print(f"   ℹ️  Each DataModule.setup() will now reuse these instead of rebuilding")

    # PlateauCallback: Stop study when no improvement for patience trials (from XGB implementation)
    class PlateauCallback:
        """Stop Optuna study when plateau detected (no improvement for N trials).

        Copied from edge_scores.py XGB implementation for consistency.

        Args:
            patience: Stop after N consecutive trials without improvement
            min_trials: Require minimum trials before stopping (avoid premature stop)
            min_delta: Improvement threshold (avoid stopping due to noise)
        """

        def __init__(self, patience=10, min_trials=30, min_delta=0.0001):
            self.patience = patience
            self.min_trials = min_trials
            self.min_delta = min_delta
            self.best_value = None
            self.trials_without_improvement = 0

        def __call__(self, study, trial):
            # Don't stop until we've seen enough trials
            if len(study.trials) < self.min_trials:
                return

            current_best = study.best_value
            if self.best_value is None:
                self.best_value = current_best
                return

            # Only count as improvement if it exceeds min_delta threshold
            if current_best > self.best_value + self.min_delta:
                self.best_value = current_best
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1

            if self.trials_without_improvement >= self.patience:
                print(f"   Early stopping: no improvement for {self.patience} trials")
                study.stop()

    # Optuna objective function
    def objective(trial):
        # Suggest hyperparameters (tuned ranges based on GNN best practices)
        hidden_dim = trial.suggest_categorical(
            "hidden_dim", hidden_dim_range
        )  # Reduced: dropped 256
        num_layers = trial.suggest_int(
            "num_layers", 2, 3
        )  # Restrict to 2-3 layers to prevent overfitting
        dropout_rate = trial.suggest_float(
            "dropout_rate", 0.2, 0.6
        )  # More aggressive dropout (0.2-0.6)
        edge_dropout_rate = trial.suggest_float(
            "edge_dropout_rate", 0.2, 0.6
        )  # More aggressive edge dropout
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 5e-3, log=True
        )  # Wider range
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-4, 1e-2, log=True
        )  # Stronger L2 regularization (1e-4 to 1e-2)

        # Create datamodule with pre-built graphs (reuses graph_cache to avoid rebuilding)
        datamodule = EdgeScoringDataModule(
            train_edges,
            val_edges,
            test_edges,
            graphs_dict,
            pooled_df,
            feature_cols,
            node_features_dict,
            graph_cache=graph_cache,
            node_to_idx_cache=node_to_idx_cache,
            num_workers=num_workers,
        )

        # Create model
        model = EdgeScorerLightning(
            node_feature_dim,
            edge_feature_dim,
            hidden_dim,
            num_layers,
            dropout_rate,
            edge_dropout_rate,
            learning_rate,
            weight_decay,
            pos_weight,
        )

        # Setup callbacks
        checkpoint_cb = ModelCheckpoint(
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            dirpath=model_dir / f"trial_{trial.number}",
            filename="best",
        )
        early_stop_cb = EarlyStopping(monitor="val_auc", patience=patience, mode="max")
        prune_cb = PyTorchLightningPruningCallback(trial, monitor="val_auc")

        logger = TensorBoardLogger(
            save_dir=model_dir / "tb_logs", name=f"trial_{trial.number}"
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_cb, early_stop_cb, prune_cb],
            logger=logger,
            accelerator=(
                "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"
            ),
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=False,
            gradient_clip_val=1.0,  # Clip gradients to prevent exploding gradients
        )

        try:
            trainer.fit(model, datamodule)
            return trainer.callback_metrics.get("val_auc", 0.0).item()
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    print(f"\n" + "=" * 70)
    print("STEP 5: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    if not optuna_enabled:
        # BRANCH: No Optuna - load hyperparameters from config file
        print("⚠️  Optuna disabled - loading hyperparameters from config")

        if hyperparams_config is None:
            raise ValueError(
                "Optuna disabled but --hyperparams-config not provided. "
                "Usage: --optuna-disabled --hyperparams-config path/to/params.json"
            )

        if not os.path.exists(hyperparams_config):
            raise FileNotFoundError(
                f"Hyperparams config not found: {hyperparams_config}"
            )

        with open(hyperparams_config, "r") as f:
            best_params = json.load(f)

        # Validate required fields for GNN
        required_fields = {
            "hidden_dim",
            "num_layers",
            "dropout_rate",
            "edge_dropout_rate",
            "learning_rate",
            "weight_decay",
        }
        missing = required_fields - set(best_params.keys())
        if missing:
            raise ValueError(
                f"Hyperparams config missing required fields: {missing}\n"
                f"Required: {required_fields}\n"
                f"Config has: {set(best_params.keys())}"
            )

        print(f"✅ Loaded hyperparameters from {hyperparams_config}")
        print(f"Hyperparameters: {json.dumps(best_params, indent=2)}")

    else:
        # BRANCH: Optuna enabled - hyperparameter search
        print("Running Optuna hyperparameter search...")

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )

        # Create plateau callback with function args
        plateau_callback = PlateauCallback(
            patience=plateau_patience,
            min_trials=plateau_min_trials,
            min_delta=plateau_min_delta,
        )

        study.optimize(
            objective,
            n_trials=max_trials,
            show_progress_bar=True,
            callbacks=[plateau_callback],
        )

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best validation AUC: {study.best_value:.4f}")
        print("Best hyperparameters:")
        print(json.dumps(study.best_params, indent=2))

        # Save best params
        with open(model_dir / "best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=4)

        best_params = study.best_params

        # Save optuna results for later reference
        optuna_results = {
            "best_trial": study.best_trial.number,
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }
        with open(model_dir / "optuna_results.json", "w") as f:
            json.dump(optuna_results, f, indent=4)
        print(f"💾 Saved Optuna results to {model_dir / 'optuna_results.json'}")

    # Retrain with best params and evaluate on test set
    print(f"\n" + "=" * 70)
    print("STEP 6: RETRAINING WITH BEST HYPERPARAMETERS")
    print("=" * 70)

    bp = best_params

    # Save best params and optuna results BEFORE retraining (backup)
    if optuna_enabled:
        # Already saved above in optuna branch
        pass
    else:
        # Save defaults for non-Optuna case
        with open(model_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)

    datamodule = EdgeScoringDataModule(
        train_edges,
        val_edges,
        test_edges,
        graphs_dict,
        pooled_df,
        feature_cols,
        node_features_dict,
        graph_cache=graph_cache,
        node_to_idx_cache=node_to_idx_cache,
        num_workers=num_workers,
    )

    best_model = EdgeScorerLightning(
        node_feature_dim,
        edge_feature_dim,
        bp["hidden_dim"],
        bp["num_layers"],
        bp["dropout_rate"],
        bp["edge_dropout_rate"],
        bp["learning_rate"],
        bp["weight_decay"],
        pos_weight,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        dirpath=model_dir,
        filename="best_model",
    )
    early_stop_cb = EarlyStopping(monitor="val_auc", patience=patience, mode="max")
    logger = TensorBoardLogger(save_dir=model_dir / "tb_logs", name="best_run")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        accelerator="gpu" if device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,  # Clip gradients to prevent exploding gradients
    )

    trainer.fit(best_model, datamodule)

    # Test the model and save predictions (Issue #7: Save prediction files)
    print(f"\n" + "=" * 70)
    print("STEP 7: EVALUATING ON TEST SET")
    print("=" * 70)

    # Get test predictions
    best_model.eval()
    device = next(best_model.parameters()).device
    test_loader = datamodule.test_dataloader()

    all_preds = []
    all_labels = []
    all_graph_ids = []
    all_edges = []

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch tuple: (batched_graph, target_edges, labels)
            batched_graph, target_edges, labels = batch

            # Move batch to the model device to avoid CPU/GPU mismatch at eval
            batched_graph = batched_graph.to(device)
            target_edges = target_edges.to(device)
            labels = labels.to(device)

            logits = best_model(
                batched_graph.x,
                batched_graph.edge_index,
                batched_graph.edge_attr,
                target_edges,
            )
            preds = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_edges.extend(target_edges.cpu().numpy())
            # Reconstruct graph_ids: first element of batch data stores the graph_id
            # Each batch contains edges from a single graph
            if hasattr(batched_graph, "graph_id"):
                graph_id = batched_graph.graph_id
            else:
                # Fallback: use index from test_edges
                graph_id = (
                    test_edges[len(all_graph_ids)][0]
                    if len(all_graph_ids) < len(test_edges)
                    else "unknown"
                )
            all_graph_ids.extend([graph_id] * len(preds))

    # Compute test metrics
    test_auc = roc_auc_score(all_labels, all_preds)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    test_f1 = f1_score(all_labels, preds_binary, zero_division=0)
    test_precision = precision_score(all_labels, preds_binary, zero_division=0)
    test_recall = recall_score(all_labels, preds_binary, zero_division=0)

    print(f"\nTest Set Performance:")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")

    # Save predictions (Issue #7)
    predictions_df = pd.DataFrame(
        {
            "graph_id": all_graph_ids,
            "edge_u": [e[0] for e in all_edges],
            "edge_v": [e[1] for e in all_edges],
            "label": all_labels,
            "prediction": all_preds,
            "prediction_binary": preds_binary,
        }
    )

    predictions_path = model_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n💾 Saved test predictions to {predictions_path}")

    # Save test metrics
    test_metrics = {
        "auc": float(test_auc),
        "f1": float(test_f1),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "n_test": len(all_labels),
        "n_positive": int(sum(all_labels)),
        "n_negative": int(len(all_labels) - sum(all_labels)),
    }

    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Plot ROC curve
    print(f"\n📊 Generating ROC curve...")
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

    # Save ROC data for later editing
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(test_auc),
    }
    with open(model_dir / "roc_curve_data.json", "w") as f:
        json.dump(roc_data, f, indent=4)

    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - GNN Edge Scoring", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_plot_path = model_dir / "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   💾 Saved ROC curve to {roc_plot_path}")
    print(f"   💾 Saved ROC data to {model_dir / 'roc_curve_data.json'}")

    # Create symlink to latest model for easy access
    latest_link = Path(output_dir) / collection / "model_gnn_latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(model_dir.name, target_is_directory=True)
    print(f"   🔗 Created symlink: {latest_link} -> {model_dir.name}")

    print(f"\n✅ Training complete. Model saved to {model_dir}")

    return model_dir


def score_manifest(
    manifest_path,
    collection,
    output_dir,
    device,
    model_dir=None,
    evaluate=False,
    seed=DEFAULT_SEED,
):
    """Score all graphs in manifest and optionally evaluate if labels (T.pkl) available.

    Args:
        manifest_path: Path to manifest JSON
        collection: Collection name
        output_dir: Output directory
        device: Device for inference
        model_dir: Model directory (optional)
        evaluate: If True and T.pkl available, compute metrics/ROC for test edges
        seed: Random seed

    Returns:
        List of score paths written
    """
    set_seed(seed)

    if model_dir is None:
        # Use the latest model symlink
        model_dir = Path(output_dir) / collection / "model_gnn_latest"
        if not model_dir.exists():
            # Fallback: find newest timestamped model
            model_dirs = list((Path(output_dir) / collection).glob("model_gnn_*"))
            model_dirs = [d for d in model_dirs if d.is_dir() and not d.is_symlink()]
            if not model_dirs:
                raise FileNotFoundError(
                    f"No model directory found in {Path(output_dir) / collection}"
                )
            model_dir = max(model_dirs, key=lambda d: d.name)
            print(f"Using newest model: {model_dir}")
    else:
        model_dir = Path(model_dir)

    print(f"Loading model from {model_dir}...")

    # Load artifacts (both edge and node scalers)
    with open(model_dir / "edge_scaler.pkl", "rb") as f:
        edge_scaler = pickle.load(f)
    with open(model_dir / "node_scaler.pkl", "rb") as f:
        node_scaler = pickle.load(f)
    with open(model_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(model_dir / "node_feature_cols.json") as f:
        node_feature_cols = json.load(f)
    with open(model_dir / "dims.json") as f:
        dims = json.load(f)
    with open(model_dir / "best_params.json") as f:
        best_params = json.load(f)

    # Load checkpoint
    ckpt_path = model_dir / "best_model.ckpt"
    if not ckpt_path.exists():
        ckpts = list(model_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint in {model_dir}")
        ckpt_path = ckpts[0]

    model = EdgeScorerLightning.load_from_checkpoint(
        str(ckpt_path),
        node_feature_dim=dims["node_feature_dim"],
        edge_feature_dim=dims["edge_feature_dim"],
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout_rate=best_params["dropout_rate"],
        edge_dropout_rate=0.0,  # No dropout for inference
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        pos_weight=dims["pos_weight"],
    )

    model.to(device)  # Move to device FIRST
    model.eval()  # Set eval mode SECOND (ensures correct dropout/norm behavior)

    # Load manifest and score graphs
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [e for e in manifest if e.get("collection") == collection]

    if not entries:
        print(f"WARNING: No entries found for collection '{collection}' in manifest")
        return []

    print(f"\nScoring {len(entries)} graphs...")

    # For evaluation mode: collect all predictions and labels
    all_eval_preds = []
    all_eval_labels = []
    all_eval_graph_ids = []
    written_paths = []

    for entry in entries:
        graph_id = entry["graph_id"]
        print(f"  Scoring {graph_id}...", end="", flush=True)

        # Load data
        G = load_graph(entry["G_path"])
        T = load_graph(entry["T_path"]) if evaluate and "T_path" in entry else None
        node_df, edge_df = load_features(
            entry["node_features_path"], entry["edge_features_path"]
        )

        # Prepare features
        processed_edge_df, _ = prepare_features(node_df, edge_df)

        # Scale edge features with edge_scaler
        processed_edge_df[feature_cols] = edge_scaler.transform(
            processed_edge_df[feature_cols].values
        )

        # Scale node features with node_scaler
        if node_feature_cols:
            # Add any missing columns with zeros and drop any extras to match training schema
            missing_cols = [c for c in node_feature_cols if c not in node_df.columns]
            for c in missing_cols:
                node_df[c] = 0.0
            node_df = node_df.reindex(columns=node_feature_cols, fill_value=0.0)
            node_df[node_feature_cols] = node_scaler.transform(
                node_df[node_feature_cols].values
            )

        # Build PyG data for full graph
        node_ids = list(G.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        node_embeddings = []
        for node_id in node_ids:
            if node_id in node_df.index:
                node_embeddings.append(node_df.loc[node_id].values)
            else:
                node_embeddings.append(np.zeros(node_df.shape[1]))

        node_embeddings = torch.tensor(
            np.array(node_embeddings), dtype=torch.float32
        ).to(device)

        edges = list(G.edges())
        edge_index = (
            torch.tensor(
                [[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long
            )
            .t()
            .contiguous()
            .to(device)
        )

        edge_attr_list = []
        for u, v in edges:
            try:
                feat = processed_edge_df.loc[(u, v), feature_cols].values
            except KeyError:
                feat = np.zeros(len(feature_cols))
            edge_attr_list.append(feat)

        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float32).to(
            device
        )

        target_edges = torch.tensor(
            [[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long
        ).to(device)

        # Score in batches
        all_scores = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(target_edges), batch_size):
                batch_target = target_edges[i : i + batch_size]
                logits = model(node_embeddings, edge_index, edge_attr, batch_target)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.extend(scores)

        # Save scores
        score_df = pd.DataFrame(
            {
                "source": [u for u, v in edges],
                "target": [v for u, v in edges],
                "score": all_scores,
            }
        )

        output_graph_dir = Path(output_dir) / collection / graph_id / "scores_gnn"
        output_graph_dir.mkdir(parents=True, exist_ok=True)
        score_path = output_graph_dir / "edge_scores.csv"
        score_df.to_csv(score_path, index=False)
        written_paths.append(str(score_path))

        # Update manifest entry with score path
        entry["score_path"] = str(score_path)

        print(f" ✓ {len(score_df)} edges", end="")

        # If evaluating and T available: collect predictions for test edges only
        if evaluate and T is not None:
            # Determine if frozenset needed
            G_is_undirected = not G.is_directed()
            T_is_undirected = not T.is_directed()
            use_frozenset = G_is_undirected or T_is_undirected

            # Build T edge keys
            T_edge_keys = {
                frozenset({u, v}) if use_frozenset else (u, v) for u, v in T.edges()
            }

            # Collect labels and predictions for all edges
            for (u, v), score in zip(edges, all_scores):
                key = frozenset({u, v}) if use_frozenset else (u, v)
                label = 1 if key in T_edge_keys else 0
                all_eval_preds.append(score)
                all_eval_labels.append(label)
                all_eval_graph_ids.append(graph_id)

            print(f" [eval: {len(edges)} edges]", end="")

        print()  # Newline

    # Write updated manifest with score paths
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Scoring complete. Updated manifest with score paths.")

    # Generate evaluation metrics and plots if in evaluate mode
    if evaluate and len(all_eval_labels) > 0:
        print(f"\n{'='*70}")
        print("EVALUATION METRICS (Test Edges)")
        print(f"{'='*70}")

        eval_auc = roc_auc_score(all_eval_labels, all_eval_preds)
        eval_preds_binary = (np.array(all_eval_preds) > 0.5).astype(int)
        eval_f1 = f1_score(all_eval_labels, eval_preds_binary, zero_division=0)
        eval_precision = precision_score(
            all_eval_labels, eval_preds_binary, zero_division=0
        )
        eval_recall = recall_score(all_eval_labels, eval_preds_binary, zero_division=0)

        print(f"\nTest Set Performance:")
        print(f"  AUC:       {eval_auc:.4f}")
        print(f"  F1:        {eval_f1:.4f}")
        print(f"  Precision: {eval_precision:.4f}")
        print(f"  Recall:    {eval_recall:.4f}")
        print(f"  N edges:   {len(all_eval_labels)}")
        print(
            f"  N positive: {sum(all_eval_labels)} ({100*sum(all_eval_labels)/len(all_eval_labels):.1f}%)"
        )

        # Save evaluation predictions
        eval_predictions_df = pd.DataFrame(
            {
                "graph_id": all_eval_graph_ids,
                "label": all_eval_labels,
                "prediction": all_eval_preds,
                "prediction_binary": eval_preds_binary,
            }
        )

        eval_dir = model_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        manifest_name = Path(manifest_path).stem
        eval_pred_path = eval_dir / f"{manifest_name}_predictions.csv"
        eval_predictions_df.to_csv(eval_pred_path, index=False)
        print(f"\n💾 Saved predictions to {eval_pred_path}")

        # Save evaluation metrics
        eval_metrics = {
            "manifest": str(manifest_path),
            "auc": float(eval_auc),
            "f1": float(eval_f1),
            "precision": float(eval_precision),
            "recall": float(eval_recall),
            "n_edges": len(all_eval_labels),
            "n_positive": int(sum(all_eval_labels)),
            "n_negative": int(len(all_eval_labels) - sum(all_eval_labels)),
        }

        eval_metrics_path = eval_dir / f"{manifest_name}_metrics.json"
        with open(eval_metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=4)

        # Plot ROC curve
        print(f"\n📊 Generating ROC curve...")
        fpr, tpr, thresholds = roc_curve(all_eval_labels, all_eval_preds)

        # Save ROC data
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(eval_auc),
        }

        roc_data_path = eval_dir / f"{manifest_name}_roc_data.json"
        with open(roc_data_path, "w") as f:
            json.dump(roc_data, f, indent=4)

        # Create ROC plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {eval_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"ROC Curve - {manifest_name}", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        roc_plot_path = eval_dir / f"{manifest_name}_roc_curve.png"
        plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   💾 Saved ROC curve to {roc_plot_path}")
        print(f"   💾 Saved ROC data to {roc_data_path}")
        print(f"   💾 Saved metrics to {eval_metrics_path}")

    return written_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN-based Edge Scoring for Hierarchy Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "score"], required=True, help="train or score mode"
    )
    parser.add_argument("--train_manifest", type=str, help="Train manifest JSON")
    parser.add_argument("--test_manifest", type=str, help="Test manifest JSON")
    parser.add_argument("--collection", type=str, help="Collection name")

    # Single graph mode (like XGB)
    parser.add_argument(
        "--gid", type=str, help="Single graph ID (creates temp manifest)"
    )
    parser.add_argument(
        "--graph", type=str, help="Path to G.pkl (for single graph mode)"
    )
    parser.add_argument(
        "--tree", type=str, help="Path to T.pkl (for single graph mode)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="GPU ID to use (0, 1, 2, etc.). -1 means auto-select (GPU if available, else CPU)",
    )
    parser.add_argument(
        "--max_trials", type=int, default=30, help="Optuna hyperparameter search trials"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="Maximum training epochs per trial"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--plateau_patience",
        type=int,
        default=20,
        help="Plateau callback patience (trials with no improvement)",
    )
    parser.add_argument(
        "--plateau_min_trials",
        type=int,
        default=50,
        help="Minimum trials before plateau stop allowed",
    )
    parser.add_argument(
        "--plateau_min_delta",
        type=float,
        default=0.0001,
        help="Minimum improvement threshold for plateau",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker processes"
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="Model directory (score-only)"
    )
    parser.add_argument(
        "--keep_apsp_features",
        action="store_true",
        help="Keep all_pairs_shortest_path_* features (default: skip them)",
    )
    parser.add_argument(
        "--hidden_dim_range",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Hidden dimension search range (space-separated, e.g., --hidden_dim_range 64 96 128 192 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--optuna-disabled",
        action="store_true",
        help="Disable Optuna hyperparameter search (use fixed defaults)",
    )
    parser.add_argument(
        "--hyperparams-config",
        type=str,
        default=None,
        help="Path to JSON config with hyperparameters (required if --optuna-disabled)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Single graph mode: create temp manifest (like XGB)
    temp_manifest_path = None
    if args.gid:
        if args.train_manifest or args.test_manifest:
            print("WARNING: --gid mode ignores --train_manifest and --test_manifest")

        if not args.collection:
            args.collection = ""  # Empty collection

        # Build graph directory path
        graph_dir = Path(args.output_dir) / args.collection / args.gid

        # Create temp manifest entry
        manifest_entry = {
            "graph_id": args.gid,
            "collection": args.collection,
            "node_features_path": str(graph_dir / "features" / "node_features.csv"),
            "edge_features_path": str(graph_dir / "features" / "edge_features.csv"),
        }

        # Add G_path and T_path if provided
        if args.graph:
            manifest_entry["G_path"] = args.graph
        if args.tree:
            manifest_entry["T_path"] = args.tree

        # Create temp manifest file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([manifest_entry], f, indent=2)
            temp_manifest_path = f.name

        # Override manifest args
        if args.mode == "train":
            args.train_manifest = temp_manifest_path
        else:
            args.test_manifest = temp_manifest_path

        print(f"📝 Created temporary manifest for single graph: {args.gid}")

    # Validate collection is provided
    if not args.collection:
        raise ValueError(
            "--collection is required (or use --gid for single graph mode)"
        )

    # Handle device selection with CUDA_VISIBLE_DEVICES
    # Determine the actual device string to use in code ("cuda" or "cpu")
    device = "cpu"
    if args.gpu_id >= 0:
        # User specified a GPU ID like --gpu_id 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu_id}")
        if torch.cuda.is_available():
            device = "cuda"
            print(
                f"🚀 Using GPU: {torch.cuda.get_device_name(0)} (physical GPU {args.gpu_id})"
            )
        else:
            print(
                f"WARNING: GPU {args.gpu_id} requested but CUDA not available. Using CPU."
            )
            device = "cpu"
    else:
        # args.gpu_id == -1, auto-select
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
            device = "cpu"

    if args.mode == "train":
        if not args.train_manifest:
            raise ValueError("--train_manifest required for train mode")

        model_dir = train_with_optuna(
            args.train_manifest,
            args.collection,
            args.output_dir,
            device,
            args.max_trials,
            args.max_epochs,
            args.patience,
            args.num_workers,
            args.plateau_patience,
            args.plateau_min_trials,
            args.plateau_min_delta,
            not args.keep_apsp_features,  # skip_shortest_path defaults to True
            args.hidden_dim_range,
            seed=args.seed,
            hyperparams_config=args.hyperparams_config,
            optuna_enabled=not args.optuna_disabled,
        )

        # Score train graphs (no evaluation - already done on test split)
        print("\n" + "=" * 70)
        print("SCORING TRAIN MANIFEST")
        print("=" * 70)
        score_manifest(
            args.train_manifest,
            args.collection,
            args.output_dir,
            device,
            model_dir,
            evaluate=False,
            seed=args.seed,
        )

        # Score test graphs with evaluation if provided
        if args.test_manifest:
            print("\n" + "=" * 70)
            print("SCORING TEST MANIFEST (with evaluation)")
            print("=" * 70)
            score_manifest(
                args.test_manifest,
                args.collection,
                args.output_dir,
                device,
                model_dir,
                evaluate=True,
                seed=args.seed,
            )

    elif args.mode == "score":
        if args.train_manifest:
            print("SCORING TRAIN MANIFEST")
            score_manifest(
                args.train_manifest,
                args.collection,
                args.output_dir,
                device,
                args.model_dir,
                evaluate=False,
                seed=args.seed,
            )

        if args.test_manifest:
            print("\nSCORING TEST MANIFEST (with evaluation)")
            score_manifest(
                args.test_manifest,
                args.collection,
                args.output_dir,
                device,
                args.model_dir,
                evaluate=True,
                seed=args.seed,
            )

        if not args.train_manifest and not args.test_manifest:
            raise ValueError(
                "At least one of --train_manifest or --test_manifest required for score mode"
            )

    # Clean up temp manifest if created
    if temp_manifest_path:
        import os as os_module

        os_module.unlink(temp_manifest_path)
        print(f"\n🗑️  Cleaned up temporary manifest")

    return 0


if __name__ == "__main__":
    sys.exit(main())
