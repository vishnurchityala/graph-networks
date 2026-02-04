import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GATConv

class GraphModule(nn.Module):
    """
    Self-contained GAT module with stored training graph.
    Forward function can take new embeddings and return contextualized embeddings.
    """
    def __init__(self, node_features, k=20, hidden_dim=32, out_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.register_buffer("node_features", torch.tensor(node_features, dtype=torch.float32))
        self.k = k
        self.edge_index = self._build_knn_graph(node_features, k)
        in_dim = node_features.shape[1]
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim*heads, out_dim, heads=1, concat=False, dropout=dropout)

    def _build_knn_graph(self, embeddings, k):
        embeddings_norm = normalize(embeddings, axis=1)
        knn = kneighbors_graph(embeddings_norm, n_neighbors=k, mode='connectivity', include_self=True)
        knn = 0.5 * (knn + knn.T)
        coo = knn.tocoo()
        return torch.tensor([coo.row, coo.col], dtype=torch.long)

    def forward(self, new_node_features, k=None):
        """
        new_node_features: tensor of shape (num_new_nodes, feature_dim)
        Returns: contextualized embeddings for new nodes
        """
        k = k or self.k
        device = self.node_features.device
        training_nodes = self.node_features
        new_nodes = new_node_features.to(device)

        all_nodes = torch.cat([training_nodes, new_nodes], dim=0)

        num_training = training_nodes.shape[0]
        num_new = new_nodes.shape[0]
        edge_rows = []
        edge_cols = []

        sim = cosine_similarity(new_nodes.cpu().numpy(), training_nodes.cpu().numpy())
        for i in range(num_new):
            topk_idx = np.argsort(sim[i])[-k:]
            new_idx = num_training + i
            edge_rows.extend([new_idx]*k + topk_idx.tolist())
            edge_cols.extend(topk_idx.tolist() + [new_idx]*k)

        existing_edges = self.edge_index
        new_edges = torch.tensor([edge_rows, edge_cols], dtype=torch.long).to(device)
        combined_edge_index = torch.cat([existing_edges.to(device), new_edges], dim=1)

        x = self.gat1(all_nodes, combined_edge_index)
        x = F.elu(x)
        x = self.gat2(x, combined_edge_index)

        return x[-num_new:]

    def save(self, path):
        torch.save({
            "gat_state_dict": self.state_dict(),
            "node_features": self.node_features,
            "edge_index": self.edge_index
        }, path)
        print(f"GraphModule saved at {path}")

    @classmethod
    def load(cls, path, k=20, hidden_dim=32, out_dim=64, heads=4, dropout=0.2, device=None):
        checkpoint = torch.load(path, map_location=device or "cpu")
        node_features = checkpoint["node_features"].cpu().numpy()
        model = cls(node_features, k=k, hidden_dim=hidden_dim, out_dim=out_dim, heads=heads, dropout=dropout)
        model.load_state_dict(checkpoint["gat_state_dict"])
        if device:
            model.to(device)
        print(f"GraphModule loaded from {path}")
        return model