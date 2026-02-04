import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GATConv

class GraphModule(nn.Module):
    """
    Graph Module with k-NN graph creation, 2-layer GAT, training, and weight save/load.
    """
    def __init__(self, in_dim, hidden_dim=32, out_dim=64, heads=4, dropout=0.2, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim*heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.to(self.device)

    def forward(self, x, edge_index):
        """
        Forward pass through 2-layer GAT.
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.normalize(x, p=2, dim=1)  # optional normalization
        return x

    @staticmethod
    def create_knn_graph(embeddings, k=20):
        """
        Create symmetric k-NN graph from embeddings.
        """
        embeddings_norm = normalize(embeddings, axis=1)
        knn = kneighbors_graph(embeddings_norm, n_neighbors=k, mode='connectivity', include_self=True)
        knn = 0.5 * (knn + knn.T)
        coo = knn.tocoo()
        edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
        return edge_index

    def save_weights(self, path):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
        print(f"[GraphModule] GAT weights saved at {path}")

    def load_weights(self, path, map_location=None):
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.to(self.device)
        print(f"[GraphModule] GAT weights loaded from {path}")

    def train_gat(self, x, edge_index, labels, mask=None, lr=0.005, weight_decay=5e-4,
                  epochs=200, verbose=True, eval_fn=None):
        """
        Train the GAT on node labels.
        """
        self.train()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        labels = labels.to(self.device)

        if mask is None:
            mask = torch.ones(len(labels), dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            out = self.forward(x, edge_index)
            loss = F.cross_entropy(out[mask], labels[mask])
            loss.backward()
            optimizer.step()

            if verbose and epoch % 20 == 0:
                pred = out.argmax(dim=1)
                acc = (pred[mask] == labels[mask]).float().mean().item()
                log_msg = f"[Epoch {epoch}/{epochs}] Loss: {loss.item():.4f}, Train Acc: {acc:.4f}"
                if eval_fn is not None:
                    log_msg += f", Eval: {eval_fn(self, out)}"
                print(log_msg)

        return self.forward(x, edge_index)
