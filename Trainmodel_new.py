import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# -------------------------------
# Global settings
# -------------------------------
DATASET_NAMES = ["MUTAG", "PROTEINS", "ENZYMES", "REDDIT-BINARY"]
GNNS = ["GCN", "GAT", "GraphSAGE"]
EPOCHS = 400
LR_DICT = {
    "GCN": 1e-3,
    "GAT": 1e-3,
    "GraphSAGE": 1e-3,
}
HIDDEN_CHANNELS = 64
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Transform: concatenate degree one-hot
# -------------------------------
class ConcatOneHotDegree:
    def __init__(self, max_degree=128):
        self.max_degree = max_degree

    def __call__(self, data):
        row, _ = data.edge_index
        deg = degree(row, num_nodes=data.num_nodes).long()
        deg = deg.clamp(max=self.max_degree)
        one_hot_deg = torch.nn.functional.one_hot(
            deg, num_classes=self.max_degree + 1
        ).float()

        if data.x is not None:
            data.x = torch.cat([data.x, one_hot_deg], dim=-1)
        else:
            data.x = one_hot_deg

        return data

# -------------------------------
# GNN Model Definition
# -------------------------------
class GNN(torch.nn.Module):
    def __init__(self, name, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.name = name
        self.dropout = 0.5

        if name == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        elif name == "GAT":
            heads = 4
            self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
            self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)

        elif name == "GraphSAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
            self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.lin(x)

# -------------------------------
# Train & Evaluate
# -------------------------------
def train_and_evaluate(model, train_loader, test_loader):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_DICT[model.name],
        weight_decay=5e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for _ in range(EPOCHS):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_pred.extend(out.argmax(dim=1).cpu().tolist())
            y_true.extend(data.y.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1

# -------------------------------
# Loop over all datasets
# -------------------------------
MAX_DEGREE = 128

for dataset_name in DATASET_NAMES:
    print(f"\n\n==============================")
    print(f" Loading dataset: {dataset_name}")
    print(f"==============================")

    SAVE_DIR = f"saved_gnns/{dataset_name}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load dataset
    dataset = TUDataset(
        root="./data",
        name=dataset_name,
        transform=ConcatOneHotDegree(MAX_DEGREE)
    )

    in_channels = dataset.num_features
    out_channels = dataset.num_classes

    labels = [data.y.item() for data in dataset]
    indices = np.arange(len(dataset))

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = dataset[train_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train all GNNs on this dataset
    for gnn_name in GNNS:
        print(f"\nTraining {gnn_name} on {dataset_name}")
        model = GNN(gnn_name, in_channels, HIDDEN_CHANNELS, out_channels)
        acc, f1 = train_and_evaluate(model, train_loader, test_loader)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "gnn_name": gnn_name,
                "dataset": dataset_name,
                "accuracy": acc,
                "f1": f1,
                "in_channels": in_channels,
                "out_channels": out_channels,
            },
            os.path.join(SAVE_DIR, f"{gnn_name}.pt"),
        )

        print(f"Saved {gnn_name} | Acc: {acc:.4f}, F1: {f1:.4f}")
