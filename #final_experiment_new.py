#final_experiment2.0.py
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, GraphMaskExplainer
from torch_geometric.explain.metric import fidelity
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
import os
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

# -------------------------------
# Transform: concatenate degree one-hot
# -------------------------------
class ConcatOneHotDegree(BaseTransform):
    def __init__(self, max_degree=128):
        self.max_degree = max_degree

    def forward(self, data):
        row, _ = data.edge_index
        deg = degree(row, num_nodes=data.num_nodes).long()
        deg = deg.clamp(max=self.max_degree)
        one_hot_deg = torch.nn.functional.one_hot(
            deg, num_classes=self.max_degree + 1
        ).float()
        # Concatenate instead of replacing
        if data.x is not None:
            data.x = torch.cat([data.x, one_hot_deg], dim=-1)
        else:
            data.x = one_hot_deg
        return data

# -------------------------------
# Global experiment settings
# -------------------------------
HIDDEN_CHANNELS = 64
BATCH_SIZE = 32

EXPLAINER_EPOCHS_LIST = [100]
LR_LIST = [1e-2, 1e-3, 1e-4, 5e-4]

datasets = ["REDDIT-BINARY"]
gnns = ["GCN", "GAT", "GraphSAGE"]
explainers_list = ["PGExplainer", "GNNExplainer", "GraphMaskExplainer"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []

# -------------------------------
# Model Definition
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

        else:
            raise ValueError(f"Unknown GNN type {name}")

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
# Progress bar setup
# -------------------------------
TOTAL_RUNS = len(datasets) * len(gnns) * len(explainers_list) * len(EXPLAINER_EPOCHS_LIST) * len(LR_LIST)
master_pbar = tqdm(total=TOTAL_RUNS, desc="Total Progress")

# -------------------------------
# Main loop
# -------------------------------
MAX_DEGREE = 128
for dataset_name in datasets:
    
    # --- Dataset Loading (apply same transform as training) ---
    dataset = TUDataset(root='./data', name=dataset_name, transform=ConcatOneHotDegree(MAX_DEGREE))
    in_channels = dataset.num_features
    out_channels = dataset.num_classes

    # --- Split dataset ---
    labels = [data.y.item() for data in dataset]
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    
    train_dataset = dataset[train_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    fold_idx = 0

    # --- Load pre-trained GNNs ---
    for gnn_name in gnns:
        checkpoint_path = f"saved_gnns/{dataset_name}/{gnn_name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        print("Checkpoint info:", checkpoint["gnn_name"], checkpoint["dataset"])

        # Use dataset-derived in/out channels
        model = GNN(
            gnn_name,
            in_channels,
            HIDDEN_CHANNELS,
            out_channels,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        
        acc = checkpoint["accuracy"]
        f1 = checkpoint["f1"]
        
        # -------------------------------
        # Explainer hyperparameter search
        # -------------------------------
        for exp_name in explainers_list:
            for lr_val_exp in LR_LIST:
                for exp_epochs in EXPLAINER_EPOCHS_LIST:
                    
                    master_pbar.set_description(
                        f"Exp: {dataset_name}/{gnn_name}/{exp_name} E:{exp_epochs} LR:{lr_val_exp}"
                    )
                    
                    model_config = dict(
                        mode='multiclass_classification' if out_channels > 1 else 'binary_classification',
                        task_level='graph',
                        return_type='raw',
                    )
                    
                    explainer_train_time = 0.0
                    explainer = None
                    
                    if exp_name == "PGExplainer":
                        explainer = Explainer(
                            model=model,
                            algorithm=PGExplainer(epochs=exp_epochs, lr=lr_val_exp),
                            explanation_type='phenomenon',
                            edge_mask_type='object',
                            model_config=model_config,
                        )
                        explainer.algorithm.to(device)
                        
                        start = time.time()
                        for epoch in range(exp_epochs):
                            for data in train_loader:
                                data = data.to(device)
                                with torch.no_grad():
                                    target = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
                                explainer.algorithm.train(epoch, model, data.x, data.edge_index, batch=data.batch, target=target)
                        explainer_train_time = time.time() - start
                    
                    elif exp_name == "GNNExplainer":
                        explainer = Explainer(
                            model=model,
                            algorithm=GNNExplainer(epochs=exp_epochs, lr=lr_val_exp),
                            explanation_type='model',
                            edge_mask_type='object',
                            model_config=model_config,
                        )
                    elif exp_name == "GraphMaskExplainer":
                        explainer = Explainer(
                            model=model,
                            algorithm=GraphMaskExplainer(num_layers=2, epochs=exp_epochs, lr=lr_val_exp, log=False),
                            explanation_type='model',
                            edge_mask_type='object',
                            model_config=model_config,
                        )
                    else:
                        raise ValueError(f"Unknown explainer {exp_name}")
                    
                    # --- Evaluation ---
                    fid_plus_list, fid_minus_list, sparsity_list, inf_time_list = [], [], [], []
                    
                    for data in test_dataset:
                        data = data.to(device)
                        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                        
                        start_inf = time.time()
                        
                        if exp_name == "PGExplainer":
                            with torch.no_grad():
                                target = model(data.x, data.edge_index, batch).argmax(dim=-1)
                            explanation = explainer(data.x, data.edge_index, batch=batch, target=target)
                        else:
                            explanation = explainer(data.x, data.edge_index, batch=batch)
                        
                        inf_time = time.time() - start_inf
                        
                        fid_plus, fid_minus = fidelity(explainer, explanation)
                        edge_mask = getattr(explanation, "edge_mask", None)
                        sparsity = 1 - edge_mask.mean().item() if edge_mask is not None else float('nan')
                        
                        fid_plus_list.append(fid_plus)
                        fid_minus_list.append(fid_minus)
                        sparsity_list.append(sparsity)
                        inf_time_list.append(inf_time)
                    
                    results.append({
                        'dataset': dataset_name,
                        'gnn': gnn_name,
                        'explainer': exp_name,
                        'Fold': fold_idx,
                        'GNN_LR': None,
                        'Explainer_Epochs': exp_epochs,
                        'Explainer_LR': lr_val_exp,
                        'Model_Accuracy': acc,
                        'Model_F1': f1,
                        'Fidelity+ (mean)': np.nanmean(fid_plus_list),
                        'Fidelity- (mean)': np.nanmean(fid_minus_list),
                        'Sparsity (mean)': np.nanmean(sparsity_list),
                        'ExplainerTrainTime': explainer_train_time,
                        'ExplainerInferenceTime (mean)': np.nanmean(inf_time_list)
                    })
                    master_pbar.update(1)

# -------------------------------
# Save results
# -------------------------------
master_pbar.close()
df = pd.DataFrame(results)
df.to_csv('ME.csv', index=False)
print("\nME.csv")
