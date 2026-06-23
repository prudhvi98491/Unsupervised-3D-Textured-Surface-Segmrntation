import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

def augment_view1(node_features, edge_index, drop_prob=0.10):
    """
    View 1: add Gaussian noise (std=0.01) to node features + randomly drop 10% edges
    """
    noise = torch.randn_like(node_features) * 0.01
    aug_features = node_features + noise
    
    aug_edge_index, _ = dropout_edge(edge_index, p=drop_prob, force_undirected=True)
    
    return aug_features, aug_edge_index

def augment_view2(node_features, edge_index, drop_prob=0.15):
    """
    View 2: randomly rotate patch features + drop 15% edges.
    Here we implement a random rotational permutation/mixing of features as an abstraction 
    for feature rotation since they are embeddings, or just shuffle features slightly.
    """
    # Simulate patch rotation: permute features locally per node by doing a simple mixing
    mix_matrix = torch.eye(node_features.shape[-1], device=node_features.device)
    # add slight random off-diagonal to mix
    mix_matrix += torch.randn_like(mix_matrix) * 0.05
    # normalize columns
    mix_matrix = F.normalize(mix_matrix, p=2, dim=0)
    
    aug_features = torch.matmul(node_features, mix_matrix)
    
    aug_edge_index, _ = dropout_edge(edge_index, p=drop_prob, force_undirected=True)
    
    return aug_features, aug_edge_index
