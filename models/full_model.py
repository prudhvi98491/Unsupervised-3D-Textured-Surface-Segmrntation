import torch
import torch.nn as nn
import torch.nn.functional as F
from models.patch_encoder import PatchEncoder3D
from models.graph_transformer import GraphTransformerModule

class FullGuidedModel(nn.Module):
    def __init__(self):
        super(FullGuidedModel, self).__init__()
        self.patch_encoder = PatchEncoder3D()
        self.graph_transformer = GraphTransformerModule(in_channels=72)
        
    def extract_node_features(self, voxels, chunk_size=128):
        if voxels.size(0) > chunk_size:
            feats = []
            for i in range(0, voxels.size(0), chunk_size):
                feats.append(self.patch_encoder(voxels[i:i+chunk_size]))
            return torch.cat(feats, dim=0)
        return self.patch_encoder(voxels)
        
    def forward_from_node_features(self, node_features, pe, edge_index, batch=None):
        combined_features = torch.cat([node_features, pe], dim=-1)
        context_embeds = self.graph_transformer(combined_features, edge_index, batch)
        return context_embeds

    def forward(self, data):
        voxels, pe, edge_index, batch = data.voxels, data.pe, data.edge_index, data.batch
        
        node_features = self.extract_node_features(voxels)
        context_embeds = self.forward_from_node_features(node_features, pe, edge_index, batch)
        
        return context_embeds

def nt_xent_loss(z_i, z_j, temperature=0.07):
    z_i = F.normalize(z_i, p=2, dim=1)
    z_j = F.normalize(z_j, p=2, dim=1)
    
    batch_size = z_i.size(0)
    
    sim_matrix = torch.matmul(z_i, z_j.T) / temperature
    
    labels = torch.arange(batch_size, device=z_i.device)
    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_j = F.cross_entropy(sim_matrix.T, labels)
    
    loss = (loss_i + loss_j) / 2
    return loss
    
def smoothness_loss(embeddings, edge_index):
    node_u, node_v = edge_index
    diff = embeddings[node_u] - embeddings[node_v]
    loss = (diff ** 2).sum(dim=-1).mean()
    return loss
