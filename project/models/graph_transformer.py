import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv, SAGEConv, global_mean_pool

class GraphTransformerModule(nn.Module):
    def __init__(self, in_channels=72, hidden_channels=128, out_channels=256):
        super(GraphTransformerModule, self).__init__()
        
        self.conv1 = TransformerConv(in_channels, hidden_channels // 4, heads=4, dropout=0.1)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // 4, heads=4, dropout=0.1)
        self.conv3 = TransformerConv(hidden_channels, out_channels // 4, heads=4, dropout=0.1)
        
        self.local_attention = GATConv(out_channels, 64 // 4, heads=4, concat=True)
        self.global_proj = nn.Linear(out_channels, 64)
        
        self.topology_sage = SAGEConv(out_channels, out_channels)
        self.topology_proj = nn.Linear(out_channels, 64)
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

    def forward(self, x, edge_index, batch=None):
        num_nodes = x.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
            
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        branch1 = F.relu(self.local_attention(x, edge_index))
        
        global_repr = global_mean_pool(x, batch)
        global_repr_broadcast = global_repr[batch]
        branch2 = F.relu(self.global_proj(global_repr_broadcast))
        
        topology_repr = F.relu(self.topology_sage(x, edge_index))
        branch3 = F.relu(self.topology_proj(topology_repr))
        
        concat_feats = torch.cat([branch1, branch2, branch3], dim=-1)
        out = self.fusion(concat_feats)
        
        return out
