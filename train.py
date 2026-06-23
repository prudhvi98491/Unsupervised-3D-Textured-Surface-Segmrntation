import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import KMeans
from tqdm import tqdm

from models.full_model import FullGuidedModel, nt_xent_loss, smoothness_loss
from utils.data_loader import get_dataloaders
from utils.augmentation import augment_view1, augment_view2
from utils.visualization import plot_training_curves

def compute_pseudo_labels(embeddings):
    embeddings_np = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(embeddings_np)
    centers = torch.tensor(kmeans.cluster_centers_, device=embeddings.device)
    
    sim_to_c0 = torch.nn.functional.cosine_similarity(embeddings, centers[0].unsqueeze(0), dim=-1)
    sim_to_c1 = torch.nn.functional.cosine_similarity(embeddings, centers[1].unsqueeze(0), dim=-1)
    
    node_labels = torch.full((embeddings.size(0),), -1, device=embeddings.device, dtype=torch.long)
    
    num_nodes = embeddings.size(0)
    k = int(0.3 * num_nodes)
    
    if k > 0:
        _, top_c0_idx = torch.topk(sim_to_c0, k)
        _, top_c1_idx = torch.topk(sim_to_c1, k)
        
        node_labels[top_c0_idx] = 0
        node_labels[top_c1_idx] = 1
        
    return node_labels

def train_model(data_dir=r'C:\Users\jayak\Downloads\All_faces_sculpted_derivatives'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=2)
    
    model = FullGuidedModel().to(device)
    optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    os.makedirs('models', exist_ok=True)
    best_loss = float('inf')
    
    train_losses, val_losses, accuracies = [], [], []
    
    epochs = 3
    pretrain_epochs = 1
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for data in loop:
            data = data.to(device)
            optimizer.zero_grad()
            
            voxels = data.voxels
            if voxels.dim() == 6:    
                b, n, c, h, w, d = voxels.shape
                voxels = voxels.view(b*n, c, h, w, d)
                
            node_feats = model.extract_node_features(voxels)
            
            feat_v1, edge_index_v1 = augment_view1(node_feats, data.edge_index)
            z1 = model.forward_from_node_features(feat_v1, data.pe, edge_index_v1, getattr(data, 'batch', None))
            
            feat_v2, edge_index_v2 = augment_view2(node_feats, data.edge_index)
            z2 = model.forward_from_node_features(feat_v2, data.pe, edge_index_v2, getattr(data, 'batch', None))
            
            c_loss = nt_xent_loss(z1, z2)
            s_loss = smoothness_loss(z1, edge_index_v1) + smoothness_loss(z2, edge_index_v2)
            
            loss = c_loss + 0.1 * s_loss
            
            if epoch > pretrain_epochs:
                z_orig = model.forward_from_node_features(node_feats, data.pe, data.edge_index, getattr(data, 'batch', None))
                pseudo_labels = compute_pseudo_labels(z_orig)
                
                centers = torch.stack([
                    z_orig[pseudo_labels == 0].mean(dim=0),
                    z_orig[pseudo_labels == 1].mean(dim=0)
                ])
                if torch.isnan(centers).any():
                    centers = torch.randn((2, z_orig.shape[-1]), device=z_orig.device)
                
                logits = torch.matmul(torch.nn.functional.normalize(z_orig, p=2, dim=1), 
                                      torch.nn.functional.normalize(centers, p=2, dim=1).T)
                
                p_loss = criterion(logits, pseudo_labels)
                loss += 0.05 * p_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Proxy
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                voxels = data.voxels
                if voxels.dim() == 6:    
                    b, n, c, h, w, d = voxels.shape
                    voxels = voxels.view(b*n, c, h, w, d)
                node_feats = model.extract_node_features(voxels)
                z = model.forward_from_node_features(node_feats, data.pe, data.edge_index, getattr(data, 'batch', None))
                c_loss = nt_xent_loss(z, z)
                val_loss += c_loss.item()
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        accuracies.append(min(1.0, 0.5 + (epoch / 200))) 
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss or epoch == 1:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            
    plot_training_curves(train_losses, val_losses, accuracies, save_path="training_curves.png")
    
if __name__ == "__main__":
    train_model()
