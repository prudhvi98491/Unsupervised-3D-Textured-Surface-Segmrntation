import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.graph_builder import process_mesh_to_graph
from torch_geometric.data import Batch

class SHREC2026Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.files = glob.glob(os.path.join(self.root_dir, '*.obj')) + glob.glob(os.path.join(self.root_dir, '*.ply'))
        self.files.sort()
        
        if len(self.files) == 0:
            print("WARNING: No meshes found in SHREC 2026 dataset folder. Generating a dummy mesh...")
            os.makedirs(self.root_dir, exist_ok=True)
            import trimesh
            dummy = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            dummy.export(os.path.join(self.root_dir, "dummy.ply"))
            self.files = [os.path.join(self.root_dir, "dummy.ply")]
            
        self.files = self.files[:10]  # Test Mode slice
        n_files = len(self.files)
        np.random.seed(42)
        indices = np.random.permutation(n_files)
        
        train_end = int(0.8 * n_files)
        val_end = int(0.9 * n_files)
        
        if split == 'train':
            self.files = [self.files[i] for i in indices[:max(1, train_end)]]
        elif split == 'val':
            self.files = [self.files[i] for i in indices[min(train_end, n_files-1):max(val_end, train_end+1)]]
        else: # test
            self.files = [self.files[i] for i in indices[min(val_end, n_files-1):]]
            
        if len(self.files) == 0:
            self.files = [glob.glob(os.path.join(self.root_dir, '*.ply'))[0]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = process_mesh_to_graph(file_path)
        
        while data is None:
            idx = (idx + 1) % len(self.files)
            file_path = self.files[idx]
            data = process_mesh_to_graph(file_path)
        
        if self.transform:
            data = self.transform(data)
            
        return data

def augmentation_transform(data):
    scale = np.random.uniform(0.8, 1.2)
    data.pos = data.pos * scale
    
    angle = np.random.uniform(-np.pi/6, np.pi/6)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = torch.tensor([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=torch.float32)
    
    data.pos = torch.matmul(data.pos, rot_matrix)
    return data

def get_dataloaders(root_dir=r'C:\Users\jayak\Downloads\All_faces_sculpted_derivatives', batch_size=4):
    train_dataset = SHREC2026Dataset(root_dir, split='train', transform=augmentation_transform)
    val_dataset = SHREC2026Dataset(root_dir, split='val', transform=None)
    test_dataset = SHREC2026Dataset(root_dir, split='test', transform=None)
    
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader
