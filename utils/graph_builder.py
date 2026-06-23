import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
import networkx as nx
from scipy.sparse.linalg import eigsh

def normalize_mesh(vertices):
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    max_dist = np.max(np.sqrt(np.sum(vertices**2, axis=1)))
    vertices = vertices / (max_dist + 1e-8)
    return vertices

def build_local_patches(vertices, k=16):
    kdtree = cKDTree(vertices)
    dists, indices = kdtree.query(vertices, k=k)
    return indices, dists

def voxelize_patch(patch_vertices, grid_size=16):
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    if len(patch_vertices) == 0:
        return voxel_grid
    
    min_val = np.min(patch_vertices, axis=0)
    max_val = np.max(patch_vertices, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-5
    
    normalized_pts = (patch_vertices - min_val) / range_val * (grid_size - 1.001)
    normalized_pts = normalized_pts.astype(np.int32)
    
    voxel_grid[normalized_pts[:, 0], normalized_pts[:, 1], normalized_pts[:, 2]] = 1.0
        
    return voxel_grid

def compute_laplacian_pe(edge_index, num_nodes, k=8):
    if num_nodes <= k:
        k = num_nodes - 1
    if k <= 0:
        return torch.zeros((num_nodes, 8))
        
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    nx_graph = nx.from_scipy_sparse_array(adj)
    L = nx.normalized_laplacian_matrix(nx_graph)
    
    try:
        evals, evecs = eigsh(L, k=k+1, which='SM')
        pe = evecs[:, 1:k+1]
    except:
        pe = np.zeros((num_nodes, k))
        
    if pe.shape[1] < 8:
        pad = np.zeros((num_nodes, 8 - pe.shape[1]))
        pe = np.hstack([pe, pad])
        
    return torch.tensor(pe, dtype=torch.float32)

def process_mesh_to_graph(mesh_path):
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces.")
        
        # Extremely fast simplification for massive meshes (prevents Quadric Decimation hang)
        if len(mesh.triangles) > 100000:
            print("Mesh is huge, performing rapid vertex clustering...")
            voxel_size = np.max(mesh.get_max_bound() - mesh.get_min_bound()) / 50.0
            mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size)
            print(f"After clustering: {len(mesh.triangles)} faces.")
        
        # Downsample the mesh so CPU can process it quickly
        target_triangles = 500 
        if len(mesh.triangles) > target_triangles:
            print("Running quadric decimation...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            print("Decimation complete.")
            
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # Point cloud fallback
        if len(faces) == 0 and len(vertices) > 1000:
            print("Point cloud (no faces) detected. Subsampling to 1000 points...")
            indices = np.random.choice(len(vertices), 1000, replace=False)
            vertices = vertices[indices]
        
        if len(vertices) == 0:
            print(f"Warning: Mesh {mesh_path} has 0 vertices or failed to load.")
            return None
    except Exception as e:
        print(f"Error loading {mesh_path}: {e}")
        return None
    
    vertices = normalize_mesh(vertices)
    num_nodes = vertices.shape[0]
    
    neighbor_indices, _ = build_local_patches(vertices, k=16)
    
    voxels = []
    for i in range(num_nodes):
        patch_verts = vertices[neighbor_indices[i]]
        voxel = voxelize_patch(patch_verts, grid_size=8)
        voxels.append(voxel)
        
    voxels = np.array(voxels)
    voxels = torch.tensor(voxels, dtype=torch.float32).unsqueeze(1)
    
    edges_np = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 0]],
        faces[:, [1, 2]], faces[:, [2, 1]],
        faces[:, [2, 0]], faces[:, [0, 2]]
    ])
    edges_np = np.unique(edges_np, axis=0)
        
    edge_index = torch.tensor(edges_np, dtype=torch.long).t().contiguous()
    pe = compute_laplacian_pe(edge_index, num_nodes, k=8)
    
    data = Data(num_nodes=num_nodes, edge_index=edge_index)
    data.pos = torch.tensor(vertices, dtype=torch.float32)
    data.voxels = voxels
    data.pe = pe
    data.face = torch.tensor(faces, dtype=torch.long)
    
    return data
