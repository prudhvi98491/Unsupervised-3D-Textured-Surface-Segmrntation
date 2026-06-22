import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from models.full_model import FullGuidedModel
from utils.graph_builder import process_mesh_to_graph
from utils.visualization import render_segmented_mesh

def run_inference(mesh_path, model_path='models/best_model.pth', num_clusters=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FullGuidedModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded customized pre-trained model.")
    except Exception as e:
        print("No pre-trained model found. Running with random weights for demo...")
        
    model.eval()
    
    print(f"Loading and processing mesh: {mesh_path}")
    data = process_mesh_to_graph(mesh_path)
    data = data.to(device)
    
    print("Running inference...")
    with torch.no_grad():
        voxels = data.voxels
        if voxels.dim() == 6:    
            b, n, c, h, w, d = voxels.shape
            voxels = voxels.view(b*n, c, h, w, d)
            
        node_feats = model.extract_node_features(voxels)
        embeddings = model.forward_from_node_features(node_feats, data.pe, data.edge_index, None)
        
    embeddings_np = embeddings.cpu().numpy()
    
    print(f"Clustering features into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42).fit(embeddings_np)
    labels = kmeans.labels_
    
    print("Saving segmentation result...")
    faces = data.face.cpu().numpy() if hasattr(data, 'face') else []
    segmented_mesh = render_segmented_mesh(data.pos.cpu().numpy(), faces, labels, num_clusters)
    
    out_path = mesh_path.replace('.obj', '_segmented.glb').replace('.ply', '_segmented.glb')
    if out_path == mesh_path:
        out_path = mesh_path + "_segmented.glb"
        
    o3d.io.write_triangle_mesh(out_path, segmented_mesh)
    
    total = len(labels)
    
    cluster_distribution = {}
    for i in range(num_clusters):
        cluster_distribution[i] = ((labels == i).sum() / total) * 100
        
    stats = {
        'total_vertices': total,
        'cluster_distribution': cluster_distribution,
        'output_file': out_path
    }
    
    return stats, out_path
