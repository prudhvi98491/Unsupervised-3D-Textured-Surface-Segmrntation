import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt

def render_segmented_mesh(verts, faces, labels, num_clusters=2):
    """
    Renders the mesh using open3d based on labels.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    colors = np.zeros((verts.shape[0], 3))
    
    color_palette = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 0.5],  # Purple
        [0.0, 0.5, 0.5],  # Teal
        [0.5, 0.5, 0.5],  # Gray
    ]
    
    for i in range(num_clusters):
        c = color_palette[i % len(color_palette)]
        colors[labels == i] = c
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def plot_training_curves(train_losses, val_losses, accuracies, save_path="training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.title('Segmentation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
