# Unsupervised 3D Textured Surface Segmentation

This repository implements a **Guided Contrastive Graph Transformer** for unsupervised 3D textured surface segmentation. The model segments 3D meshes by learning high-quality representation vectors using graph neural networks combined with multiview contrastive learning and self-guided clustering.

---

## 🚀 Key Features

* **Voxel Patch Encoder**: Extracts deep visual and geometric features from local 3D patches.
* **Multiview Contrastive Learning**: Uses graph structure and feature augmentations trained with NT-Xent loss.
* **Unsupervised Self-Guidance**: Automatically generates high-quality pseudo-labels via K-Means clustering to refine decision boundaries.
* **Gradio 3D UI**: An interactive web-based interface that allows users to upload `.obj` or `.ply` 3D files, specify cluster sizes, and visualize segmented 3D models directly in the browser.

---

## 📁 Repository Structure

```
├── models/
│   ├── patch_encoder.py       # Encoder for extracting local 3D patch features
│   ├── graph_transformer.py   # Graph Transformer network for mesh node embeddings
│   └── full_model.py          # Unified model compiling encoders, graph layers, and loss functions
├── utils/
│   ├── data_loader.py         # PyTorch Geometric dataloader for 3D meshes
│   ├── graph_builder.py       # Builds graph representation from mesh geometry
│   ├── augmentation.py        # Graph augmentations (feature drop, edge drop)
│   └── visualization.py       # Training plots and curve visualization
├── app.py                     # Gradio-based 3D Web UI
├── train.py                   # Unsupervised training pipeline
├── inference.py               # Mesh inference and segmentation logic
├── main.py                    # Unified entrypoint CLI script
├── requirements.txt           # Python library dependencies
└── .gitignore                 # Excluded temporary files
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jayakarthik6788/Unsupervised-3D-Textured-Surface-Segmrntation.git
   cd Unsupervised-3D-Textured-Surface-Segmrntation
   ```

2. **Install requirements**:
   Ensure you have Python 3.8+ and PyTorch installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

The `main.py` script serves as a unified entrypoint for training, inference, and launching the Web UI.

### 1. Training the Model
To start training the model on a dataset:
```bash
python main.py --mode train --data_dir /path/to/your/dataset
```

### 2. Running Inference
To run segmentation inference on a single mesh file:
```bash
python main.py --mode infer --mesh /path/to/mesh.obj
```

### 3. Launching the Web UI
To launch the interactive Gradio web application for 3D mesh segmentation and visualization:
```bash
python main.py --mode ui
```
Once launched, open your web browser and navigate to `http://127.0.0.1:7860/` to use the application.

---

## 📊 Methodology

1. **Graph Representation**: The mesh vertices and textured coordinates are modeled as nodes in a graph, with mesh edges forming the graph connections.
2. **Feature Extraction**: Local 3D patch features are encoded using a 3D CNN patch encoder.
3. **Contrastive Pre-training**: Graph Transformer is pre-trained by maximizing agreement between different augmented views of the same mesh.
4. **Pseudo-Label Clustering**: A clustering mechanism dynamically assigns pseudo-labels to guide classification training iteratively.
