# PROGRESS.md - Misogyny Classification with Graph Attention Networks

**Last Updated**: February 26, 2026  
**Project Status**: Active Development (Underfitting Investigation Phase)

---

## Project Overview

A multimodal deep learning system designed to classify images into 4 misogyny-related categories:
- **Kitchen** (0)
- **Shopping** (1)
- **Working** (2)
- **Leadership** (3)

**Architecture**: Combines BERT text embeddings + CLIP ViT image embeddings with Graph Attention Networks (GAT) for context-aware classification.

---

## Completed Implementation

### 1. **Data Pipeline** 
**Location**: [`data_loader.py`](data_loader.py)

#### Components Implemented:
- **`MisogynyDataset`** class
  - Loads images and captions from CSV
  - Image preprocessing: 224×224 resize + ImageNet normalization (CLIP-specific stats)
  - Returns tuples: `(image_tensor, caption_text, label_int)`
  - Label mapping: kitchen→0, shopping→1, working→2, leadership→3

- **`MisogynyDataLoader`** wrapper
  - Stratified train/test split: 80% training / 20% test
  - Maintains class distribution across splits
  - Batch size: 16 (configurable)
  - DataLoader parameters: num_workers=0, pin_memory=False (defaults)

- **Dataset Source**: [`data_csv.csv`](data_csv.csv)
  - Columns: `[index, image_path, image_caption, image_label]`
  - Images organized in `data/` by category with subdirectories: `different/`, `image/`, `same/`

---

### 2. **Text Embeddings - BERT**
**Location**: [`modules/embedders/bert_embedder.py`](modules/embedders/bert_embedder.py)

#### `BERTEmbedder` Features:
- **Model**: `bert-base-uncased` (pre-trained, frozen)
- **Output Dimension**: 768-dimensional vectors
- **Processing**:
  - Tokenization with padding and truncation enabled
  - Masked average pooling using attention masks
  - L2 normalization on outputs
- **Trainability**: Frozen parameters (`requires_grad=False`)
- **Device Support**: Automatic CUDA/CPU detection

---

### 3. **Image Embeddings - CLIP ViT**
**Location**: [`modules/embedders/clip_embedder.py`](modules/embedders/clip_embedder.py)

#### `OpenClipVitEmbedder` Features:
- **Model**: CLIP ViT-B-32 (Vision Transformer, OpenAI pre-trained)
- **Output Dimension**: 512-dimensional vectors
- **Processing**: L2 normalization
- **Trainability**: Frozen parameters
- **Integration**: Uses `open_clip` library for model/preprocessor loading

---

### 4. **Dimensionality Reduction - PCA**
**Location**: [`modules/layers/pca_layer.py`](modules/layers/pca_layer.py)

#### `PCALayer` Implementation:
- **Mechanism**: Linear transformation using pre-computed PCA components
- **Configurable Output**: Can reduce to any subset of available components
- **Dual Path Setup**:
  - **Text PCA**: BERT 768-dim → 400-dim output
    - Weights: `weights/bert_pca_components.npy`, `weights/bert_pca_mean.npy`
  - **Image PCA**: CLIP 512-dim → 400-dim output
    - Weights: `weights/clip_pca_components.npy`, `weights/clip_pca_mean.npy`
- **Parameters**: Non-trainable (frozen during training)
- **Formula**: `output = (input - mean) @ components.T`

---

### 5. **Linear Discriminant Analysis Layer**
**Location**: [`modules/layers/lda_layer.py`](modules/layers/lda_layer.py)

#### `LDALayer` Implementation:
- **Purpose**: Dimensionality reduction + class discrimination for combined embeddings
- **Mechanism**: Learned transformation from pre-fitted LDA
- **Weights**:
  - `weights/combined_lda_mean.npy`: Input mean (1280-dim)
  - `weights/combined_lda_coef.npy`: LDA projection coefficients
- **Input**: Combined text+image (1280-dim from 768+512 concatenation)
- **Parameters**: Non-trainable

---

### 6. **Graph Attention Network Module**
**Location**: [`modules/layers/graph_layer.py`](modules/layers/graph_layer.py)

#### `GraphModule` Architecture:
- **Graph Construction**:
  - KNN graph with k=20 neighbors (default)
  - Similarity metric: Cosine distance
  - Symmetric, undirected edges
  - Training-time context: Stores all training node features

- **GAT Layers**:
  - **Layer 1**: Input → 32 hidden channels, 4 attention heads, concat
  - **Layer 2**: 128 (32×4) → 64 output channels, 1 attention head
  - **Activation**: ELU between layers
  - **Dropout**: 0.2 (configurable)

- **Inference Pipeline**:
  1. Takes new node embeddings (test samples)
  2. Finds k-nearest training neighbors via cosine similarity
  3. Constructs edges between new nodes and neighbors
  4. Applies both stored training-graph edges + test-training edges
  5. Returns contextualized 64-dim embeddings for test nodes

- **Serialization**: Full `save()` / `load()` checkpoint API
  - Stores: GAT parameters + node features + edge index
  - Loaded weights: `weights/graph_module.pth`

---

### 7. **Classification Head**
**Location**: [`modules/layers/classification_layer.py`](modules/layers/classification_layer.py)

#### `ClassificationLayer` Architecture:
- **Input Dimension**: 164 total
  - Text PCA: 50-dim (configurable, currently 50 from original 400)
  - Image PCA: 50-dim 
  - Graph output: 64-dim
  - LDA output: (varies, typically ~150-200 from combined LDA)

- **Network Structure**:
  ```
  Input (164) 
    ↓ FC1 + BatchNorm + GELU + Dropout(0.3) → (256)
    ↓ FC2 + BatchNorm + GELU + Dropout(0.3) → (128)
    ↓ FC3 + BatchNorm + GELU + Dropout(0.3) → (64)
    ↓ Output Linear → (4 logits)
  ```

- **Regularization**: BatchNorm + GELU + Dropout(0.3) on all hidden layers
- **Output**: Raw logits for 4 classes (cross-entropy loss applied downstream)

---

### 8. **End-to-End Model Pipeline**
**Location**: [`modules/models/misogyny_model.py`](modules/models/misogyny_model.py)

#### `MisogynyModel` (Default)
**Default Data Flow**:
```
Text Input → BERT Embedder (768-dim)
              ↓
           LDA Layer ← Combined with Image (1280-dim total via concat)
              ↓
           Graph Module (64-dim contextualized)

Image Input → CLIP Embedder (512-dim)
              ↓
           LDA Layer (same as above)
              ↓
           Graph Module (64-dim contextualized)

Parallel:
- Text (768) → PCA (50-dim)
- Image (512) → PCA (50-dim)

Fusion:
Text PCA (50) + Image PCA (50) + Graph Output (64) → (164-dim)
              ↓
         Classification Head → (4-class logits)
```

**Freezing Strategy**:
- Frozen by default: BERT, CLIP, LDA, PCA layers
- Trainable: GAT1, GAT2, Classification layers
- Option: `freeze_non_trainable=False` to unfreeze all

#### `MisogynyModelNoGraph` (Ablation)
- Same architecture minus the Graph Module
- Feature path: Text PCA + Image PCA + LDA directly → Classifier
- Used for ablation studies

#### `MisogynyModelPCAOnly` (Ablation)
- Simplest variant: PCA on text + PCA on image only
- No graph, no LDA
- Used for baseline comparison

---

### 9. **Training & Evaluation Infrastructure**
**Location**: [`main.py`](main.py), [`test.py`](test.py)

#### Training Configuration:
- **Optimizer**: Adam with lr=1e-4
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 16
- **Epochs**: Variable per model (graph=25, no_graph=20, pca_only=10)
- **Metrics Tracked**: 
  - Loss (train/val)
  - Accuracy (macro)
  - Precision (macro)
  - Recall (macro)
  - F1-Score (macro and weighted)

#### Checkpoint Saving:
- **Best Checkpoint**: Highest validation F1-score
- **Final Checkpoint**: Last epoch
- **Format**: PyTorch state_dict with epoch and metrics metadata
- **Storage**: `saved_models/` directory

#### Evaluation Metrics:
```python
def compute_metrics(y_true, y_pred):
    # accuracy, precision (macro), recall (macro), f1_macro, f1_weighted
```

---

## Training Results & Performance

### Best Model Performance
**File**: `saved_models/graph_20260217_1150_BEST_ep22_acc0.955_f10.925.pth`

| Model Variant | Metric | Value | Epoch |
|---|---|---|---|
| **With Graph** | Val Accuracy | **95.54%** | 22 |
| | Val F1 (Weighted) | **95.43%** | 22 |
| | Val F1 (Macro) | **92.47%** | 22 |
| **No Graph** | Val Accuracy | 90.6% | 18 |
| | Val F1 (Weighted) | 81.4% | 18 |
| **PCA Only** | Val Accuracy | 95.1% | 10 |
| | Val F1 (Weighted) | 91.0% | 10 |

### Full Training Logs
- **Graph Model**: `logs/logs_graph.csv` (25 epochs)
  - Shows progression from ~44% → 95.5% validation accuracy
  - Training accuracy plateaus around epoch 15-20
  - Stable convergence with GAT component

- **No Graph Model**: `logs/logs_no_graph.csv` (20 epochs)
- **PCA Only Model**: `logs/logs_pca_only.csv` (10 epochs)

### Key Observations
1. **Graph variant performs best** - suggesting relational context matters
2. **Early improvement** - rapid accuracy gains epochs 1-10
3. **Validation F1 higher than training** - possible data leakage or augmentation imbalance (investigation needed)
4. **Stable plateau** - model converges around epoch 15-22 for graph variant

---

## Pre-trained Weights & Checkpoints

### Dimensionality Reduction Weights
Location: `weights/`

| File | Size | Purpose |
|---|---|---|
| `bert_pca_components.npy` | (400, 768) | BERT PCA transformation matrix |
| `bert_pca_mean.npy` | (768,) | BERT feature mean (for centering) |
| `clip_pca_components.npy` | (400, 512) | CLIP PCA transformation matrix |
| `clip_pca_mean.npy` | (512,) | CLIP feature mean |
| `combined_lda_mean.npy` | (1280,) | Combined embedding mean for LDA |
| `combined_lda_coef.npy` | (LDA_dim, 1280) | LDA projection coefficients |

### Pre-trained Graph Module
- `weights/graph_module.pth` - Serialized GAT with pre-fit training node features

### Model Checkpoints
Location: `saved_models/`

```
graph_20260217_1150_BEST_ep22_acc0.955_f10.925.pth    # ← Best overall
graph_20260217_1150_FINAL_ep25.pth                     # Final epoch
no_graph_20260217_1208_BEST_ep18_acc0.906_f10.814.pth  # Ablation
no_graph_20260217_1208_FINAL_ep20.pth
pca_only_20260217_1217_BEST_ep10_acc0.951_f10.910.pth  # Baseline
pca_only_20260217_1217_FINAL_ep10.pth
```

---

## Experimentation & Development

### Jupyter Notebooks
Location: `trial_notebooks/`

1. **`gat_trial.ipynb`**
   - GAT module development and testing
   - KNN graph construction verification
   - Embedding pipeline validation
   - Graph attention visualization experiments

2. **`trying-out-bert-clipvit.ipynb`**
   - BERT embedder implementation & testing
   - CLIP ViT embedder integration
   - Embedding dimension verification
   - Normalization verification

3. **`autoencoder-multimodal.ipynb`**
   - Multimodal architecture exploration
   - Feature extraction and fusion pipeline
   - Dimensionality reduction approaches

4. **`pca_trial.ipynb`**
   - PCA component selection
   - Variance explained analysis
   
5. **`lda_trial.ipynb`**
   - LDA fitting and coefficient extraction
   - Class separability analysis

### Trial/Experimental Code
**Location**: [`trial.py`](trial.py)

**Components Tested**:
- CSV generation from image directory structure
- Dataset visualization and sampling
- Graph module inference testing
- Full model pipeline end-to-end testing
- Embedding dimension verification
- KNN graph construction validation

**Status**: Contains mostly commented-out experimental snippets

---

## Dependencies

**Core ML/DL**:
- `torch` - Deep learning framework
- `torch-geometric` - GATConv for graph neural networks
- `transformers` - BERT model and tokenizer
- `open_clip_torch` - CLIP ViT models

**Data & Processing**:
- `pandas` - CSV handling, data manipulation
- `scikit-learn` - PCA, LDA, KNN, train/test split
- `PIL` - Image loading and preprocessing
- `numpy` - Numerical operations

**Utilities**:
- `tqdm` - Progress bars

**Full requirements**: See [`requirements.txt`](requirements.txt)

---

## Known Issues & Next Steps

### Current Investigation: Underfitting Patterns
As documented in [`README.md`](README.md):
- Initially designed with target of ~50% accuracy on underfitting task
- Graph attention not initially performing as expected
- Plan: Redesign architecture for next week

### Potential Improvements
1. **Architecture Refinement**:
   - Investigate why validation F1 > training F1
   - Larger hidden dimensions in classification head
   - Additional GAT layers or alternative graph construction
   - Different fusion strategies (concatenation vs. attention-based)

2. **Graph Module Enhancements**:
   - Experiment with different k values (currently k=20)
   - Try alternative similarity metrics (Euclidean, dot product)
   - Dynamic graph adaptation per batch

3. **Training Optimization**:
   - Learning rate scheduling
   - Warmup strategy
   - Data augmentation for images/text
   - Class-weighted loss for imbalanced categories

4. **Evaluation Pipeline**:
   - Per-class confusion matrices
   - Detailed error analysis
   - Visualization of learned graph structures
   - Test-time augmentation (TTA)

---

## Project File Structure

```
graph-networks/
├── PROGRESS.md                    # This file
├── README.md                      # Project overview
├── requirements.txt               # Dependencies
├── data_csv.csv                   # Dataset manifest
├── data_loader.py                 # PyTorch Dataset & DataLoader
├── main.py                        # Training script (unified trainer)
├── test.py                        # Evaluation script (mostly commented)
├── trial.py                       # Experimental code snippets
│
├── modules/                       # Core model modules
│   ├── __init__.py
│   ├── embedders/                 # Text & image encoders
│   ├── layers/                    # Transformation layers
│   └── models/                    # End-to-end models
│
├── data/                          # Image data
│   ├── kitchen/
│   ├── shopping/
│   ├── working/
│   ├── leadership/
│   ├── *.tsv                      # Train/val/test splits
│   └── MAMI_2022_images/          # Additional dataset
│
├── weights/                       # Pre-computed transforms
│   ├── bert_pca_*
│   ├── clip_pca_*
│   ├── combined_lda_*
│   └── graph_module.pth
│
├── saved_models/                  # Trained model checkpoints
│   ├── graph_*.pth
│   ├── no_graph_*.pth
│   └── pca_only_*.pth
│
├── logs/                          # Training logs
│   ├── logs_graph.csv
│   ├── logs_no_graph.csv
│   ├── logs_pca_only.csv
│   └── train_log.txt
│
├── trial_notebooks/               # Jupyter experimentation
│   ├── gat_trial.ipynb
│   ├── trying-out-bert-clipvit.ipynb
│   ├── autoencoder-multimodal.ipynb
│   ├── pca_trial.ipynb
│   └── lda_trial.ipynb
│
└── images/                        # Model diagrams & visualizations
```

---

## Quick Usage Guide

### Loading a Pre-trained Model
```python
from modules import MisogynyModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MisogynyModel(device=device).to(device)

# Load best checkpoint
checkpoint = torch.load("saved_models/graph_20260217_1150_BEST_ep22_acc0.955_f10.925.pth", 
                        map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    logits = model(text_captions, image_tensors)
    predictions = torch.argmax(logits, dim=1)
```

### Training Custom Model
```python
from main import train_model, MODEL_CONFIGS

# Select model variant and train
train_model("graph", MODEL_CONFIGS["graph"]["class"], 
            epochs=MODEL_CONFIGS["graph"]["epochs"])
```

---

## Development Notes

- **Latest Checkpoint**: February 17, 2026 (~11:50, 12:08, 12:17 timeframes)
- **Training Device**: Multi-GPU capable (uses CUDA when available)
- **Virtual Environment**: Active at `venv/bin/activate`
- **Code Format**: Standard Python 3.x with type hints
- **Git Status**: Not documented (ensure to commit regularly)

---

**For use as context for future prompts and tasks. Refer back to specific sections for architectural details, performance metrics, or implementation references.**
