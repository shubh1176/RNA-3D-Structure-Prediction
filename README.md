# RNA 3D Structure Prediction

A deep learning approach to predict 3D structures of RNA molecules from their sequences using transformer and graph attention networks.

## Project Overview

### Goal
This project aims to develop an accurate and efficient computational method for predicting the 3D structure of RNA molecules directly from their primary sequences. RNA structure prediction is a crucial challenge in molecular biology with applications in drug discovery, vaccine development, and understanding RNA-based diseases.

### Dataset
We use the Stanford RNA 3D Folding dataset from the [Stanford RNA 3D Folding Kaggle competition](https://www.kaggle.com/c/stanford-rna-3d-folding), which contains:

- RNA sequences in one-hot encoded format (A, U, G, C)
- 3D coordinates of each nucleotide in the RNA backbone
- A diverse set of RNA molecules with different sizes and structural complexities
- Training, validation, and test splits for model development and evaluation

## Model Architecture

Our model uses a two-stage approach combining transformers and graph neural networks:

### Stage 1: Initial Structure Prediction
- **Input Processing**: One-hot encoded RNA sequences
- **Positional Encoding**: Custom sinusoidal positional encoding layer
- **Sequence Encoding**: 6-layer transformer encoder with multi-head attention
- **Multi-scale Feature Extraction**: Parallel dilated convolutions to capture patterns at different scales
- **Coordinate Prediction**: Dense layers predicting x, y, z coordinates for each nucleotide

### Stage 2: Graph-based Structure Refinement
- **Graph Construction**: Creates a spatial graph based on the initial structure prediction
- **Node Features**: Combines sequence and structural information
- **Distance-based Attention Masking**: Limits attention to biologically relevant distances (15Å cutoff)
- **Graph Attention Network**: 3 layers of graph attention to refine the structure
- **Residual Refinement**: Small, controlled adjustments to the initial structure

```
Input RNA Sequence → Transformer Encoder → Initial 3D Coordinates → 
Graph Attention Refinement → Refined 3D Coordinates
```

### Loss Functions
The model uses multiple structure-aware loss components:

1. **Coordinate Loss**: MSE between predicted and true coordinates
2. **Distance Matrix Loss**: Preserves pairwise distances between nucleotides
3. **Orientation Loss**: Captures backbone geometry and torsion angles
4. **GDT-TS Loss**: Global Distance Test score for overall structural accuracy

## Training Approach

We employ a two-stage training strategy:

1. **Initial Prediction Training**:
   - Train only the transformer and initial coordinate prediction
   - Focus on getting the global topology correct

2. **Refinement Training**:
   - Freeze the initial prediction layers
   - Train only the graph attention refinement network
   - Focus on local structural details and geometric constraints

This approach allows the model to first learn overall RNA folding patterns before refining local structural details.

## Results

Our model achieves impressive performance on RNA structure prediction:

- **RMSD**: 1.37Å on validation set (6.03% improvement through refinement)
- **GDT-TS**: 88.75% (High accuracy in global topology)
- **Contact Map Overlap**: 0.9998 (Near-perfect topology preservation)

### Visualization

The model produces high-quality 3D structure predictions that capture the overall topology and local structural features of RNA molecules. Visualization tools show:

- True RNA structure
- Initial prediction
- Refined prediction
- Contact maps
- Local quality assessment
- Ramachandran-style backbone angle plots

## Future Directions

Several potential enhancements to improve model performance:

- Integration of RNA secondary structure information
- Incorporating evolutionary information through multiple sequence alignments
- Implementing SE(3)-equivariant operations for coordinate prediction
- Expanding the model to handle longer RNA sequences
- Adding physics-based energy terms for more realistic structure refinement

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- SciPy
- tqdm

## Usage

```python
# Load model
model = tf.keras.models.load_model('rna_structure_model.keras')

# Predict structure from sequence
predictions = model.predict(rna_sequence)
initial_coords = predictions['initial_coords']
refined_coords = predictions['refined_coords']

# Visualize prediction
visualize_rna_structure(model, rna_sequence, true_coordinates)
```

## Citation

If you use this model in your research, please cite:
```
@article{RNA-Structure-Prediction,
  title={RNA 3D Structure Prediction using Transformer and Graph Attention Networks},
  author={Your Name},
  year={2023}
}
``` 
