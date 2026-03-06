# COMP0248 Coursework 1: RGB-D Hand Gesture Recognition

A multi-task deep learning system for hand gesture recognition, segmentation, and bounding box detection using RGB-D (RGB + Depth) input data.

## Project Overview

This project implements a multi-task learning framework that simultaneously performs:
- **Segmentation**: Hand vs background pixel-level segmentation
- **Classification**: 10-class hand gesture classification
- **Detection**: Hand bounding box prediction derived from segmentation masks

The model uses a shared UNet encoder with separate segmentation and classification decoders, processing concatenated RGB-D (4-channel) input images.

## Architecture

### Core Components

```
Input: RGB-D (4 channels)
├── Shared Encoder (UNet downsampling with skip connections)
│   └── 4 levels: 32→64→128→256 + 512 bottleneck channels
├── Segmentation Decoder (UNet upsampling)
│   └── Output: Hand segmentation mask (1×H×W)
├── Classification Head (Global Average Pooling + MLP)
│   └── Output: 10-class gesture logits
└── Bounding Box (derived from predicted mask)
    └── Output: (xmin, ymin, xmax, ymax)
```

### Key Features

- **4-Channel Input**: RGB concatenated with normalized depth maps
- **Multi-task Learning**: Shared feature extraction with task-specific heads
- **Loss Functions**:
  - Segmentation: BCEWithLogitsLoss + Dice Loss
  - Classification: CrossEntropyLoss
  - Combined: `L = α·L_seg + β·L_cls` (default α=1.0, β=0.5)

## Dataset Structure

```
COMP0248_set_dataset/
└── data_edded/full_data/
    └── [Subject]/[Gesture]/[Clip]/
        ├── rgb/              # RGB frames (*.png)
        ├── depth/            # Normalized depth (*.png)
        ├── depth_raw/        # Raw depth arrays (*.npy)
        └── annotation/       # Segmentation masks (*.png)
```

## Project Structure

```
COMP0248_CW1/
├── src/
│   ├── dataloader.py         # RGB-D data loading with augmentation
│   ├── model.py              # UNet multi-task architecture
│   ├── model_maskpool.py     # Improved model variant
│   ├── losses.py             # Multi-task loss functions
│   └── metrics.py            # Evaluation metrics (Dice, IoU, bbox IoU, accuracy)
│
├── train.py                  # Main training script
├── train_maskpool.py         # Training with maskpool variant
├── train_model_*.py          # Training script variants
│
├── evaluate.py               # Full evaluation on validation/test set
├── evaluate_best.py          # Evaluate best checkpoint
├── inference_single.py       # Single image inference
│
├── Data_Processing/          # Data preprocessing utilities
├── COMP0248_set_dataset/     # Full dataset
├── train_data/               # Preprocessed training data
├── weights/                  # Model checkpoints (best_val_dice.pt, last.pt)
├── results/                  # Training results and visualizations
│
└── COMP0248_Coursework_1.pdf # Original coursework specification
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- opencv-python (cv2)
- pandas
- matplotlib

### Setup

```bash
# Clone or navigate to project directory
cd COMP0248_CW1

# Install dependencies
pip install torch torchvision numpy opencv-python pandas matplotlib

# Verify dataset structure
ls COMP0248_set_dataset/
```

## Usage

### Training

#### Basic Training
```bash
python train.py
```

The script will:
- Load RGB-D keyframe dataset
- Train for 10 epochs (configurable)
- Compute metrics: Dice, IoU, Bbox IoU, Classification Accuracy
- Save best model to `weights/best_val_dice.pt`

#### Training Variants

```bash
# With MaskPool layer
python train_maskpool.py

# With CSV-based data loading
python train_maskpool_csv.py

# With custom learning rate groups
python train_model_1_lr_groups.py

# With input resizing
python train_model_resize_csv.py
```

### Evaluation

```bash
# Evaluate on validation set
python evaluate.py

# Evaluate best checkpoint
python evaluate_best.py
```

**Output Metrics:**
- Segmentation: Dice score, mean IoU
- Detection: Bounding box IoU, accuracy@0.5
- Classification: Top-1 accuracy, macro F1 score

### Inference

```bash
python inference_single.py --image path/to/rgb.png --depth path/to/depth.png
```

**Output:**
- Segmentation mask
- Predicted gesture class
- Bounding box coordinates

## Model Checkpoints

| File | Purpose |
|------|---------|
| `weights/best_val_dice.pt` | Best model checkpoint (highest validation Dice) |
| `weights/last.pt` | Last epoch checkpoint |

To load and use a checkpoint:

```python
import torch
from src.model import UNetMultiTask

model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32)
checkpoint = torch.load('weights/best_val_dice.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

## Training Configuration

Key hyperparameters in `train.py`:

```python
batch_size = 2              # Batch size
learning_rate = 1e-3        # Learning rate (Adam optimizer)
num_epochs = 10             # Number of training epochs
input_size = variable       # Keep original size or 256×256
use_depth = True            # Use depth channel
```

Loss weights in `src/losses.py`:

```python
w_seg = 1.0                 # Segmentation loss weight
w_dice = 1.0                # Dice loss weight
w_cls = 0.5                 # Classification loss weight
```

## Evaluation Metrics

### Segmentation
- **Dice Score**: 2·∑(p·g) / (∑p + ∑g + ε)
- **IoU**: TP / (TP + FP + FN)

### Detection
- **Bbox IoU**: Intersection-over-Union of predicted vs. ground truth boxes
- **Accuracy@0.5**: Percentage of detections with IoU ≥ 0.5

### Classification
- **Top-1 Accuracy**: Percentage of correctly classified gestures
- **Macro F1**: F1 score averaged across all 10 gesture classes
- **Confusion Matrix**: Per-class prediction breakdown

## Advanced Features

### Data Augmentation

The dataloader (`src/dataloader.py`) supports:
- Random horizontal flip
- Random rotation
- Random brightness/contrast adjustment
- Depth normalization (per-channel or z-score)

### Multi-Task Learning Strategy

1. **Shared Encoder**: Reduces computation and improves feature reuse
2. **Loss Weighting**: Balanced contribution of segmentation and classification tasks
3. **Mask-Derived BBox**: Ensures geometric consistency between segmentation and detection

## Results

Training curves and evaluation results are saved to:
- `results/` - Metrics visualizations and detailed results
- `curve_*.png` - Training/validation curves (Dice, IoU, classification, bbox loss)

## Common Issues & Solutions

### Dataset Path Errors
```
Error: CSV file not found
Solution: Update CSV_PATH in train.py to point to Data_Processing/meta/index_keyframes.csv
```

### GPU Memory Issues
```
Error: CUDA out of memory
Solution: Reduce batch_size (2→1) or model base_ch (32→16) in train.py
```

### Missing Depth Data
```
Error: Depth file not found
Solution: Set use_depth=False or ensure depth/*.png files exist alongside rgb/
```

## References

- Original Architecture Specification: `Network Architecture.txt`
- Coursework Details: `COMP0248_Coursework_1.pdf`

## Notes

- Bounding boxes are **automatically derived** from predicted segmentation masks, not trained directly
- Each training script variant in the project explores different architectural choices or training strategies
- Depth input is crucial for improving segmentation robustness in occluded/ambiguous regions
- RGB-D combination provides complementary geometric and appearance information

## Author

COMP0248 Coursework 1 - University College London
