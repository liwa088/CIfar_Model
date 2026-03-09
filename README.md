# CIFAR-10 Image Classification with Vision Transformer (ViT)

## Overview

This notebook demonstrates **transfer learning** using a pretrained **Vision Transformer (ViT-B/16)** model to classify images from the **CIFAR-10** dataset. The model is fine-tuned as a feature extractor, with only the classification head retrained on the new task.

---

## Dataset

**CIFAR-10** — a benchmark dataset for image classification.

- **Train set:** 50,000 images  
- **Test set:** 10,000 images  
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)  
- **Image size:** Resized to 224×224 (required by ViT)

---

## Model Architecture

### Base Model: ViT-B/16 (Vision Transformer)

| Component | Details |
|-----------|---------|
| Backbone | `torchvision.models.vit_b_16` (pretrained on ImageNet) |
| Patch projection | Conv2d: 3×224×224 → 768×14×14 |
| Encoder | 12 Transformer EncoderBlocks |
| Classification head | Linear(768 → 10) — **only trained layer** |
| Total parameters | 85,806,346 |
| **Trainable parameters** | **7,690** |

All backbone layers are **frozen**; only the new classification head is trained.

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss function | CrossEntropyLoss |
| Epochs | 5 |
| Batch size | 32 |
| Device | CUDA (T4 GPU) |

**Preprocessing transforms** (from ViT pretrained weights):
- Resize to 256, center crop to 224
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## Results

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.2138 | 93.46% | 0.1723 | 94.37% |
| 2 | 0.1407 | 95.36% | 0.1605 | 94.82% |
| 3 | 0.1245 | 95.88% | 0.1596 | 94.86% |
| 4 | 0.1154 | 96.10% | 0.1543 | 95.00% |
| 5 | 0.1078 | 96.40% | 0.1602 | 94.78% |

**Final test accuracy: ~94.78–95.00%** in just 5 epochs of fine-tuning.

---

## Saved Model

| Property | Value |
|----------|-------|
| File | `models/CIFAR_Model.pth` |
| Size | 327.38 MB |
| Parameters | 85,806,346 |

---

## Dependencies

```
torch >= 1.12
torchvision >= 0.13
torchinfo
going_modular  # cloned from mrdbourke/pytorch-deep-learning
```

Install:
```bash
pip install torch torchvision torchaudio torchinfo
git clone https://github.com/mrdbourke/pytorch-deep-learning
mv pytorch-deep-learning/going_modular/going_modular/ .
```

---

## Project Structure

```
├── CIFAR_Model.ipynb       # Main notebook
├── models/
│   └── CIFAR_Model.pth     # Saved model weights
└── going_modular/          # Helper modules (data_setup, engine, utils)
```

---

## Key Takeaways

- **Transfer learning is highly efficient:** Only 7,690 out of ~85.8M parameters were trained.
- **ViT generalizes well:** Pretrained ImageNet features transfer effectively to CIFAR-10, reaching ~95% accuracy in 5 epochs (~73 minutes on a T4 GPU).
- **Feature extraction strategy:** Freezing the backbone avoids overfitting and drastically reduces compute.
