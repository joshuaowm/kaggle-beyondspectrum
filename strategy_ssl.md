# Self-Supervised Learning Architecture for Sentinel-2 Crop Disease Classification

NOTE: THIS WAS OBVIOUS AI GENERATED. Just giving you guys the overview of the implementation of the SSL, to make it easier writing the report later. 
The whole idea is using constrastive learning of the images inside the crops folder. Instead of training the encoder from scratch, I will use the CROMA as a backbone and train the decoder constrative learning to leverage the spectral features of the S2. After training it, we are fine-tunning with the disease folder. Hopelly will perform better. 

## Project Overview

**Goal**: Pretrain a CROMA encoder using self-supervised learning on unlabeled Sentinel-2 crop images, then fine-tune for disease classification.

**Current Baseline in our project**: 64% accuracy using CROMA (pretrained) + MLP decoder with focal loss

**Expected Improvement**: 5-15% accuracy gain through domain-specific self-supervised pretraining

---

## 1. Data Structure

### Unlabeled Data (for SSL Pretraining)
```
train/
├── maize/
│   ├── 000302/
│   │   ├── 20210102T113359_20210102T113446_T30VVH/
│   │   ├── 20210701T113319_20210701T113320_T30VVH/
│   │   └── ... (2-3 timesteps per location)
│   └── 000303/
├── rice/
│   ├── 000601/
│   └── 000602/
├── soybean/
│   ├── 000901/
│   └── 000902/
└── wheat/
    ├── 000000/
    └── 000001/
```

**Characteristics**:
- Multiple crop types (maize, rice, soybean, wheat)
- Multiple locations per crop
- 2-3 temporal observations per location
- Each observation contains 12 Sentinel-2 bands (B1-B12)
- Image size: 120×120 pixels

### Labeled Data (for Supervised Fine-tuning)
```
kaggle/
├── Aphid/
│   └── {sample_id}/
│       ├── B1.tif
│       ├── B2.tif
│       └── ... (12 bands)
├── Blast/
├── RPH/
├── Rust/
└── evaluation/
```

**Characteristics**:
- 4 disease classes
- Each sample has 12 Sentinel-2 bands
- Image size: 120×120 pixels

---

## 2. Self-Supervised Learning Strategy

### 2.1 Pretraining Objective: **Spectral-Temporal Multi-View Contrastive Learning**

We combine two complementary approaches:
1. **Spectral Multi-View**: Different band combinations of the same image
2. **Temporal Consistency**: Different timesteps of the same location

### 2.2 Positive and Negative Pairs

#### Positive Pairs (should have similar representations)
1. **Spectral views** of the same image with spatial augmentations
   - Example: RGB-NIR view vs SWIR view of same location
2. **Temporal pairs** from the same location at different times
   - Example: `000302/timestep1` vs `000302/timestep2`
3. **Augmented versions** of the same view
   - Example: Original crop vs rotated/flipped version

#### Negative Pairs (should have different representations)
1. Different locations within same crop type
2. Different crop types
3. Different samples from other batches

### 2.3 Spectral Band Grouping Strategy

We define multiple complementary views based on spectral properties:

| View Name | Bands | Spectral Range | Purpose |
|-----------|-------|----------------|---------|
| **Visible-NIR** | B2, B3, B4, B8 | 490-842 nm | General vegetation structure |
| **Red Edge-SWIR** | B5, B6, B7, B8A, B11, B12 | 705-2190 nm | Vegetation stress, moisture |
| **Random Subset** | Random 6-8 bands | Mixed | Encourage diverse representations |
| **Vegetation Indices** | B3, B4, B5, B8, B11 | Mixed | NDVI, EVI-relevant bands |

**Implementation**:
- Each training sample generates 2 random spectral views
- Views are mutually exclusive (no band overlap when possible)
- Additional spatial augmentations applied to each view

---

## 3. Architecture Design

### 3.1 Overall Pipeline

```
                    Unlabeled Sentinel-2 Image (12 bands, 120×120)
                                    ↓
        ┌──────────────────────────┴──────────────────────────┐
        ↓                                                      ↓
   View 1 (e.g., B2,B3,B4,B8)                    View 2 (e.g., B5,B6,B7,B11,B12)
   + Spatial Aug (crop, flip)                     + Spatial Aug (crop, flip)
        ↓                                                      ↓
   CROMA Encoder (FROZEN)                         CROMA Encoder (FROZEN)
   Output: 768-dim embedding                      Output: 768-dim embedding
        ↓                                                      ↓
   Projection Head                                Projection Head
   768 → 256 → 128                                768 → 256 → 128
        ↓                                                      ↓
   L2 Normalize                                   L2 Normalize
        ↓                                                      ↓
   z1 (128-dim)                                   z2 (128-dim)
        └──────────────────────────┬──────────────────────────┘
                                   ↓
                          Contrastive Loss
                          (InfoNCE / NT-Xent)
```

### 3.2 Component Details

#### 3.2.1 CROMA Encoder (Frozen)
- **Architecture**: Vision Transformer (ViT)
- **Parameters**:
  - `encoder_dim`: 768
  - `encoder_depth`: 12 transformer blocks
  - `num_heads`: 16
  - `patch_size`: 8×8
  - `image_size`: 120×120
- **Weights**: Pretrained `CROMABase_Weights.CROMA_VIT`
- **Training Mode**: **FROZEN** (all parameters frozen)
  - Rationale: Faster training, lower memory, leverages existing pretrained knowledge

#### 3.2.2 Projection Head (Trainable)
```python
ProjectionHead:
  - Linear(768 → 256)
  - BatchNorm1d(256)
  - ReLU
  - Linear(256 → 128)
  - L2 Normalization (output unit sphere)
```

**Design Rationale**:
- Maps encoder features to lower-dimensional contrastive space
- BatchNorm for training stability
- L2 norm ensures embeddings lie on unit hypersphere (required for cosine similarity)

#### 3.2.3 Contrastive Loss: InfoNCE (NT-Xent)

**Formula**:
```
For a batch of N samples, we create 2N views (2 per sample)
For view i, positive pair is j (other view of same sample)

ℓ(i,j) = -log[ exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) ]

Total Loss = 1/(2N) Σ_i [ℓ(2i, 2i+1) + ℓ(2i+1, 2i)]
```

Where:
- `sim(u,v)` = cosine similarity = u·v / (||u|| ||v||)
- `τ` = temperature parameter (default: 0.07)
- Sum over k excludes i (no self-similarity)

**Why InfoNCE?**
- Standard in contrastive learning (SimCLR, MoCo)
- Maximizes agreement between positive pairs
- Minimizes agreement with negatives
- Well-suited for representation learning

---

## 4. Data Augmentation Strategy

### 4.1 Spectral View Selection
- **Method**: Random selection from predefined spectral groups
- **Constraint**: Minimize band overlap between view1 and view2
- **Frequency**: Applied to every sample

### 4.2 Spatial Augmentations
Applied to each spectral view independently:

| Augmentation | Parameters | Probability |
|--------------|-----------|-------------|
| Random Crop | 80-100% of original | 0.8 |
| Random Horizontal Flip | - | 0.5 |
| Random Vertical Flip | - | 0.5 |
| Random Rotation | ±15° | 0.3 |
| Gaussian Blur | kernel=3, σ=0.1-2.0 | 0.2 |

**Normalization**: 
- Per-band normalization using Sentinel-2 statistics
- Mean/Std computed from training set

### 4.3 Temporal Pairing (Additional Positives)
- **When available**: Use images from different timesteps of same location
- **Implementation**: 
  - Sample 2 different timesteps → create 2 spectral views from each → 4 total views
  - Temporal pairs treated as additional positive pairs in loss
- **Fallback**: If only 1 timestep available, use only spectral views

---

## 5. Training Configuration

### 5.1 Phase 1: Self-Supervised Pretraining

**Dataset**: Unlabeled crop data (train folder)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Batch Size | 64 | Balance between GPU memory and convergence |
| Epochs | 100 | Sufficient for convergence on contrastive task |
| Learning Rate | 3e-4 | Standard for projection head training |
| Optimizer | AdamW | Better generalization than Adam |
| Weight Decay | 1e-4 | Regularization |
| LR Scheduler | Cosine Annealing | Smooth decay to final LR |
| Temperature (τ) | 0.07 | Standard for contrastive learning |
| Warmup Epochs | 10 | Stabilize early training |

**Sampling Strategy**: 
- **Crop-proportional sampling**: Sample based on number of images per crop
- Ensures model sees all crop types but respects data distribution

**Training Time Estimate**: ~3-4 hours on Kaggle GPU (T4/P100)

### 5.2 Phase 2: Supervised Fine-tuning

**Dataset**: Labeled disease data (kaggle folder)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Batch Size | 32 | Smaller dataset, allow larger effective batch through accumulation |
| Epochs | 50 | Sufficient for convergence |
| Learning Rate | 1e-3 | Higher LR for classification head |
| Encoder LR | 0 (frozen) OR 1e-5 (unfrozen) | Start frozen, optionally unfreeze |
| Optimizer | AdamW | Consistency with pretraining |
| Weight Decay | 1e-4 | Regularization |
| Loss | Focal Loss (γ=1.0) | Handle class imbalance |
| Class Weights | Computed from training set | Further handle imbalance |

**Transfer Strategy**:
1. Load pretrained CROMA encoder (with SSL-learned features)
2. **Discard** projection head
3. **Initialize** new classification head (768 → 512 → 4)
4. **Option A** (default): Freeze encoder, train only classification head
5. **Option B** (if needed): Unfreeze encoder with low LR after 20 epochs

---

## 6. Validation and Monitoring

### 6.1 Self-Supervised Pretraining Validation

Since we don't have labels during pretraining, we use:

#### Primary Metric: Contrastive Loss Convergence
- Monitor training loss every epoch
- Expect steady decrease over 100 epochs
- Target: Loss < 1.5 indicates good convergence

#### Secondary Metric: Linear Probe Evaluation
- **Frequency**: Every 10 epochs
- **Method**: 
  1. Freeze CROMA encoder + projection head
  2. Train simple linear classifier (768 → 4) on labeled disease data
  3. Evaluate on validation set
- **Metric**: Validation accuracy
- **Purpose**: Check if learned representations are useful for downstream task

**Expected Linear Probe Performance**:
- Epoch 10: ~40-50% accuracy (random = 25%)
- Epoch 50: ~55-65% accuracy
- Epoch 100: ~60-70% accuracy

### 6.2 Fine-tuning Validation

Standard supervised metrics:
- **Validation Loss**: Focal loss
- **Validation Accuracy**: Balanced (macro-averaged)
- **Validation F1**: Weighted by class frequency
- **Per-class F1**: Monitor performance on each disease

**Target Performance**:
- Baseline (no SSL): 64% accuracy
- With SSL pretraining: **70-75% accuracy** (target improvement: 6-11%)

---

## 7. Implementation Architecture (PyTorch Lightning)

### 7.1 Module Structure

#### `SSLPretrainer` (LightningModule)
```
Responsibilities:
- Load CROMA encoder (frozen)
- Initialize projection head
- Implement forward pass for contrastive learning
- Compute InfoNCE loss
- Handle training/validation steps
- Log metrics (loss, temperature, similarity distributions)
- Periodic linear probe evaluation
```

#### `SentinelDiseaseClassifier` (Existing, Modified)
```
Responsibilities:
- Load pretrained CROMA encoder (from SSL checkpoint)
- Initialize classification head
- Implement forward pass for disease classification
- Compute focal loss
- Handle training/validation/test steps
- Log metrics (accuracy, F1, per-class metrics)
```

### 7.2 Dataset Classes

#### `SentinelSSLDataset` (New)
```
Responsibilities:
- Load unlabeled crop images
- Create spectral views (band selection)
- Apply spatial augmentations
- Handle temporal pairs when available
- Return: (view1, view2, crop_type, location_id)
```

#### `SentinelDiseaseDataset` (Existing)
```
Responsibilities:
- Load labeled disease images
- Load all 12 bands
- Apply normalization
- Return: (image, label, sample_id)
```

---

## 8. Expected Outcomes

### 8.1 Qualitative Benefits

After SSL pretraining, we expect the encoder to learn:
1. **Spectral relationships**: Which bands co-occur in healthy vs stressed vegetation
2. **Temporal patterns**: Seasonal changes, crop growth cycles
3. **Crop-specific features**: Distinctive spectral signatures of maize vs wheat vs rice
4. **Domain adaptation**: Agricultural-specific features beyond general ImageNet patterns

### 8.2 Quantitative Targets

| Metric | Baseline (No SSL) | Target (With SSL) | Improvement |
|--------|-------------------|-------------------|-------------|
| Validation Accuracy | 64% | 70-75% | +6-11% |
| Validation F1 (weighted) | ~0.62 | 0.68-0.73 | +0.06-0.11 |
| Linear Probe (frozen) | ~45% | 60-70% | +15-25% |

### 8.3 Ablation Studies (Future)

To understand contribution of each component:
1. **Spectral views only** (no temporal)
2. **Temporal pairs only** (no spectral views)
3. **Combined** (spectral + temporal)
4. **Different band groupings**
5. **Different augmentation strengths**

---

## 9. Computational Requirements

### 9.1 Memory Footprint

**Pretraining (SSL)**:
- Batch size: 64
- Model: CROMA (frozen) + Projection head
- Expected GPU memory: ~8-10 GB
- Compatible with: Kaggle T4 (16GB), P100 (16GB)

**Fine-tuning**:
- Batch size: 32
- Model: CROMA (frozen) + Classification head
- Expected GPU memory: ~6-8 GB
- Compatible with: Kaggle T4, P100

### 9.2 Training Time Estimates

| Phase | Epochs | Time per Epoch | Total Time |
|-------|--------|----------------|------------|
| SSL Pretraining | 100 | ~2-3 min | ~3-5 hours |
| Fine-tuning | 50 | ~1-2 min | ~1-2 hours |
| **Total** | - | - | **4-7 hours** |

**Note**: Using Kaggle's GPU quota (30-40 hours/week), this is very feasible.

---

## 10. Potential Challenges and Mitigations

### Challenge 1: Limited Temporal Pairs
- **Issue**: Not all locations have multiple timesteps
- **Mitigation**: Spectral views are primary objective; temporal is bonus

### Challenge 2: Kaggle GPU Timeouts
- **Issue**: Kaggle sessions timeout after 9-12 hours
- **Mitigation**: 
  - Checkpoint every 10 epochs
  - Can complete pretraining in single session (~5 hours)
  - Resume from checkpoint if needed

### Challenge 3: Optimal Temperature (τ)
- **Issue**: Temperature hyperparameter affects contrastive learning
- **Mitigation**: Start with 0.07 (standard), try [0.05, 0.1] if poor results

### Challenge 4: Projection Head Overfitting
- **Issue**: Small trainable component might overfit
- **Mitigation**: 
  - Weight decay (1e-4)
  - Dropout in projection head (0.1)
  - Monitor linear probe performance

### Challenge 5: Class Imbalance in Fine-tuning
- **Issue**: Disease classes may be imbalanced
- **Mitigation**: Already handled with focal loss + class weights

---

## 11. Success Criteria

### Minimum Viable Success
- ✅ SSL pretraining converges (loss decreases smoothly)
- ✅ Linear probe shows >55% accuracy by epoch 100
- ✅ Fine-tuned model achieves >67% accuracy (>3% improvement over baseline)

### Target Success
- ✅ Linear probe achieves 60-70% accuracy
- ✅ Fine-tuned model achieves 70-75% accuracy (6-11% improvement)
- ✅ Per-class F1 scores improve across all disease types

### Exceptional Success
- ✅ Fine-tuned model achieves >75% accuracy (>11% improvement)
- ✅ Learned representations transfer to other downstream tasks
- ✅ Ablation studies show clear benefit of both spectral and temporal views

---

## 12. Next Steps

### Implementation Order
1. ✅ **Define architecture** (this document)
2. ⏭️ **Modify dataset** to create spectral views + augmentations
3. ⏭️ **Implement projection head** module
4. ⏭️ **Implement SSL pretrainer** (PyTorch Lightning module)
5. ⏭️ **Add InfoNCE loss** implementation
6. ⏭️ **Add linear probe** evaluation callback
7. ⏭️ **Train SSL model** on unlabeled data
8. ⏭️ **Modify supervised classifier** to load SSL weights
9. ⏭️ **Fine-tune** on disease classification
10. ⏭️ **Evaluate and compare** with baseline

### Timeline Estimate
- Implementation: 1-2 days
- SSL pretraining: 3-5 hours
- Fine-tuning experiments: 2-3 hours per run
- **Total**: ~3-4 days to first results

---

## 13. References and Inspiration

### Key Papers
1. **SimCLR**: "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., 2020)
2. **MoCo**: "Momentum Contrast for Unsupervised Visual Representation Learning" (He et al., 2020)
3. **CROMA**: Foundation model for multi-spectral satellite imagery
4. **SeCo**: "Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data" (Manas et al., 2021)

### Design Decisions Influenced By
- **Frozen encoder**: CLIP fine-tuning practices
- **Spectral views**: Multi-modal contrastive learning (CLIP, ALIGN)
- **Temporal consistency**: SeCo, satellite imagery SSL literature
- **InfoNCE loss**: SimCLR, MoCo standard

---

## Appendix A: Sentinel-2 Band Reference

| Band | Name | Wavelength (nm) | Resolution (m) | Purpose |
|------|------|-----------------|----------------|---------|
| B1 | Coastal Aerosol | 443 | 60 | Atmospheric correction |
| B2 | Blue | 490 | 10 | Visual, water |
| B3 | Green | 560 | 10 | Visual, vegetation |
| B4 | Red | 665 | 10 | Visual, vegetation |
| B5 | Red Edge 1 | 705 | 20 | Vegetation stress |
| B6 | Red Edge 2 | 740 | 20 | Vegetation stress |
| B7 | Red Edge 3 | 783 | 20 | Vegetation stress |
| B8 | NIR | 842 | 10 | Vegetation, biomass |
| B8A | Narrow NIR | 865 | 20 | Vegetation |
| B9 | Water Vapour | 945 | 60 | Atmospheric |
| B11 | SWIR 1 | 1610 | 20 | Moisture, soil |
| B12 | SWIR 2 | 2190 | 20 | Moisture, geology |

---

## Appendix B: Hyperparameter Summary

### SSL Pretraining
```python
ssl_config = {
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'temperature': 0.07,
    'warmup_epochs': 10,
    'projection_dim': 128,
    'hidden_dim': 256,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
}
```

### Fine-tuning
```python
finetune_config = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'focal_gamma': 1.0,
    'freeze_encoder': True,  # Can unfreeze after epoch 20
    'hidden_dim': 512,
    'num_classes': 4,
}
```

---

**Document Version**: 1.0  
**Date**: February 13, 2026  
**Author**: SSL Architecture Planning Session  
**Status**: Ready for Implementation