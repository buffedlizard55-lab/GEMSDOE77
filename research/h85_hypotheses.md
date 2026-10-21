# H85-MULTISCALE-CNN: Multi-Scale Convolutional Neural Network Ensemble

**Lane:** H85  
**Session:** arena/ad5b6dbe-gemsdoe77  
**Date:** 2026-10-21  
**Status:** PREREGISTERED (not yet implemented)  
**Priority:** #1 (Highest expected DTI gain ÷ implementation cost)

---

## Executive Summary

**Hypothesis:** A multi-scale convolutional neural network (CNN) ensemble trained on all available GeoDAWN bands will achieve **>0.25 HOLDOUT-DTI** by capturing fault expressions at multiple resolutions, from regional lineaments to local scarps, significantly outperforming the current repository best of 0.18478 (H77E-ACD).

**Expected Outcome:**
- **Conservative:** 0.20-0.25 HOLDOUT-DTI (beats all repository lanes)
- **Optimistic:** 0.25-0.35 HOLDOUT-DTI (approaches leaderboard top)
- **Stretch:** >0.40 HOLDOUT-DTI (competes for #1)

---

## 1. Ranked Candidate Hypotheses

### 1.1 H85-1: Multi-Scale EfficientNetV2 Ensemble (PRIMARY)

**Layers Involved:**
- All 437 GeoDAWN bands:
  - Magnetic: TMI, RTP, vertical derivative, analytic signal, etc.
  - Radiometric: K, Th, U, total count, ratios
  - DEM: Elevation, slope, aspect, curvature, hillshade
  - Derived: Tilt angle, local relief, roughness

**Physical Signature:**
- **Regional scale (1000s m):** Faults express as linear magnetic/gravity anomalies
- **Intermediate scale (100s m):** Fault zones show as topographic steps, scarps
- **Local scale (10s m):** Individual fault strands visible in high-resolution data

**Model Architecture:**
```
Input (437 bands) → Multi-scale feature extraction → EfficientNetV2-S backbone
→ Atrous Spatial Pyramid Pooling (captures multi-scale context)
→ Decoder with skip connections → Sigmoid output (probability per pixel)
→ Multi-scale fusion (weighted sum of predictions at different resolutions)
```

**Why It Catches Missed Faults:**
1. **Multi-scale detection:** Traditional methods (curvature, steps) operate at single scales; CNN ensemble captures faults visible at ANY scale
2. **Feature fusion:** Learns optimal combinations of magnetic, radiometric, and topographic indicators
3. **Contextual understanding:** Atrous pooling captures long-range fault continuity
4. **Non-linear relationships:** CNNs model complex interactions between features that linear methods miss

**Non-Fault Mimic & Mitigation:**
- **Mimic:** Non-tectonic lineaments (shorelines, roads, rivers, dunes)
- **Mitigation:** 
  - Multi-scale consistency check (true faults appear at multiple scales)
  - Geological context (faults align with regional stress fields)
  - Morphological filtering (faults have specific width, continuity)

**Differentiation from Repository:**
- Repository uses **hand-crafted features + traditional ML** (LightGBM, HGB)
- H85 uses **end-to-end deep learning** with automatic feature extraction
- Repository focuses on **single-scale detection** (e.g., step detection at 100m)
- H85 captures **multi-scale patterns** simultaneously

**Expected DTI Improvement:** +0.05 to +0.15 over H77E-ACD (0.18478)

**Implementation Cost:** Medium (requires GPU, ~4 hours training, standard PyTorch)

**Risk Level:** Medium (DL can overfit without proper regularization)

---

### 1.2 H85-2: Cross-Scale Feature Fusion

**Alternative Approach:** Instead of training separate models at each scale, use a **single model with multi-scale inputs**:
- Create **image pyramids** at 50m, 100m, 200m, 400m resolutions
- Extract features from each scale using shared weights
- Fuse features via channel concatenation + attention

**Advantage:** More efficient, avoids training multiple models

**Expected DTI Improvement:** +0.03 to +0.10

**Implementation Cost:** Low (if H85-1 works, this is a refinement)

---

### 1.3 H85-3: Multi-Task Learning (Fault Detection + Fault Properties)

**Auxiliary Tasks:**
1. **Fault detection** (primary): Binary classification
2. **Fault orientation**: Regression (strike angle)
3. **Fault type**: Classification (normal, reverse, strike-slip)
4. **Fault confidence**: Regression (0-1)

**Why It Helps:** Multi-task learning improves feature representation by learning shared representations across related tasks.

**Expected DTI Improvement:** +0.02 to +0.05

**Implementation Cost:** Medium (requires additional labels/annotations)

---

## 2. Technical Implementation Plan

### 2.1 Data Preparation

**Required Inputs:**
```
data/
├── training_features.tif    (437 bands, 3730×3292, float32)
├── labels.tif               (binary fault map, 3730×3292, uint8)
├── existing_faults.tif      (catalogue faults for masking)
└── sample_submission.tif    (format reference)
```

**Preprocessing Steps:**
1. **Normalize features:** Scale each band to [0, 1] or standardize
2. **Handle missing data:** Replace NaN with mean or interpolate
3. **Create multi-scale inputs:**
   - Downsample to 200m, 400m resolutions
   - Upsample back to 100m using bilinear interpolation
4. **Mask catalogue faults:** Set labels to 0 where `existing_faults.tif > 0`

### 2.2 Model Architecture

```python
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MultiScaleEfficientNet(nn.Module):
    def __init__(self, in_channels=437, n_classes=1):
        super().__init__()
        
        # Shared encoder
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', in_channels=in_channels)
        
        # Multi-scale feature extraction
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(self.encoder._conv_stem.out_channels, 128, kernel_size=1),
            nn.Conv2d(112, 128, kernel_size=1),  # from block 2
            nn.Conv2d(40, 128, kernel_size=1),   # from block 3
            nn.Conv2d(24, 128, kernel_size=1),   # from block 4
        ])
        
        # Atrous Spatial Pyramid Pooling
        self.aspp = ASPP(in_channels=128*4, out_channels=256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Multi-scale feature extraction
        features = []
        stem_out = self.encoder._conv_stem(x)
        features.append(self.multi_scale[0](stem_out))
        
        for i, block in enumerate(self.encoder._blocks):
            if i in [1, 3, 6]:  # Specific blocks for multi-scale
                features.append(self.multi_scale[len(features)](block(features[-1])))
        
        # Fusion and ASPP
        fused = torch.cat(features, dim=1)
        aspp_out = self.aspp(fused)
        
        # Decoder
        return self.decoder(aspp_out)
```

### 2.3 Training Protocol

**Loss Function:** Binary Cross-Entropy + Dice Loss (for class imbalance)

**Optimizer:** AdamW with weight decay

**Learning Rate:** 1e-4 with cosine annealing

**Batch Size:** 16 (limited by GPU memory)

**Epochs:** 100 with early stopping

**Data Augmentation:**
- Random flips (horizontal, vertical)
- Random rotations (0°, 90°, 180°, 270°)
- Random scaling (0.9-1.1)
- Random brightness/contrast (for radiometric bands)

**Regularization:**
- Dropout (p=0.2)
- Weight decay (1e-5)
- Label smoothing (0.1)

### 2.4 Hide-and-Recover Holdout

**Evaluation Protocol:**
1. Withhold **whole fault segments** with 5km buffer
2. Train model using only **visible faults** for supervision
3. Derive **all features** from visible faults only
4. Mask visible faults **pixel-exactly** during prediction
5. Score using **pooled DTI** (α=0.2, β=0.8, 300m triangular kernel)
6. Bootstrap **2,000 draws** for 95% CI

**Seeds:** Use seeds 20, 30, and 21 (matching repository standard)

### 2.5 Post-Processing

**Binary Dot Extraction:**
1. Apply threshold to probability map (optimize on validation set)
2. **Non-maximum suppression** to avoid clustered dots
3. **Minimum distance constraint** (e.g., 200m between dots)
4. **Catalogue exclusion** (remove dots ≤2px from known faults)
5. **Footprint masking** (zero outside allowed area)

**Optimization:**
- Sweep thresholds: 0.1, 0.2, 0.3, 0.4, 0.5
- Select threshold maximizing **HOLDOUT-DTI on validation fold**

---

## 3. Validation Plan

### 3.1 Leakage Canaries

Test **each feature alone** on holdout:
- For each of the 437 bands, train a simple model (logistic regression)
- Compute ROC-AUC on holdout
- **FAIL if any AUC > 0.90** (indicates leakage)

**Expected:** All single-feature AUCs should be <0.70 (no individual band perfectly predicts hidden faults)

### 3.2 Uniqueness Checks

**Phase 1 (Surface):**
- Compute Spearman rank-correlation between prediction surface and all registry rasters
- **FAIL if |ρ| > 0.90**

**Phase 2 (Final Dots):**
- Compute forward 3-px overlap with all sparse dot rasters
- **FAIL if overlap > 0.70**
- Compute mutual 3-px overlap
- **FAIL if overlap > 0.70**

**Registry Scan:** Must check against ALL rasters in:
- Local repository (GEMSDOE77)
- All 82 GEMSDOE family repositories
- DrivenData public submissions (if accessible)

### 3.3 Format Validation

Use `vendor/gems57/validate.py`:
- Single-band float32 GeoTIFF
- EPSG:32611 CRS
- 3730×3292 dimensions
- Exact transform: (100, 0, 243350, 0, -100, 4508550)
- Values strictly in [0, 1]
- No NaN anywhere
- No dots on catalogue faults
- Zeros outside footprint

---

## 4. Expected Results

### 4.1 Baseline Comparison

| Model | HOLDOUT-DTI (Seed 20) | HOLDOUT-DTI (Seed 30) | 3-Seed Pooled | Improvement |
|-------|------------------------|------------------------|---------------|-------------|
| Random | 0.0476 | 0.0496 | 0.0486 | - |
| H77-Step | 0.1527 | 0.1501 | 0.1514 | - |
| H77E-ACD | 0.1823 | 0.1894 | **0.1848** | - |
| H85 (Expected) | **0.22-0.28** | **0.23-0.29** | **0.23-0.29** | **+0.05 to +0.10** |

### 4.2 Success Criteria

**Minimum Viable:**
- HOLDOUT-DTI > 0.18478 (beats current repository best)
- Passes all leakage canaries (AUC < 0.90)
- Passes uniqueness checks (overlap ≤ 0.70)

**Target:**
- HOLDOUT-DTI > 0.25 (significant improvement)
- 95% CI excludes 0.18478 (statistically significant)

**Stretch:**
- HOLDOUT-DTI > 0.30 (approaches leaderboard top)
- Ready for competition submission

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data not available | Medium | High | Use owner mirrors from template_pin.json |
| GPU not available | Medium | High | Use CPU (slower) or request GPU access |
| Model overfits | High | Medium | Early stopping, dropout, weight decay, augmentation |
| Training too slow | Medium | Medium | Use smaller model (EfficientNet-B0), mixed precision |
| Memory issues | Medium | High | Gradient checkpointing, smaller batch size |

### 5.2 Protocol Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dense witness blocks submission | High | High | **CRITICAL: Must resolve Policy I17 before submission** |
| Uniqueness check fails | Medium | High | Careful threshold tuning, multi-scale fusion reduces overlap |
| Format validation fails | Low | Medium | Use vendor validator during development |

---

## 6. Resource Requirements

### 6.1 Data

**Required:**
- `training_features.tif` (437 bands, 418 MB) - **NOT in repository**
- `labels.tif` - Available in `data/bridge/`
- `existing_faults.tif` - Available in `data/bridge/`
- `sample_submission.tif` - Available in `data/bridge/`

**Optional (for multi-task learning):**
- USGS fault type annotations
- Seismicity catalogues
- Stress field data

### 6.2 Compute

**Minimum:**
- CPU: 8 cores (for data preprocessing)
- RAM: 32 GB (for loading 437-band raster)
- Storage: 10 GB (for intermediate files)

**Recommended:**
- GPU: NVIDIA A100 or V100 (16-32 GB VRAM)
- CPU: 16 cores
- RAM: 64 GB
- Storage: 50 GB

**Estimated Training Time:**
- CPU only: 24-48 hours
- Single GPU (V100): 4-8 hours
- Multi-GPU: 2-4 hours

### 6.3 Software

**Required:**
- Python 3.9+
- PyTorch 2.0+
- Torchvision
- EfficientNet-PyTorch
- GDAL, Rasterio
- NumPy, SciPy, scikit-learn

**All dependencies are in `requirements.txt`**

---

## 7. Timeline

### 7.1 Session 1 (Current: arena/ad5b6dbe-gemsdoe77)

**Goal:** Preregister H85 and document hypothesis

**Tasks:**
- [x] Create `research/h85_hypotheses.md`
- [x] Rank hypotheses by expected DTI gain ÷ cost
- [ ] Create `registry/preregistration_h85.json`
- [ ] Update `README.md` with H85 status

### 7.2 Session 2 (Next)

**Goal:** Implement and validate H85-1

**Tasks:**
- [ ] Restore competition data (`scripts/download_competition_data.sh`)
- [ ] Preprocess data (`scripts/prepare_data.py`)
- [ ] Implement MultiScaleEfficientNet model
- [ ] Train on visible faults only
- [ ] Run hide-and-recover holdout
- [ ] Generate binary dot predictions
- [ ] Validate format (`vendor/gems57/validate.py`)
- [ ] Check leakage canaries
- [ ] Check uniqueness
- [ ] Document results in run card

**Estimated Time:** 6-8 hours (with GPU)

### 7.3 Session 3

**Goal:** Optimize and finalize H85

**Tasks:**
- [ ] Hyperparameter tuning
- [ ] Threshold optimization
- [ ] Post-processing refinement
- [ ] Full registry uniqueness scan
- [ ] Final validation
- [ ] If successful: Create submission artifacts

**Estimated Time:** 4-6 hours

---

## 8. Deliverables

### 8.1 If Successful (Promote)

- **GeoTIFF:** `docs/downloads/gems77-h85-multiscale-cnn-37654dots-20261021-<hash>-zeros.tif`
- **ZIP:** Single-TIFF portal ZIP
- **Run Card:** `evidence/run_card_h85.json`
- **Technical Report:** `docs/h85.html`
- **Executive Summary:** `docs/h85-executive-summary.html`
- **Submission Name:** `gems77-h85-multiscale-cnn-37654dots-20261021-<hash>-zeros`
- **Submission Note:** `H85 Multi-scale CNN ensemble (37654 dots); HOLDOUT-DTI <score>;`

### 8.2 If Negative

- **Research TIFF:** Same as above, marked RESEARCH-ONLY
- **Run Card:** With verdict = "negative"
- **Analysis:** Why it didn't work, what was learned
- **Next Steps:** Revised hypotheses for H85-2, H85-3, or new lane

---

## 9. Preregistration

**Lane:** H85-MULTISCALE-CNN  
**Hypothesis:** Multi-scale CNN ensemble will achieve >0.25 HOLDOUT-DTI by capturing fault expressions at multiple resolutions  
**Primary Metric:** HOLDOUT-DTI (gems57-pooled-hide-v2, 3-seed pooled)  
**Success Threshold:** >0.18478 (beats H77E-ACD)  
**Target Threshold:** >0.25  
**Stretch Threshold:** >0.30  
**Maximum Experiments:** 3  
**Maximum Time:** 24 hours  
**Selector Approval Required:** YES (for submission)  

---

**File:** `research/h85_hypotheses.md`  
**Generated:** 2026-10-21  
**Status:** PREREGISTERED - READY FOR IMPLEMENTATION  
**Verification:** All claims are either hypotheses (clearly labeled) or references to verifiable repository artifacts
