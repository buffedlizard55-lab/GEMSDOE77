# 🚀 QUICK START: H85-MULTISCALE-CNN

## TL;DR

**Goal:** Generate a UNIQUE TIF submission scoring >0.25 HOLDOUT-DTI (current repo best: 0.18478, leaderboard top: 0.4413)

**Lane:** H85-MULTISCALE-CNN (PREREGISTERED)

**Status:** ✅ Analysis complete, ready for implementation

---

## 🎯 THE PLAN

### What We're Building
A **multi-scale CNN ensemble** that:
1. Takes all 437 GeoDAWN bands as input
2. Extracts features at 100m, 200m, and 400m resolutions
3. Uses EfficientNetV2-S with atrous spatial pyramid pooling
4. Fuses multi-scale predictions
5. Generates binary fault dot predictions
6. Optimizes for HOLDOUT-DTI > 0.25

### Why It Will Work
- **Multi-scale:** Captures faults at regional (1000s m), intermediate (100s m), and local (10s m) scales
- **Deep learning:** Learns optimal feature combinations automatically
- **First in GEMSDOE:** No one has tried this yet in the repository
- **Proven approach:** CNNs work well for fault detection in geoscience

### Expected Results
- **Minimum:** > 0.18478 (beats repo best)
- **Target:** > 0.25 (significant improvement)
- **Stretch:** > 0.30 (approaches leaderboard top)
- **Leaderboard Impact:** Top 20-30 position

---

## 🚨 CRITICAL BLOCKERS

### Blocker #1: Policy I17 Dense Witness (CRITICAL)
```
❌ ISSUE: Universal raster ab0a0a62 covers 100% of allowed footprint
❌ IMPACT: ALL non-empty submissions fail uniqueness check
⚠️  RESOLUTION: Requires protocol owner decision
📍 EVIDENCE: evidence/registry_preflight.json
```

### Blocker #2: Missing Training Data (HIGH)
```
❌ ISSUE: training_features.tif (437 bands, 418 MB) NOT in repository
❌ IMPACT: Cannot train H85 or any new lane
✅ RESOLUTION: Run scripts/download_competition_data.sh
📍 LOCATION: data/bridge/ (after download)
```

---

## 📁 FILES CREATED IN THIS SESSION

### Documentation
- `docs/h85.html` - Technical report
- `docs/h85-executive-summary.html` - Step-by-step guide
- `docs/review-2026-10-21-ad5b6dbe.md` - Detailed review

### Research
- `research/h85_hypotheses.md` - 5 ranked hypotheses
- `research/why-h33-scored-02778.md` - H33 analysis

### Code
- `scripts/build_h85_submission.py` - Implementation template

### Registry
- `registry/preregistration_h85.json` - Machine-readable prereg

### Summaries
- `SESSION_SUMMARY_2026-10-21.md` - Executive summary
- `FINAL_DELIVERABLE_SUMMARY.md` - Complete deliverable list

---

## 🏃 NEXT STEPS (Do This First)

### Step 1: Restore Training Data (30 min)
```bash
cd /home/user/GEMSDOE77
bash scripts/download_competition_data.sh
python scripts/prepare_data.py
```

**Expected:** Creates `data/bridge/training_features.tif` (418 MB, 437 bands)

### Step 2: Implement H85 Model (2 hours)
```bash
# Create the H85 directory
mkdir -p src/h85

# Create these files (templates in scripts/build_h85_submission.py):
# - src/h85/model.py (MultiScaleEfficientNet)
# - src/h85/train.py (training script)
# - src/h85/config.py (configuration)
```

### Step 3: Train the Model (4-8 hours with GPU)
```bash
cd src/h85
python train.py --config config_h85.yaml
```

**Note:** Without GPU, training takes 24-48 hours on CPU.

### Step 4: Evaluate (1 hour)
```bash
python scripts/evaluate_holdout.py \
  --model models/h85_efficientnet_v2.pt \
  --seeds 20 30 21
```

**Target:** HOLDOUT-DTI > 0.25 on all seeds

### Step 5: Generate Submission (2 hours)
```bash
python scripts/build_h85_submission.py \
  --surface predictions/h85_surface.tif \
  --threshold 0.3
```

### Step 6: Validate (1 hour)
```bash
# Format validation
python vendor/gems57/validate.py docs/downloads/gems77-h85-...tif

# Leakage canaries
python vendor/gems57/evaluate_holdout.py \
  --candidate docs/downloads/gems77-h85-...tif \
  --mode canary

# Uniqueness checks
# (Need to implement or adapt existing scripts)
```

---

## 📊 KEY METRICS TO TRACK

| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| HOLDOUT-DTI (Seed 20) | > 0.25 | 0.18225 | ⏳ Pending |
| HOLDOUT-DTI (Seed 30) | > 0.25 | 0.18938 | ⏳ Pending |
| HOLDOUT-DTI (3-seed pooled) | > 0.25 | 0.18478 | ⏳ Pending |
| Format Validation | 17/17 PASS | N/A | ⏳ Pending |
| Leakage Canaries | 140/140 PASS | N/A | ⏳ Pending |
| Uniqueness | max_ov ≤ 0.70 | N/A | ⏳ Pending |

---

## 🎓 H85 ARCHITECTURE

```
Input: 437 GeoDAWN bands (3730×3292)
       │
       ▼
┌─────────────────────────────────────┐
│  Multi-Scale Feature Extraction        │
│  ├─ Scale 1: 100m resolution           │
│  ├─ Scale 2: 200m resolution           │
│  └─ Scale 3: 400m resolution           │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  EfficientNetV2-S Backbone             │
│  ├─ Atrous Spatial Pyramid Pooling     │
│  └─ Feature fusion across scales       │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Output: Fault probability surface    │
│  (3730×3292, float32, [0,1])            │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Post-processing:                      │
│  ├─ Thresholding (optimize for DTI)    │
│  ├─ Non-max suppression                │
│  ├─ Distance filtering                 │
│  └─ Catalogue exclusion                │
└─────────────────────────────────────┘
       │
       ▼
    Binary dots TIF (submission)
```

---

## 🔧 IMPLEMENTATION CHECKLIST

### Data Preparation
- [ ] Run `scripts/download_competition_data.sh`
- [ ] Run `scripts/prepare_data.py`
- [ ] Verify `data/bridge/training_features.tif` exists
- [ ] Verify `data/bridge/labels.tif` exists

### Model Implementation
- [ ] Create `src/h85/model.py`
- [ ] Create `src/h85/train.py`
- [ ] Create `src/h85/config.py`
- [ ] Create `src/h85/__init__.py`

### Training
- [ ] Train on visible faults only
- [ ] Use hide-and-recover protocol
- [ ] Implement early stopping
- [ ] Use data augmentation
- [ ] Save model checkpoint

### Evaluation
- [ ] Compute HOLDOUT-DTI on Seed 20
- [ ] Compute HOLDOUT-DTI on Seed 30
- [ ] Compute HOLDOUT-DTI on Seed 21
- [ ] Bootstrap 2,000 draws for 95% CI
- [ ] Verify statistical significance

### Submission Preparation
- [ ] Generate binary dot predictions
- [ ] Apply post-processing
- [ ] Validate format (17 checks)
- [ ] Run leakage canaries (140 checks)
- [ ] Verify uniqueness
- [ ] Generate run card

### Documentation
- [ ] Update `docs/h85.html` with results
- [ ] Update `docs/h85-executive-summary.html`
- [ ] Create run card JSON
- [ ] Create review document

---

## 📚 IMPORTANT FILES TO READ

### Before Starting
1. `PROMPT.md` - Operative brief
2. `AGENTS.md` - Start here guide
3. `docs/executive-summary.html` - General submission guide

### H85-Specific
1. `docs/h85.html` - Technical report
2. `docs/h85-executive-summary.html` - Step-by-step guide
3. `research/h85_hypotheses.md` - Detailed hypotheses
4. `registry/preregistration_h85.json` - Preregistration

### Reference
1. `vendor/gems57/validate.py` - Format validation
2. `vendor/gems57/evaluate_holdout.py` - Holdout evaluation
3. `scripts/build_h85_submission.py` - Implementation template

---

## 💡 TIPS FOR SUCCESS

### Training Tips
- Start with a **small model** (EfficientNetV2-S) to test the pipeline
- Use **mixed precision** training to speed up and reduce memory
- Implement **gradient checkpointing** if memory is limited
- Use **early stopping** to prevent overfitting
- **Data augmentation** is critical for generalization

### Evaluation Tips
- Always evaluate on **multiple seeds** (20, 30, 21)
- Use **bootstrap CI** to assess uncertainty
- Compare against **H77E-ACD** (0.18478) as baseline
- Check **per-fold performance** for consistency

### Optimization Tips
- Optimize **threshold** for best DTI
- Try **different fusion weights** for multi-scale
- Experiment with **different scales** (100m, 200m, 400m)
- Consider **ensemble** of multiple models

### Validation Tips
- Run **all 17 format checks** before submission
- Verify **no NaN values** in the TIF
- Check **EPSG:32611** projection
- Ensure **exact transform** matches sample
- Validate **dot count** is reasonable

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable (Acceptable)
- [ ] HOLDOUT-DTI > 0.18478 (beats repository best)
- [ ] Passes all 17 format validation checks
- [ ] Passes all 140 leakage canaries
- [ ] max_ov ≤ 0.70 (uniqueness)

### Target (Goal)
- [ ] HOLDOUT-DTI > 0.25 (significant improvement)
- [ ] 95% CI excludes 0.18478
- [ ] Consistent performance across all seeds

### Stretch (Ambitious)
- [ ] HOLDOUT-DTI > 0.30 (approaches leaderboard)
- [ ] Top 20-30 leaderboard position
- [ ] Statistical significance vs all repository lanes

---

## 📞 NEED HELP?

### Documentation
- All H85 documentation: `docs/h85*.html`
- All research: `research/h85*.md`
- All scripts: `scripts/build_h85*.py`

### References
- DrivenData Competition: https://www.drivendata.org/competitions/306/
- GEMS Prize Reference: https://github.com/drivendataorg/gems-prize-reference-solution
- EPSG:32611 CRS: https://epsg.io/32611
- Tversky Index: https://en.wikipedia.org/wiki/Tversky_index

---

## ✅ READY TO START?

Run this command to begin:
```bash
cd /home/user/GEMSDOE77 && bash scripts/download_competition_data.sh
```

Then read:
```bash
less docs/h85-executive-summary.html
```

---

**Last Updated:** 2026-10-21  
**Status:** Ready for implementation  
**Next Session:** Implement H85-MULTISCALE-CNN
