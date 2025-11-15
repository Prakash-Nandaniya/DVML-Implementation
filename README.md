# Deep Variational Metric Learning (DVML)

**Implementation of the ECCV 2018 paper:**  
[Deep Variational Metric Learning](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.pdf)  
*Xudong Lin, Yueqi Duan, Qiyuan Dong, Jiwen Lu, Jie Zhou (Tsinghua University)*

---

## Problem Solved

Traditional deep metric learning methods assume **all samples from the same class should map to a single point** in the embedding space.

This leads to:
- **Overfitting** on limited training data.
- Ignoring **intra-class variance** (pose, viewpoint, illumination, etc.).
- **Poor generalization** to unseen classes.

Models become sensitive to training-specific variations and fail to learn robust decision boundaries.

---

## What the Paper Solves

**DVML** introduces a key insight:

> **Intra-class variance is class-independent** — the same type of variation (e.g., pose change) affects features across different classes in a similar way.

### Core Idea:
Disentangle:
- **z<sub>I</sub>** → *Intra-class invariance* (class center)
- **z<sub>V</sub>** → *Intra-class variance* (modeled as isotropic Gaussian)

Then **generate hard synthetic samples** by:
ẑ = z_I + z_V  (sampled from N(μ, σ²I))

### Training with 4 Losses:
1. **KL Divergence** → Enforce Gaussian prior on variance
2. **Reconstruction Loss** → Preserve sample-specific info
3. **Metric Loss on z<sub>I</sub>** → Learn discriminative class centers
4. **Metric Loss on ẑ** → Robust boundaries using generated samples

Result: **Better generalization, robustness, and compatibility with any metric loss.**

---

## What This Implementation Does

Reproduces **DVML from scratch in PyTorch** with:
- GoogleNet backbone (1024-D features)
- Variational encoder for `μ`, `log σ²`
- Decoder reconstructs 1024-D features (not full images)
- **T=20** synthetic samples per input
- Two-phase training (decoder gradients detached in phase 1)
- KL annealing over 30 epochs
- Proxy-NCA++ with label smoothing & proxy L2
- Zero-shot evaluation: **100 seen → 100 unseen classes** (CUB-200-2011)
- Metrics: Recall@K, NMI, F1, Precision/Recall
- Visualizations: PCA & t-SNE of learned embeddings

Demonstrates **DVML improves generalization to unseen classes** by explicitly modeling and leveraging intra-class variance.

---

> *"We are the first to utilize variational inference to disentangle intra-class variance and generate discriminative samples to improve robustness."*  
> — Lin et al., ECCV 2018
