# Tredence AI Engineering Intern - Case Study Report

**Problem Title:** The Self-Pruning Neural Network

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity
We associate each weight with a learnable gate parameter (sigmoid output between 0 and 1).  
We add an L1 regularization term (λ × sum of gate values) to the classification loss.  
The L1 penalty naturally drives many gate values towards exactly 0 during training, which multiplies the corresponding weights by nearly zero and effectively prunes them on-the-fly.

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|--------------------|
| 0.0001 | 56.72             | 0.00               |
| 0.001  | 53.82             | 0.00               |
| 0.01   | 48.37             | 0.00               |

**Best model:** λ = 0.0001 with Test Accuracy = 56.72% and Sparsity = 0.00%

## Gate Distribution Plot (Best Model)
![Distribution of Final Gate Values](gate_distribution.png)

The plot shows the final distribution of gate values for the best model.

## Implementation Summary
- Custom `PrunableLinear` layer with learnable `gate_scores`
- Correct forward pass and gradient flow through pruned weights
- Custom sparsity loss added to CrossEntropyLoss
- Trained on CIFAR-10 dataset with data augmentation
- Single clean Python file with full training and evaluation loop

**Files in this repository:**
- `self_pruning_nn.py` → Complete implementation
- `report.md` → This report
- `gate_distribution.png` → Gate distribution plot

---

**Submitted for Tredence AI Engineering Intern Position (2026 Cohort)**