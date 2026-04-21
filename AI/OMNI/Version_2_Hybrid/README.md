# OmniAnomaly Version 2: Hybrid Optimized

This folder contains the **final, optimized hybrid architecture** of OmniAnomaly. It features gated recurrent units (GRU), a **Causal Graph Module**, and advanced feature engineering for superior anomaly detection.

## 🚀 Quick Run (Evaluation)
To run the model using the pre-trained optimized checkpoint:
```bash
python main.py --restore_dir model_coreX_v2_optimized --max_epoch 0
```

## 🛠️ Full Training
To retrain the model from scratch:
```bash
python main.py
```

## 🏗️ Key Upgrades vs V1
- **Relational Awareness**: Uses `causal_adj_matrix.npy` to model sensor dependencies.
- **Improved Stability**: Patched probability wrappers for newer TFP/TF environments.
- **Enhanced Accuracy**: Association Discrepancy logic for better sensitivity.

## 📂 Structure
- `main.py`: Entry point for the optimized pipeline.
- `data/processed/`: Version 2 datasets.
- `model_coreX_v2_optimized/`: Final production-ready weights.

*For full installation instructions, refer to the [Master Guide](../README_GLOBAL.md).*
