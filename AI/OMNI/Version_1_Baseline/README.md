# OmniAnomaly Version 1: Baseline

This folder contains the **original baseline implementation** of OmniAnomaly. Use this version to establish a performance benchmark before testing the Hybrid architecture.

## 🚀 Quick Run (Evaluation)
To run the model using the pre-trained checkpoint and see results immediately:
```bash
python main.py --restore_dir model_coreX_v1 --max_epoch 0
```

## 🛠️ Full Training
To retrain the model from scratch:
```bash
python main.py
```

## 📂 Structure
- `main.py`: Entry point for training and evaluation.
- `data/processed/`: Contains the specific windowed dataset for V1.
- `model_coreX_v1/`: Pre-trained model weights.
- `omni_anomaly/`: Core model logic and wrappers.

*For full installation instructions, refer to the [Master Guide](../README_GLOBAL.md).*
