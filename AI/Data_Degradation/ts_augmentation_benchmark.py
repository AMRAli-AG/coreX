# -*- coding: utf-8 -*-
"""
Robotic Telemetry Augmentation Evaluator & Benchmark Suite.
Redirected to the new, research-validated dynamic evaluation pipeline.
"""

import sys
import os

if __name__ == "__main__":
    # Standard redirect to the new dynamic orchestrator
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    import evaluate_augmentations
    evaluate_augmentations.main()
