# Integrated 2-Stage Augmentation & Injection Framework

The framework has been completely redesigned and modularized to support your 2-stage integrated pipeline requirement. It now supports expanding existing datasets robustly, and mathematically injecting physically realistic faults with an interactive terminal interface.

## 1. Directory Structure Implemented
The new structure perfectly matches your specifications, ensuring clean modularity and scalability:
```
Data_Degradation/
├── augmentation/
│   └── expander.py        # Stage 1: Data expansion/concatenation
├── injectors/
│   ├── base_injector.py
│   ├── local/             # Stage 2: Localized faults
│   │   ├── backlash.py 
│   │   ├── friction.py 
│   │   └── bearing_wear.py
│   ├── propagation/       # Stage 2: System-wide faults
│   │   └── payload_collision.py
│   └── combos/            # Stage 2: Multi-fault scenarios
│       └── combo.py
├── utils/
│   ├── data_manager.py    # Chunking & automated feature mapping
│   └── labeler.py         # Precision labeling & severity scoring
├── main.py                # Interactive CLI application
├── requirements.txt       # Dependencies
├── README.md              # How-To-Run Guide
└── Technical_Review.md    # Physical & Mathematical Documentation
```

## 2. Interactive Terminal (`main.py`)
Running `python main.py` provides a rich, interactive terminal experience:
- **Welcome & Analysis:** Automatically scans for `ready_for_ai.csv` and prints a formatted summary table of its dimensions.
- **Stage 1 (Augmentation):** Prompts you to expand the dataset via a multiplier (e.g., 2x, 5x, 10x). It uses continuous time-stitching to safely repeat the kinematic cycles without breaking physical relationships.
- **Stage 2 (Fault Injection):** Displays a menu of "Diseases" (BL, LF, BW, SYS, COMBO) and interactively asks for the target joint index, starting row, and duration.
- **Real-time Logging:** As the injection executes, it prints a real-time terminal log (e.g., `-> Injecting FAULT_LF: Current-Thermal coupling active on Joint 2...`).

## 3. Mathematical Logic & Labeling
All physical accuracy rules have been strictly maintained:
- **Incipient Degradation:** The severity scores organically grow from 0.0 (Healthy) to 1.0 (Critical) over the duration window you specify. 
- **Modular Workflow:** The new `Labeler` utility centrally manages the `fault_label` and `severity_score` columns, ensuring precise, overlapping annotations even for combo faults.
- **Physics-Based Algorithms:** Backlash applies exponential lag on velocity sign-flips. Friction coupling linearly increases current and temperature. Bearing Wear injects high-frequency RMS noise and extreme peaks to raise Kurtosis. Payload propagation scales all active current columns by 1.5x.

## 4. Visualization & Output
Upon completion, the framework saves your data as `labeled_robot_data.csv`. A detailed summary report table is printed to the terminal, and `matplotlib` is automatically triggered to plot the degradation signal, color-coded by the continuous `severity_score`.

> [!TIP]
> You can launch the pipeline directly in your terminal using:
> ```bash
> python main.py
> ```
> Follow the interactive prompts to generate the custom datasets.
