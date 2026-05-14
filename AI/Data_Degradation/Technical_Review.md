# Technical Review Report: Fault Injection Framework

## Overview
This framework provides a physically realistic, 2-stage integrated pipeline for preparing and corrupting robotic arm time-series datasets. 

## Stage 1: Data Augmentation & Expansion
Deep learning algorithms for predictive maintenance require large datasets. The `DataExpander` module addresses this by expanding the provided baseline data (e.g., `ready_for_ai.csv`).
To preserve the physical relationships (the mathematical correlations between position, velocity, and current), the expansion relies on continuous, repetitive time-stitching. Rather than utilizing generative adversarial networks (GANs) which may accidentally generate mathematically impossible kinematic trajectories, this module repeats the exact recorded kinematic cycle, smoothly adjusting the time-index to maintain continuity. 

## Stage 2: Realistic Fault Injection
Instead of injecting pure random noise, this tool uses targeted mathematical transformations that mirror the true degradation patterns described in robotic research literature.

### 1. Feature Mapping Logic
Our `DataManager` dynamically maps columns based on standard conventions (e.g., matching `joint1_q` or `actual_current_2`):
- **`q` (Position)**: Used to identify the actual spatial configuration and calculate velocity derivatives.
- **`qd` (Velocity)**: Crucial for identifying direction shifts (zero-crossings) which trigger phenomena like backlash.
- **`current` (Motor Current)**: The primary indicator of effort. Increased physical resistance (friction, payload) directly causes a spike in current.
- **`temp` (Joint Temperature)**: Excess energy lost to friction is dissipated as heat.

### 2. Mathematical Logic for Degradations
#### FAULT_BL (Gear Backlash)
- **Logic:** Backlash is the gap between gear teeth. It only manifests when the motor changes direction.
- **Math:** We compute the derivative of `target_q` to find velocity sign flips. During these flips, a positional lag (offset) is subtracted from `actual_q`. The magnitude of this lag increases *exponentially* over the duration window to simulate the accelerating nature of gear tooth wear.

#### FAULT_LF (Lubrication Failure)
- **Logic:** Loss of grease increases dynamic friction. The motor must draw more current, generating excess heat.
- **Math:** `actual_current` is multiplied by a linearly increasing severity factor (up to a 30% increase). Simultaneously, an additive linear slope is applied to `joint_temperature` to simulate the thermal coupling effect.

#### FAULT_BW (Bearing Wear)
- **Logic:** Spalling or pitting in the bearings creates physical micro-shocks as the balls roll over the pits.
- **Math:** We inject high-frequency stochastic noise into `actual_qd` and `actual_current`. As severity increases, the RMS (Root Mean Square) of the noise increases. We also introduce sparse, extreme-amplitude spikes using a probabilistic mask; this artificially raises the *Kurtosis* of the signal.

#### FAULT_SYS (System Propagation / Payload Collision)
- **Logic:** If the robot picks up an unexpected heavy object, or suffers a soft collision, the entire kinematic chain feels the resistance.
- **Math:** This injector sweeps through the `actual_current` columns of *all* joints simultaneously, applying a global 1.5x multiplier to simulate a 50% jump in payload. Additionally, Gaussian error is added to the Tool Center Point (TCP) coordinates.

## Labeling and Severity Scoring
Every fault injection automatically adds two new columns to the resulting `labeled_robot_data.csv`:
1. `fault_label`: A categorical string denoting the presence of a specific fault.
2. `severity_score`: A continuous float from 0.0 to 1.0 indicating the progression level of the incipient fault.

This labeling allows downstream machine learning models to perform not only classification (What is the fault?) but also prognostics/RUL prediction (How severe is it?).
