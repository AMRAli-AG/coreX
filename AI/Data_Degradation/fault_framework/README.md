# Technical Review Report: Fault Injection Framework

## Overview
This framework provides a physically realistic, modular environment for injecting mechanical faults into robotic arm time-series datasets. Rather than injecting pure random noise, this tool uses targeted mathematical transformations that mirror the true degradation patterns described in robotic research literature.

## 1. Feature Mapping Logic
Robotic datasets contain complex kinematics and dynamics data. Our `data_manager.py` dynamically maps columns based on standard conventions (e.g., matching `joint1_q` or `actual_q_0`):
- **`q` (Position)**: Used to identify the actual spatial configuration and calculate velocity derivatives if not explicitly provided.
- **`qd` (Velocity)**: Crucial for identifying direction shifts (zero-crossings) which trigger phenomena like backlash.
- **`current` (Motor Current)**: The primary indicator of effort. Increased physical resistance (friction, payload) directly causes a spike in current to maintain the required target moment.
- **`temp` (Joint Temperature)**: Excess electrical and mechanical energy lost to friction is dissipated as heat, providing a delayed but heavily correlated indicator of internal mechanical wear.

## 2. Mathematical Logic for Degradations

### FAULT_BL (Gear Backlash)
- **Logic:** Backlash is the gap between gear teeth. It only manifests when the motor changes direction.
- **Math:** We compute the derivative of `target_q` to find velocity sign flips. During these flips, a positional lag (offset) is subtracted from `actual_q`. The magnitude of this lag increases *exponentially* over the duration window to simulate the accelerating nature of gear tooth wear.

### FAULT_LF (Lubrication Failure)
- **Logic:** Loss of grease increases dynamic friction. The motor must draw more current, generating excess heat.
- **Math:** `actual_current` is multiplied by a linearly increasing severity factor (up to a 30% increase). Simultaneously, an additive linear slope is applied to `joint_temperature` to simulate the thermal coupling effect.

### FAULT_BW (Bearing Wear)
- **Logic:** Spalling or pitting in the bearings creates physical micro-shocks as the balls roll over the pits.
- **Math:** We inject high-frequency Gaussian noise into `actual_qd` and `actual_current`. As severity increases, the RMS (Root Mean Square) of the noise increases. We also introduce sparse, extreme-amplitude spikes using a probabilistic mask; this artificially raises the *Kurtosis* of the signal, which is the gold-standard statistical indicator for bearing wear.

## 3. Combo and Propagation Scenarios

### FAULT_SYS (System Propagation / Payload Change)
- **Logic:** If the robot picks up an unexpected heavy object, or suffers a soft collision, the entire kinematic chain feels the resistance.
- **Math:** Instead of targeting a single joint index, this injector sweeps through the `actual_current` columns of *all* joints simultaneously, applying a global multiplier. Additionally, Gaussian error is added to the Tool Center Point (TCP) coordinates, simulating structural deflection under the heavy load.

### Combinations (Combo Injector)
- **Logic:** Faults rarely happen in pure isolation. A worn bearing often leads to excess heat, which degrades lubrication.
- **Execution:** The `ComboInjector` accepts a list of base injectors. It streams the data chunk sequentially through each injector's `inject()` method. For example, injecting Backlash *then* Bearing Wear on the same joint, ensuring compounding mathematical transformations accurately reflect a dying joint.

## 4. Instructions for Use
1. **Prepare Data:** Ensure your dataset (CSV/Excel) is accessible. The framework handles large files via chunking.
2. **Run:** Execute `python main.py` in your terminal.
3. **Input:** The script will interactively ask:
   - File path (e.g., `../ready_for_ai.csv`)
   - Which Joint? (e.g., `0` for Joint 1)
   - Which Fault? (Select 1-5 from the menu)
   - Starting Time? (Row index to begin injection)
   - Duration? (Number of rows the fault will span)
4. **Output:** The framework outputs a new CSV file prefixed with `injected_` containing two new automated labeling columns: `fault_label` and `severity_score` (ranging from 0.0 to 1.0 based on degradation progression).
