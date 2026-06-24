# utils/phm_engine.py

import numpy as np
from utils.fault_kb import FaultKB

class PHMEngine:
    """
    Autonomous PHM Logic for Robot Arm Fault Injection.
    Calculates injection windows based on industry standard failure profiles.
    """
    
    @staticmethod
    def calculate_window(total_rows, fault_key):
        """
        Determines the start_idx and duration autonomously.
        
        Logic: 
        - Faults typically begin after a "Healthy" burn-in period.
        - Start time is usually around 30% of the lifecycle for incipient faults.
        - Duration is calculated based on the 'incipient_period' defined in FaultKB.
        """
        profile = FaultKB.PROFILES.get(fault_key, {})
        incipient_period = profile.get('incipient_period', 0.5)
        
        # Start fault at 25% to 35% of the dataset to ensure a healthy baseline
        start_pct = np.random.uniform(0.25, 0.35)
        start_idx = int(total_rows * start_pct)
        
        # Duration depends on the fault type. Collision is fast, Friction is slow.
        max_possible_duration = total_rows - start_idx - 100 # buffer at end
        target_duration = int(total_rows * incipient_period)
        
        duration = min(target_duration, max_possible_duration)
        
        print(f"[PHM Engine] Autonomous Calculation for {fault_key}:")
        print(f"   - Dataset Lifecycle: {total_rows} rows")
        print(f"   - Injection Start: Row {start_idx} ({start_pct:.1%})")
        print(f"   - Progression Window: {duration} rows")

        
        return start_idx, duration
