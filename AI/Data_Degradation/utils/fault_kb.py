# utils/fault_kb.py

from injectors.fault_registry import LOCAL_FAULT_PROFILES


class FaultKB:
    """
    Robotic Failure Knowledge Base (Encyclopedia of Robotic Diseases).
    Defines physical characteristics, causes, and cascading effects of failures.
    Synchronized with the 14-profile localized injector registry.
    """

    PROFILES = {
        'FAULT_BL': {
            'name': 'Gear Backlash (Positional Lag)',
            'description': 'Excessive clearance between gear teeth causing a dead-zone in motion.',
            'primary_symptom': 'Positional error at direction changes.',
            'physical_coupling': 'Accelerates Bearing Wear due to impact loading.',
            'incipient_period': 0.4,
            'registry_profiles': ['BL_FULL', 'BL_COMPACT'],
        },
        'FAULT_LF': {
            'name': 'Lubrication Failure (Friction-Thermal)',
            'description': 'Degradation of grease/oil leading to increased friction and heat.',
            'primary_symptom': 'Increase in RMS Current and Joint Temperature.',
            'physical_coupling': 'Directly triggers Bearing Wear and Thermal Expansion.',
            'incipient_period': 0.6,
            'registry_profiles': ['LF_DIST_FULL', 'LF_JOINT_FULL', 'LF_COMPACT'],
        },
        'FAULT_BW': {
            'name': 'Bearing Wear (Stochastic Degradation)',
            'description': 'Pitting or flaking on bearing races causing vibration and noise.',
            'primary_symptom': 'High-frequency stochastic noise in velocity and current.',
            'physical_coupling': 'Leads to Backlash and system-wide vibration propagation.',
            'incipient_period': 0.5,
            'registry_profiles': ['BW_FULL', 'BW_COMPACT'],
        },
        'FAULT_EE': {
            'name': 'Electrical Erosion (EDM Fluting)',
            'description': 'Parasitic bearing currents causing arcing and step-wise thermal jumps.',
            'primary_symptom': 'Wideband current hiss, arcing spikes, and thermal steps.',
            'physical_coupling': 'Accelerates Bearing Wear via fluted race friction.',
            'incipient_period': 0.35,
            'registry_profiles': ['EE_FULL', 'EE_COMPACT'],
        },
        'FAULT_MD': {
            'name': 'Permanent Magnet Demagnetization',
            'description': 'Kt decay in PMSM actuators causing torque-current divergence.',
            'primary_symptom': 'Rising current for constant torque; thermal vicious cycle.',
            'physical_coupling': 'Thermal runaway can trigger lubrication breakdown.',
            'incipient_period': 0.55,
            'registry_profiles': ['MD_INC_FULL', 'MD_ABR_FULL', 'MD_COMPACT'],
        },
        'FAULT_ED': {
            'name': 'Encoder Drift & Cable Fatigue',
            'description': 'Feedback measurement bias and phantom PID corrections.',
            'primary_symptom': 'Cumulative positional bias and erratic current jitter.',
            'physical_coupling': 'Masks backlash until catastrophic mis-tracking.',
            'incipient_period': 0.5,
            'registry_profiles': ['ED_FULL', 'ED_COMPACT'],
        },
        'FAULT_SYS': {
            'name': 'System Payload Collision',
            'description': 'Sudden change in inertia or external impact on the manipulator.',
            'primary_symptom': 'Impulsive current spike across all joints.',
            'physical_coupling': 'Can cause immediate structural damage or permanent backlash.',
            'incipient_period': 0.05,
            'registry_profiles': ['SYS_COLLISION'],
        },
        
    }

    @staticmethod
    def get_cascading_effects(fault_key: str) -> list[str]:
        """Returns fault_keys likely triggered by the primary fault."""
        couplings = {
            'FAULT_LF': ['FAULT_BW'],
            'FAULT_BW': ['FAULT_BL'],
            'FAULT_BL': ['FAULT_BW'],
            'FAULT_EE': ['FAULT_BW'],
            'FAULT_MD': ['FAULT_LF'],
            'FAULT_ED': ['FAULT_BL'],
            'FAULT_SYS': ['FAULT_BL', 'FAULT_BW'],
        }
        return couplings.get(fault_key, [])

    @staticmethod
    def localized_profile_ids() -> list[str]:
        return [p.profile_id for p in LOCAL_FAULT_PROFILES]

    @staticmethod
    def profile_count() -> int:
        return len(LOCAL_FAULT_PROFILES)
