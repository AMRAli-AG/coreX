import pandas as pd
import numpy as np

from injectors.base_injector import BaseInjector
from utils.labeler import Labeler


class PayloadChangeInjector(BaseInjector):
    """
    Scenario: Payload/Collision (FAULT_SYS)
    Physics: Half-sine impulse spike followed by damped sinusoidal oscillation.
    System-wide shock across all 6 joints.
    Registered as profile_id SYS_COLLISION in PROPAGATION_FAULT_PROFILES.
    """

    PROFILE_ID = 'SYS_COLLISION'

    def __init__(self, start_idx: int, end_idx: int, joint_index: int | None = None):
        super().__init__(start_idx, end_idx, None)
        self.fault_name = 'FAULT_SYS'
        self._has_printed = False
        from injectors.fault_registry import get_profile

        self._profile = get_profile(self.PROFILE_ID)

    @classmethod
    def profile(cls) -> "object":
        from injectors.fault_registry import get_profile

        return get_profile(cls.PROFILE_ID)

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df

        if not self._has_printed:
            print(
                f"-> [{self._profile.profile_id}] Injecting {self.fault_name}: "
                "High-amplitude system-wide collision shock (all joints)..."
            )
            self._has_printed = True

        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = int(mask.sum())
        if duration == 0:
            return df

        severity = Labeler.calculate_severity(
            overlap_start,
            overlap_end,
            self.start_idx,
            self.end_idx,
            profile=self._profile.severity_profile,
        )

        impulse_len = 20
        impulse = np.zeros(duration)
        if duration > impulse_len:
            impulse[:impulse_len] = (
                np.sin(np.pi * np.arange(impulse_len) / impulse_len) * 3.0
            )

        settling_start = impulse_len
        if duration > settling_start:
            t_settle = np.arange(duration - settling_start)
            decay, freq = 0.05, 0.5
            impulse[settling_start:] = (
                1.0 * np.exp(-decay * t_settle) * np.sin(freq * t_settle)
            )

        shock_wave = impulse
        cur_cols = feature_map.get('actual_current', [])

        for col in cur_cols:
            if col and col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] * (1 + shock_wave)

        vel_cols = feature_map.get('actual_qd', [])
        for col in vel_cols:
            if col and col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] + (0.15 * shock_wave)

        return Labeler.apply_labels(
            df, overlap_start, overlap_end, self.fault_name, severity
        )


def list_propagation_profiles():
    """Return all registered propagation-layer fault profiles."""
    from injectors.fault_registry import PROPAGATION_FAULT_PROFILES

    return PROPAGATION_FAULT_PROFILES
