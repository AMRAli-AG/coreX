from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import pandas as pd

from utils.labeler import Labeler


class BaseInjector(ABC):
    """
    Abstract base class for all fault injectors.
    Provides the standard interface and registry-backed factory helpers.
    """

    def __init__(self, start_idx: int, end_idx: int, joint_index: int | None = None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.joint_index = joint_index
        self.fault_name = "BaseFault"

    @abstractmethod
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        pass

    def get_chunk_overlap(self, df: pd.DataFrame):
        """Returns the start and end indices within the current dataframe chunk."""
        chunk_start = df.index.min()
        chunk_end = df.index.max()

        overlap_start = max(chunk_start, self.start_idx)
        overlap_end = min(chunk_end + 1, self.end_idx)

        if overlap_start < overlap_end:
            return overlap_start, overlap_end
        return None, None

    @staticmethod
    def from_profile(
        profile_id: str,
        start_idx: int,
        end_idx: int,
        joint_index: Optional[int] = None,
    ) -> "BaseInjector":
        """Build an injector instance from a registry profile_id."""
        from injectors.fault_registry import build_injector

        return build_injector(profile_id, start_idx, end_idx, joint_index)

    @staticmethod
    def build_suite(
        start_idx: int,
        end_idx: int,
        profiles: Optional[Sequence] = None,
    ) -> List["BaseInjector"]:
        """
        Build one injector per localized profile (default: all 14).
        Joint targeting uses each profile's first eligible joint when not distributed.
        """
        from injectors.fault_registry import LOCAL_FAULT_PROFILES

        suite_profiles = profiles if profiles is not None else LOCAL_FAULT_PROFILES
        injectors: List[BaseInjector] = []
        for profile in suite_profiles:
            joint = None if profile.distributed else profile.eligible_joints[0]
            injectors.append(profile.build(start_idx, end_idx, joint))
        return injectors
