import pandas as pd

from injectors.base_injector import BaseInjector
from injectors.fault_registry import (
    LOCAL_FAULT_PROFILES,
    build_injector,
    get_profile,
)


class ComboInjector(BaseInjector):
    """
    Composite multi-fault injector.
    Executes any combination of registry-defined localized profiles sequentially.
    """

    def __init__(
        self,
        start_idx: int,
        end_idx: int,
        joint_index: int | None,
        injectors: list[BaseInjector],
        profile_ids: list[str] | None = None,
    ):
        super().__init__(start_idx, end_idx, joint_index)
        self.injectors = injectors
        self.profile_ids = profile_ids or []
        self.fault_name = "_PLUS_".join(inj.fault_name for inj in injectors)
        self._has_printed = False

    @classmethod
    def from_profile_ids(
        cls,
        profile_ids: list[str],
        start_idx: int,
        end_idx: int,
        joint_index: int | None = None,
    ) -> "ComboInjector":
        """
        Build a combo from explicit registry profile IDs (e.g. ['BL_FULL', 'BW_FULL']).
        Validates every ID against the full 14-profile localized suite.
        """
        local_ids = {p.profile_id for p in LOCAL_FAULT_PROFILES}
        unknown = [pid for pid in profile_ids if pid not in local_ids]
        if unknown:
            raise ValueError(
                f"Invalid combo profile_id(s): {unknown}. "
                f"Must be one of: {sorted(local_ids)}"
            )

        injectors: list[BaseInjector] = []
        resolved_joint = joint_index

        for pid in profile_ids:
            profile = get_profile(pid)
            if profile.distributed:
                inj = build_injector(pid, start_idx, end_idx)
            else:
                joint = resolved_joint if resolved_joint is not None else profile.eligible_joints[0]
                inj = build_injector(pid, start_idx, end_idx, joint)
            injectors.append(inj)

        return cls(start_idx, end_idx, resolved_joint, injectors, profile_ids)

    @classmethod
    def from_fault_keys(
        cls,
        fault_keys: list[str],
        start_idx: int,
        end_idx: int,
        joint_index: int | None = None,
    ) -> "ComboInjector":
        """
        Build a combo using the default full-tier profile for each fault_key.
        Maps FAULT_BL -> BL_FULL, FAULT_LF -> LF_DIST_FULL, etc.
        """
        default_map = {
            p.fault_key: p.profile_id
            for p in LOCAL_FAULT_PROFILES
            if p.implementation == 'full' and p.profile_id != 'LF_JOINT_FULL'
        }
        default_map['FAULT_LF'] = 'LF_DIST_FULL'
        default_map['FAULT_MD'] = 'MD_INC_FULL'

        profile_ids = []
        for key in fault_keys:
            if key not in default_map:
                raise ValueError(f"No default profile for fault_key '{key}'")
            profile_ids.append(default_map[key])

        return cls.from_profile_ids(profile_ids, start_idx, end_idx, joint_index)

    @classmethod
    def cascading_from_primary(
        cls,
        primary_profile_id: str,
        start_idx: int,
        end_idx: int,
        joint_index: int | None = None,
    ) -> "ComboInjector":
        """
        Build a physics-informed cascade combo from FaultKB coupling rules.
        """
        from utils.fault_kb import FaultKB

        primary = get_profile(primary_profile_id)
        coupled_keys = FaultKB.get_cascading_effects(primary.fault_key)
        if not coupled_keys:
            inj = primary.build(start_idx, end_idx, joint_index)
            return cls(start_idx, end_idx, joint_index, [inj], [primary_profile_id])

        profile_ids = [primary_profile_id]
        for key in coupled_keys:
            full_profiles = [
                p.profile_id
                for p in LOCAL_FAULT_PROFILES
                if p.fault_key == key and p.implementation == 'full'
                and p.profile_id != 'LF_JOINT_FULL'
            ]
            if full_profiles:
                profile_ids.append(full_profiles[0])

        return cls.from_profile_ids(profile_ids, start_idx, end_idx, joint_index)

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        result_df = df.copy()

        overlap_start, overlap_end = self.get_chunk_overlap(result_df)
        if overlap_start is None:
            return result_df

        if not self._has_printed:
            ids = ", ".join(self.profile_ids) if self.profile_ids else self.fault_name
            scope = f"Joint {self.joint_index}" if self.joint_index is not None else "Multi-joint"
            print(f"-> Injecting COMBO [{ids}] on {scope}...")
            self._has_printed = True

        for inj in self.injectors:
            inj.start_idx = self.start_idx
            inj.end_idx = self.end_idx
            if self.joint_index is not None and inj.joint_index is not None:
                inj.joint_index = self.joint_index
            result_df = inj.inject(result_df, feature_map)

        mask = (result_df.index >= overlap_start) & (result_df.index < overlap_end)
        if 'fault_label' in result_df.columns:
            result_df.loc[mask, 'fault_label'] = self.fault_name

        return result_df
