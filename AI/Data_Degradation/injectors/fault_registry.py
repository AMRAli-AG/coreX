"""
Central registry for all 14 localized fault/degradation profiles and propagation modes.

Architecture:
  - 12 single-fault mechanisms: 6 physics-informed modules + 6 structural injectors
  - 2 dynamic MD variants (incipient / abrupt) counted in the full tier
  - Propagation (FAULT_SYS) and combo composition sit outside the 14-local set
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Type

# --- Full physics-informed local modules (tier A) ---
from injectors.local.backlash import BacklashInjector as BacklashInjectorFull
from injectors.local.friction import LubricationFailureInjector as LubricationFailureInjectorFull
from injectors.local.bearing_wear import BearingWearInjector as BearingWearInjectorFull
from injectors.local.electrical_erosion import ElectricalErosionInjector as ElectricalErosionInjectorFull
from injectors.local.demagnetization import DemagnetizationInjector as DemagnetizationInjectorFull
from injectors.local.encoder_drift import EncoderDriftInjector as EncoderDriftInjectorFull

# --- Structural / compact local injectors (tier B) ---
from injectors.local.backlash_injector import BacklashInjector as BacklashInjectorCompact
from injectors.local.lubrication_injector import LubricationFailureInjector as LubricationFailureInjectorCompact
from injectors.local.bearing_wear_injector import BearingWearInjector as BearingWearInjectorCompact
from injectors.local.electrical_erosion_injector import ElectricalErosionInjector as ElectricalErosionInjectorCompact
from injectors.local.demagnetization_injector import DemagnetizationInjector as DemagnetizationInjectorCompact
from injectors.local.encoder_drift_injector import EncoderDriftInjector as EncoderDriftInjectorCompact

from injectors.propagation.payload_collision import PayloadChangeInjector


@dataclass(frozen=True)
class FaultProfile:
    """Declarative configuration for one injectable degradation profile."""

    profile_id: str
    fault_key: str
    name: str
    injector_cls: Type[Any]
    eligible_joints: Tuple[int, ...]
    distributed: bool
    severity_profile: str
    implementation: str  # 'full' | 'compact'
    mode: Optional[str] = None  # e.g. demagnetization: 'incipient' | 'abrupt'
    layer: str = 'local'  # 'local' | 'propagation'
    reasoning: str = ''

    def build(
        self,
        start_idx: int,
        end_idx: int,
        joint_index: Optional[int] = None,
        mode_override: Optional[str] = None,
    ) -> Any:
        """Instantiate the injector for this profile with validated joint targeting."""
        if self.distributed:
            joint_index = None
        elif joint_index is None:
            joint_index = self.eligible_joints[0]

        if joint_index is not None and joint_index not in self.eligible_joints:
            raise ValueError(
                f"Joint {joint_index} is ineligible for {self.profile_id} "
                f"(allowed: {self.eligible_joints})"
            )

        md_mode = mode_override or self.mode or 'incipient'

        if self.injector_cls is DemagnetizationInjectorFull:
            return DemagnetizationInjectorFull(
                start_idx, end_idx, joint_index, mode=md_mode
            )
        if self.injector_cls is DemagnetizationInjectorCompact:
            return DemagnetizationInjectorCompact(
                start_idx, end_idx, joint_index, mode=md_mode
            )
        if self.injector_cls is LubricationFailureInjectorFull:
            return LubricationFailureInjectorFull(start_idx, end_idx, joint_index)
        if self.injector_cls in (
            LubricationFailureInjectorCompact,
            PayloadChangeInjector,
        ):
            return self.injector_cls(start_idx, end_idx)

        return self.injector_cls(start_idx, end_idx, joint_index)


# ---------------------------------------------------------------------------
# 14 localized anomaly / degradation profiles
# 12 mechanisms (6 faults × 2 implementation tiers) + 2 MD dynamic modes (full tier)
# ---------------------------------------------------------------------------
LOCAL_FAULT_PROFILES: Tuple[FaultProfile, ...] = (
    # --- Tier A: physics-informed (8 profiles; MD splits into incipient + abrupt) ---
    FaultProfile(
        profile_id='BL_FULL',
        fault_key='FAULT_BL',
        name='Gear Backlash (Physics-Informed)',
        injector_cls=BacklashInjectorFull,
        eligible_joints=(0, 1, 2),
        distributed=False,
        severity_profile='exponential',
        implementation='full',
        reasoning='Cycloidal/harmonic gear dead-zone at base, shoulder, and elbow.',
    ),
    FaultProfile(
        profile_id='LF_DIST_FULL',
        fault_key='FAULT_LF',
        name='Lubrication Failure — Distributed (Physics-Informed)',
        injector_cls=LubricationFailureInjectorFull,
        eligible_joints=tuple(range(6)),
        distributed=True,
        severity_profile='exponential',
        implementation='full',
        reasoning='System-wide boundary lubrication transition with joint-weighted load.',
    ),
    FaultProfile(
        profile_id='LF_JOINT_FULL',
        fault_key='FAULT_LF',
        name='Lubrication Failure — Localized Joint (Physics-Informed)',
        injector_cls=LubricationFailureInjectorFull,
        eligible_joints=tuple(range(6)),
        distributed=False,
        severity_profile='exponential',
        implementation='full',
        reasoning='Single-joint viscous friction escalation before chain-wide propagation.',
    ),
    FaultProfile(
        profile_id='BW_FULL',
        fault_key='FAULT_BW',
        name='Bearing Wear RCF (Physics-Informed)',
        injector_cls=BearingWearInjectorFull,
        eligible_joints=(0, 1, 4, 5),
        distributed=False,
        severity_profile='linear',
        implementation='full',
        reasoning='MCSA sidebands and spall pulses on high-load / high-speed joints.',
    ),
    FaultProfile(
        profile_id='EE_FULL',
        fault_key='FAULT_EE',
        name='Electrical Erosion EDM (Physics-Informed)',
        injector_cls=ElectricalErosionInjectorFull,
        eligible_joints=tuple(range(6)),
        distributed=False,
        severity_profile='linear',
        implementation='full',
        reasoning='Parasitic bearing currents and step-wise thermal fluting.',
    ),
    FaultProfile(
        profile_id='MD_INC_FULL',
        fault_key='FAULT_MD',
        name='Demagnetization — Incipient (Physics-Informed)',
        injector_cls=DemagnetizationInjectorFull,
        eligible_joints=(1, 2),
        distributed=False,
        severity_profile='exponential',
        implementation='full',
        mode='incipient',
        reasoning='Gradual Kt decay under sustained holding torque on shoulder/elbow.',
    ),
    FaultProfile(
        profile_id='MD_ABR_FULL',
        fault_key='FAULT_MD',
        name='Demagnetization — Abrupt (Physics-Informed)',
        injector_cls=DemagnetizationInjectorFull,
        eligible_joints=(1, 2),
        distributed=False,
        severity_profile='exponential',
        implementation='full',
        mode='abrupt',
        reasoning='Step-drop Kt from thermal overload or short-circuit event.',
    ),
    FaultProfile(
        profile_id='ED_FULL',
        fault_key='FAULT_ED',
        name='Encoder Drift (Physics-Informed)',
        injector_cls=EncoderDriftInjectorFull,
        eligible_joints=tuple(range(6)),
        distributed=False,
        severity_profile='exponential',
        implementation='full',
        reasoning='Cumulative feedback bias, phantom PID jitter, and error latching.',
    ),
    # --- Tier B: structural compact injectors (6 profiles) ---
    FaultProfile(
        profile_id='BL_COMPACT',
        fault_key='FAULT_BL',
        name='Gear Backlash (Compact)',
        injector_cls=BacklashInjectorCompact,
        eligible_joints=(0, 1, 2),
        distributed=False,
        severity_profile='exponential',
        implementation='compact',
        reasoning='Dead-zone lag and re-engagement strikes (reduced channel set).',
    ),
    FaultProfile(
        profile_id='LF_COMPACT',
        fault_key='FAULT_LF',
        name='Lubrication Failure — Distributed (Compact)',
        injector_cls=LubricationFailureInjectorCompact,
        eligible_joints=tuple(range(6)),
        distributed=True,
        severity_profile='linear',
        implementation='compact',
        reasoning='Viscous drag and thermal ramp across all six joints.',
    ),
    FaultProfile(
        profile_id='BW_COMPACT',
        fault_key='FAULT_BW',
        name='Bearing Wear (Compact)',
        injector_cls=BearingWearInjectorCompact,
        eligible_joints=(0, 1, 4, 5),
        distributed=False,
        severity_profile='polynomial',
        implementation='compact',
        reasoning='RCF velocity jitter and torque ripple on current.',
    ),
    FaultProfile(
        profile_id='EE_COMPACT',
        fault_key='FAULT_EE',
        name='Electrical Erosion (Compact)',
        injector_cls=ElectricalErosionInjectorCompact,
        eligible_joints=tuple(range(6)),
        distributed=False,
        severity_profile='abrupt',
        implementation='compact',
        reasoning='Arcing spikes and localized thermal steps.',
    ),
    FaultProfile(
        profile_id='MD_COMPACT',
        fault_key='FAULT_MD',
        name='Demagnetization — Incipient (Compact)',
        injector_cls=DemagnetizationInjectorCompact,
        eligible_joints=(1, 2),
        distributed=False,
        severity_profile='linear',
        implementation='compact',
        mode='incipient',
        reasoning='Kt decay with inverse current elevation (compact model).',
    ),
    FaultProfile(
        profile_id='ED_COMPACT',
        fault_key='FAULT_ED',
        name='Encoder Drift (Compact)',
        injector_cls=EncoderDriftInjectorCompact,
        eligible_joints=tuple(range(6)),
        distributed=False,
        severity_profile='linear',
        implementation='compact',
        reasoning='Linear measurement bias on actual joint position.',
    ),
)

PROPAGATION_FAULT_PROFILES: Tuple[FaultProfile, ...] = (
    FaultProfile(
        profile_id='SYS_COLLISION',
        fault_key='FAULT_SYS',
        name='Payload / Collision Shock',
        injector_cls=PayloadChangeInjector,
        eligible_joints=tuple(range(6)),
        distributed=True,
        severity_profile='constant',
        implementation='full',
        layer='propagation',
        reasoning='Half-sine impulse and damped oscillation on all motor currents.',
    ),
)

ALL_FAULT_PROFILES: Tuple[FaultProfile, ...] = LOCAL_FAULT_PROFILES + PROPAGATION_FAULT_PROFILES

_PROFILE_BY_ID = {p.profile_id: p for p in ALL_FAULT_PROFILES}
_PROFILE_BY_KEY = {}
for _p in ALL_FAULT_PROFILES:
    _PROFILE_BY_KEY.setdefault(_p.fault_key, []).append(_p)


def get_profile(profile_id: str) -> FaultProfile:
    if profile_id not in _PROFILE_BY_ID:
        raise KeyError(
            f"Unknown profile_id '{profile_id}'. "
            f"Valid IDs: {list(_PROFILE_BY_ID.keys())}"
        )
    return _PROFILE_BY_ID[profile_id]


def get_profiles_for_fault_key(fault_key: str) -> List[FaultProfile]:
    return list(_PROFILE_BY_KEY.get(fault_key, []))


def build_injector(
    profile_id: str,
    start_idx: int,
    end_idx: int,
    joint_index: Optional[int] = None,
    mode_override: Optional[str] = None,
) -> Any:
    return get_profile(profile_id).build(
        start_idx, end_idx, joint_index, mode_override=mode_override
    )


def iter_local_profiles() -> Sequence[FaultProfile]:
    return LOCAL_FAULT_PROFILES


def validate_registry() -> None:
    """Ensures the localized suite contains exactly 14 profiles."""
    if len(LOCAL_FAULT_PROFILES) != 14:
        raise RuntimeError(
            f"Expected 14 localized fault profiles, found {len(LOCAL_FAULT_PROFILES)}"
        )
    ids = [p.profile_id for p in LOCAL_FAULT_PROFILES]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate profile_id entries in LOCAL_FAULT_PROFILES")


validate_registry()
