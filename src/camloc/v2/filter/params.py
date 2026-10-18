"""Parameters, tiers and config loading for the track-coordinate filter (PRG-002 S1.3).

Telemetry tiers (D-042). Fields are names of `camloc.io.telemetry.ALLOWED_FIELDS` (contract `inputs-v2`, telemetry-v1); everything else is default-denied
by the store. Forbidden for every tier: pose fields, `yaw_vel_rad`, `sn_map_state`, accuracy/status fields, GT.
  T2  vx_mps, vy_mps (body velocity -> slip-aware course), wz_radps (yaw rate), ax_mps2 (speed hold extrapolation)
  T1  v_mps (speed), wz_radps (yaw rate), ax_mps2 (speed hold extrapolation). Optional experiment t1_slip (off; negative result, see REPORT): ay_mps2 drives an estimated
      lateral velocity, d(vy)/dt = ay - omega vx.
  T0  none: constant-velocity track-following default motion (speed is a state), camera measurements only
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

TIER_FIELDS: dict[str, tuple[str, ...]] = {
    "T2": ("vx_mps", "vy_mps", "wz_radps", "ax_mps2"),
    "T1": ("v_mps", "wz_radps", "ax_mps2"),
    "T0": (),
}


@dataclass
class FilterParams:
    tier: str = "T2"
    # ---- nuisance constants (S10a DRIFT.md, train-fitted) and clamps
    a0: float = 1.0039          # speed scale
    a_w: float = 1.0052         # yaw-rate scale
    b0: float = 3.3e-5          # yaw-rate bias, rad/s
    a_lo: float = 0.99
    a_hi: float = 1.01
    b_lim: float = 5e-4
    # ---- process noise (per sqrt second)
    q_s: float = 0.07
    q_n: float = 0.07
    q_psi: float = 0.005
    q_v: float = 1.5            # speed random walk where speed is a state (T0, telemetry fallback), m/s/sqrt(s)
    q_a: float = 3.8e-4
    q_b: float = 1e-6
    t1_slip: bool = False       # T1 experiment (negative result, default off): estimate the lateral velocity from ay_mps2; adds the field ay_mps2 to the tier
    q_vy: float = 0.5           # T1: lateral (slip) velocity random walk on top of the ay-driven prediction, m/s/sqrt(s)
    vy_tau_s: float = 8.0       # T1: leak time constant of the estimated lateral velocity (ay bias would otherwise integrate without bound)
    init_sig_vy: float = 0.3    # T1: initial sigma of the lateral velocity, m/s
    # ---- initial uncertainty (first pose is organizer-provided)
    init_sig_s: float = 0.5
    init_sig_n: float = 0.5
    init_sig_psi: float = 0.02
    init_sig_v: float = 10.0    # T0 only
    v0: float = 25.0            # T0 only
    init_sig_a: float = 1.5e-3
    init_sig_b: float = 1e-4
    # ---- telemetry
    fallback_age_s: float = 1.0   # latest telemetry row older than this: T0-style motion for the stretch
    ax_extrap_s: float = 0.02     # speed hold extrapolation horizon with ax (0 = off)
    # ---- measurements (defaults used when the meas-v1 row declares no sigma)
    fix_sig_s: float = 1.2
    fix_sig_n: float = 1.2
    fix_sig_psi: float = 0.02
    fix_w_inl: float = 0.0
    fix_n_ref: float = 300.0
    coarse_sig_s: float = 10.0
    coarse_sig_n: float = 2.5
    coarse_sig_psi: float = 0.06
    border_sig_n: float = 0.3
    border_sig_psi: float = 0.01
    use_coarse_update: bool = False
    use_border: bool = True
    latency_comp: bool = True     # shift a measurement stamped before the tick by the filter's own path length since its stamp
    # ---- innovation gate (chi-square, declared): 3 dof p=0.001 -> 16.27; Huber inflation above gate_soft (3 dof p=0.05 -> 7.81); 2 dof p=0.001 -> 13.82
    gate_hard: float = 16.27
    gate_soft: float = 7.81
    coarse_gate: float = 11.34    # 3 dof p=0.01
    border_gate: float = 13.82
    # ---- consensus rule (healthy state, innovation beyond the gate)
    cons_window_s: float = 5.0
    cons_tol_s: float = 3.0
    cons_tol_n: float = 1.5
    cons_tol_psi: float = 0.1
    cons_min_fixes: int = 2       # gate-rejected fixes that agree with each other (carried by the filter's own path length)
    cons_min_sources: int = 2     # distinct sources (camera, kind) in the agreeing set; a single repeating source never counts twice
    coarse_tol_s: float = 20.0    # coarse place (~8 m accurate) tolerance: normalised distance D = hypot(ds / tol_s, dn / tol_n)
    coarse_tol_n: float = 6.0
    coarse_ratio: float = 2.0     # healthy state: a coarse place supports a cluster if D(coarse, cluster) <= 1 and D(coarse, prediction) >= ratio * D(coarse, cluster); it vetoes in the opposite case
    # ---- lost mode
    lost_after_s: float = 8.0     # gate-rejected fixes for this long without an accepted measurement
    lost_after_fresh_s: float = 2.0    # same, while the state is not yet supported (fewer than fresh_n_acc accepted fixes since start / adoption / lost exit; S10g lesson)
    fresh_n_acc: int = 10
    lost_sigma_m: float = 3.0     # or predicted position sigma above this
    lost_k: int = 3               # agreeing fixes (any single source) to leave lost mode
    lost_window_s: float = 3.0
    lost_coarse_veto_s: float = 10.0   # coarse place may veto an exit only during the first seconds of the lost episode
    lost_widen_s: float = 20.0
    lost_widen_n: float = 3.0
    lost_widen_psi: float = 0.15

    def __post_init__(self):
        if self.tier not in TIER_FIELDS:
            raise ValueError(f"tier must be one of {tuple(TIER_FIELDS)}, got {self.tier!r}")

    @property
    def fields(self) -> tuple[str, ...]:
        return TIER_FIELDS[self.tier] + (("ay_mps2",) if (self.tier == "T1" and self.t1_slip) else ())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FilterParams":
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown filter parameters: {sorted(bad)}")
        return cls(**d)


def load_config(path) -> tuple[FilterParams, dict]:
    """configs/v2/filter/*.yaml -> (FilterParams, raw config). The listed telemetry `fields` must equal the tier's field set."""
    cfg = yaml.safe_load(Path(path).read_text())
    p = FilterParams.from_dict({"tier": cfg["tier"], **(cfg.get("params") or {})})
    if "fields" in cfg and tuple(cfg["fields"]) != p.fields:
        raise ValueError(f"config lists fields {cfg['fields']} but tier {p.tier} reads {list(p.fields)}")
    return p, cfg
