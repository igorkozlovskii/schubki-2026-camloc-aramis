"""Parameters, tiers and config loading for the rework filter2 (PRG-002 S2.1; copy of S1.3 `filter/params.py` with the S2.1 consensus / lost-mode / likelihood parameters).

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
    use_coarse_update: bool = True   # S2.1: tuned soft update from the coarse place (G1 does this); False = S1.3 behaviour
    use_border: bool = True
    latency_comp: bool = True     # shift a measurement stamped before the tick by the filter's own path length since its stamp
    # ---- innovation gate (chi-square, declared): 3 dof p=0.001 -> 16.27; Huber inflation above gate_soft (3 dof p=0.05 -> 7.81); 2 dof p=0.001 -> 13.82
    gate_hard: float = 16.27
    gate_soft: float = 7.81
    coarse_gate: float = 11.34    # 3 dof p=0.01
    border_gate: float = 13.82
    # ---- lateral regularisation (S1.8 finding: gains attributed to border rows were n-variance regularisation): weak prior on n at its own value, no measurement involved.
    # After every tick, if P_nn > n_prior_sig^2 a pseudo-observation z = n_hat with sigma n_prior_sig shrinks it (mean unchanged, cross-covariances kept). 0 = off.
    n_prior_sig: float = 0.0
    # ---- fix likelihood (S2.1): `student` = Student-t (nu = t_nu) via R_eff = R * max(1, (nu + d2) / (nu + 3)); `huber` = S1.3 inflation above gate_soft
    lik: str = "student"
    t_nu: float = 4.0
    fix_w_rms: float = 0.0        # sigma inflation per unit of (q_rms_px / fix_rms_ref - 1)+ (per-fix sigma from quality fields)
    fix_rms_ref: float = 1.5
    coarse_evidence: bool = True  # a coarse place far from the prediction (d2 > coarse_lost_gate) counts as rejected-fix evidence for lost-mode entry (never as a correction)
    coarse_lost_gate: float = 25.0
    coarse_w_gate: float = 1.0    # coarse soft update: R is the coarse sigma times this
    # ---- evidence for a large correction (D-013). Independence: a fix and a coarse place from one camera image are ONE source; units are cameras, or repeated
    # fixes of one camera whose positions follow a constant-velocity fit on their OWN stamps (speed in [cv_v_lo, cv_v_hi], residual <= cv_res_m).
    cons_window_s: float = 5.0    # healthy state: how long rejected fixes are kept
    cons_tol_n: float = 1.5       # lateral spread inside an agreeing set (m)
    cons_tol_psi: float = 0.1     # heading spread (rad)
    cons_tol_s: float = 3.0       # along-track residual of the constant-velocity fit / pairwise cross-camera tolerance (m)
    cv_v_lo: float = 3.0
    cv_v_hi: float = 90.0
    rep_k: int = 6                # repeated single-camera fixes needed (distinct stamps)
    rep_span_s: float = 2.0       # ... spanning at least this long
    fresh_k: int = 4              # same, while the state is unsupported (fresh): hard-reset path (F3)
    fresh_span_s: float = 0.7
    fresh_n_acc: int = 10         # accepted fixes since start / adoption / exit below which the state counts as fresh
    cons_min_cams: int = 2        # healthy-state adoption: cameras (independent units) in the agreeing set
    coarse_n_min: int = 3         # coarse places (distinct images) needed to rate a cluster; their mean has sigma coarse_sig / sqrt(n)
    coarse_dchi: float = 4.0      # coarse support / veto threshold on sum (d_pred^2 - d_cluster^2) / sigma^2
    coarse_sep_k: float = 2.0     # a coarse mean discriminates only if cluster and prediction are further apart than coarse_sep_k * its sigma
    # ---- second camera on demand (S2.1 / coordinator note): rows of camera `on_demand_cam` are consumed only when the primary camera has no fix at that stamp (+-60 ms) or its fix
    # was gate-rejected, or in lost mode; the skipped rows model 'fr not run'. '' = consume every row.
    on_demand_cam: str = ""
    primary_cam: str = "fl"
    cam2_sig_scale: float = 1.0   # sigma multiplier for landmark fixes of camera on_demand_cam (the coarser second camera)
    # ---- S2.1r switches (all default to the S2.1 behaviour, bit for bit)
    coarse_lost_update: bool = False   # coarse place as a soft update ONLY while lost (search prior), R x coarse_lost_r; independent of use_coarse_update
    coarse_lost_r: float = 1.0
    nis_k: int = 0                     # covariance-aware telemetry-bias entry: mean chi-square of the last nis_k coarse places (3 dof, coarse sigma) >= 3 * nis_thr -> lost. 0 = off
    nis_thr: float = 3.0
    cal_scale: float = 1.0             # output-only inflation of the predicted error and the reported covariance (calibration); the state covariance is untouched
    # ---- lost mode (S2.1): entered on rejected-fix evidence (lost_after_s, or lost_after_fresh_s while fresh) or a declared long gap (no accepted fix for
    # lost_gap_s AND predicted sigma above lost_sigma_m). While lost: wide-gated, covariance-inflated Kalman updates (R x lost_r_infl); exit by soft update.
    lost_after_s: float = 8.0
    lost_after_fresh_s: float = 2.0
    lost_gap_s: float = 15.0
    lost_sigma_m: float = 3.0
    lost_gate: float = 30.0
    lost_r_infl: float = 4.0
    lost_window_s: float = 4.0
    lost_widen_s: float = 6.0
    lost_widen_n: float = 1.5
    lost_widen_psi: float = 0.08
    exit_r_div: float = 2.0       # exit update: R / min(n_fixes / exit_r_div, 4) (>= 1)
    exit_sig_v: float = 1.5       # T0: sigma of the speed obtained from the fit (m/s) before its own fit error is added

    def __post_init__(self):
        if self.lik not in ("student", "huber"):
            raise ValueError(f"lik must be student|huber, got {self.lik!r}")
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
