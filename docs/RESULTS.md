# Method and results


## Method overview

The system localizes the car by combining image-based position fixes with a motion filter in
track coordinates. The final model is called mc7 + borders + poles with a speed-aware v_max. "mc7" means multi-camera, seven cameras; a "tick" is one 10 Hz output pose; "G1" is the earlier single-front-left-camera model with the same filter, kept as a baseline; "leave-one-lap-out" means the maps and networks scored on a lap were built without that lap.

1. **mc7 landmark matching.** Each of the seven cameras (fl, fr, cl, cr, r, rl, rr) has its own
   prebuilt map of 3D landmarks with keypoints, built from training laps 0, 1, 3 and 5. A frame
   is matched by a global descriptor (DINOv2-small) to retrieve candidate map keyframes, then
   by ALIKED keypoints and LightGlue matching, and a RANSAC perspective-n-point solve yields a
   camera pose. Fixes are taken every second tick. Each fix passes an inlier-ratio gate and is
   weighted by a noise scale fitted on training laps.
2. **Closed-loop track-border fits (every fifth tick, 2 Hz).** A segmentation network marks the
   track edges; a fit of the visible borders to the packaged track bounds gives a weak lateral
   and heading constraint. The fit starts from the filter's own current state and becomes
   available after its measured service time, so it never uses future data.
3. **Pole bearings.** Poles detected in four cameras (fl, fr, cl, cr) are associated with a
   prebuilt pole map and add bearing constraints.
4. **Telemetry-driven track-coordinate filter.** Speed and body rates from allowlisted telemetry
   propagate the pose between image fixes; the camera fixes, border fits and pole bearings are
   fused in a filter expressed in track coordinates.
5. **Speed-aware v_max.** The filter re-anchors (re-snaps its track position to a camera fix) when an implausible jump appears. The
   plausibility bound is `max(40 m/s, 1.3 x highest telemetry speed seen so far)`. It guards a
   higher-speed test domain; on all laps scored here it never acted (see below).

The first pose is applied once at the start. No learned sequence layer is included: models of
that kind were better on some held-out night laps and not on dev lap 2 or on simulated daylight,
so the submission does not use them.

## Evaluation populations

| Name | Laps | Role | Notes |
|---|---|---|---|
| Held-out | 1, 3, 5 | scored with leave-one-lap-out maps and networks | night, pooled over laps |
| Dev | 2, 4 | development and selection | night; daylight versions are simulated by relighting |
| Audit | 6, 7 | reserved, scored once | night; real recording; maps from laps 0, 1, 3, 5 only |
| Shared second recording | n/a | domain understanding only | no GT, no first pose; not used for training, tuning or validation |

All rows use the contract first pose (the ground-truth row at the start of the lap). All
metrics are per-tick pose level (one value per 10 Hz output tick), not entry or sequence level.

## Results scorecard

### Audit laps 6 and 7, night, scored once (organizer scorer per lap; pooled over 2,419 ticks by us)

| Configuration | Position RMSE (m) |
|---|---|
| Final model (mc7 + borders + poles + dynamic v_max; v_max has no effect here) | 0.420 |
| mc7 only | 0.415 |
| Single front-left camera model (G1) | 0.588 |
| Telemetry + borders + poles (no multi-camera) | 0.662 |
| Final model with a GRU layer (not submitted) | 0.489 |

Final model, further metrics: along-track 0.311 m, across-track 0.282 m, heading 0.218 degrees,
96.0 % of ticks within 1 m, maximum error 1.62 m, no tick above 3 m. Per lap the final model has
0.291 m (lap 6) and 0.502 m (lap 7); these two laps are reported together because they are one
reserved set, and the difference between the laps is as large as the difference between
configurations.

Reading. The audit does not separate mc7 alone (0.415 m) from the final model (0.420 m); both
beat the single-camera model by about 0.17 m. The borders and poles add nothing measurable at
night on this set. The dynamic v_max is identical in output to the same model without it,
because no re-anchor event occurred on these laps; this equality is by construction and not a
second confirmation. Laps 6 and 7 were chosen as the audit before the final model was fixed;
after this scoring they are no longer a clean test for anything chosen later.

The audit predictions were produced by the research runner. The same model packaged as the
release driver agreed with the research pipeline to within 5 mm RMS on dev lap 2 (3.0 to 3.5 mm
measured); the audit laps themselves were not rerun through the release driver.

### Held-out laps 1, 3, 5, night, pooled (position RMSE, leave-one-lap-out)

| Configuration | Position RMSE (m) |
|---|---|
| Single front-left camera model (G1) | 0.542 |
| mc7 | 0.334 |
| mc7 + borders + poles (final model; v_max has no effect here) | 0.326 |

Borders and poles move the pooled error by 0.008 m, inside the noise of the comparison; the gain
of mc7 over G1 is the substantial effect. The speed-aware v_max changes nothing at the recorded
speed. Dev lap 2 at night: 0.276 m for the final model (mc7 0.287 m).

### Simulated daylight, one lap per input ("relit v3" and "sunnier relit" are two settings of our offline image generator that turns night frames into daylight-like frames) (position RMSE; night RMSE of the same lap in brackets)

| Input | mc7 + borders + poles | mc7 |
|---|---|---|
| Dev lap 4, relit v3 | 0.590 (night: not scored) | 0.670 (0.276) |
| Dev lap 2, sunnier relit | 0.699 (0.276) | 1.503 (0.287) |
| Held-out lap 1, relit v3 | 1.104 (0.295) | 0.928 (0.316) |
| Dev lap 2, relit v3 | 0.697 | 0.992 |

The daylight images are produced by an image generator and are not real daylight. Each cell is
one lap and one generated relighting, so a better or worse entry is a single-lap observation.
The median position error under simulated daylight is 1.8 to 2.7 times the night value in every
cell that has a night reference, over all arms scored including arms not shown; RMSE ratios run from 0.34 to 5.2 because single transients
dominate them. Telemetry + borders + poles diverged on dev lap 4 (RMSE 81.4 m), which is why the
submission always combines telemetry with the seven-camera fixes.

### Processing time per frame

| Measurement | Hardware | Value | Scope |
|---|---|---|---|
| Release driver, run 1, shared second recording (722 ticks, 72.1 s) | one RTX 4090, shared host (load 11 to 14) | 597 ms per output tick (431 s total) | export 6 s, image probing 384 s, detection 17 s, filter 19.5 s (27 ms per tick) |
| Release driver, run 2, same input | same, busier | 752 ms per output tick (543 s total) | image probing 498 s |
| Research driver, final model on the shared second recording (722 ticks, 72.1 s) | one RTX 4090, shared host | 298 ms per output tick (total 215 s) | export, probes, detection and filter; 177 s of it is image probing; the filter alone is 25 ms per tick |
| Seven-camera matching, serial control | one A100 80 GB, shared host | 776 ms per 7-camera frame | per image stage timers; a re-implementation, not the production latency |
| Seven-camera matching, batched variant | one A100 80 GB | 568 ms per 7-camera frame | not in the final model; accuracy unchanged (0.334 m held-out) |

The release-run time on a quiet machine is not measured; the two runs above shared the GPU with other jobs. The processing
is offline in structure: image fixes arrive every second tick and the total wall time exceeds
the real time of the recording, so the program is causal in its data use but not real-time.

### Shared second recording

In the research checkout, the same frozen configuration ran to completion on it (a plumbing check;
the release repository's own two end-to-end runs are described under "Release-run parity" below): 722
finite rows at 10 Hz over 72.1 s (100 % of the expected ticks), 145 of 145 border fits valid.
There is no ground truth, so this is not a score. The first run showed that probes for cameras
cl, cr and r were wrongly refused by a calibration-fingerprint check; the check was fixed and the
configuration was re-frozen once. Nothing was tuned on this recording.

### Release-run parity

The release driver ran the shared second recording twice. Both runs gave 722 finite rows, 144 border fits used,
283 pole bearings used and no re-anchor. Against the research output (pooled position RMS, same 722
timestamps): run 1 14.0 mm, run 2 9.5 mm; median 1.3 mm, 95th percentile 18 mm; maximum 0.216 m and 0.085 m;
yaw RMS 1.5e-4 and 0.9e-4 rad. The research driver rerun on the same inputs differs from its own earlier
output by 8.4 mm RMS (median 1.3 mm, maximum 0.078 m). The cause is that each camera stream carries an
availability time equal to the frame stamp plus the measured per-frame service latency, and that time decides
when the causal filter may use the measurement; on a loaded machine the latencies differ from run to run.
Probe predictions, detections and border-fit counts are identical up to 1e-16. The release reproduces the
research driver within this run-to-run envelope (about 1 cm RMS, median about 1 mm); it does not reproduce it
bitwise, and a 5 mm RMS tolerance is not reached by either driver on a shared machine. Behaviour on the hidden
recording was not tested; on the test bag the telemetry came from the `/validation/twist` and `/accel` adapter
because `state_estimation` was absent.

## Known limitations

- The audit set is two laps from the same recording session as training; the hidden test run may
  differ in speed and lighting. Daylight evidence is simulated.
- The camera-only fallback is not implemented; a run needs one of the two telemetry sources.
- Results on different GPU types differ (RANSAC amplifies float-level differences); compare aggregate
  numbers, not rows.
- final-v2 output depends on the GPU type. On the test bag the research v2 driver on an A100 (research
  reference run) and the same code on an RTX 4090 differ by median 9.6 cm, RMS 0.55 m and a maximum of 2.9 m,
  confined to the start transient (about the first 2 s). On one GPU type the release matches the research driver
  (median 0.2 mm, RMS 8 mm). Compare aggregate metrics, not rows.
- The track bounds of the shared second recording differ from the packaged copy (cause not
  investigated); the packaged copy is used.

## Telemetry fields

The exporter writes only these 38 allowlisted fields of `/constructor0/state_estimation` (default-deny;
pose, accuracy and map-state fields are never written). The filter reads only the four in bold.

| Group | Fields |
|---|---|
| Speed | **`v_mps`**, `v_raw_mps`, **`vx_mps`**, **`vy_mps`** |
| Body rates | `wx_radps`, `wy_radps`, **`wz_radps`** |
| Accelerations | `ax_mps2`, `ay_mps2`, `az_mps2`, `ax_vel_mps2`, `ay_vel_mps2` |
| Slip and curvature | `beta_rad`, `dbeta_radps`, `kappa_radpm`, `lambda_{fl,fr,rl,rr}_perc`, `alpha_{fl,fr,rl,rr}_rad`, `diff_fr_alpha_rad` |
| Wheels and driver | `omega_w_{fl,fr,rl,rr}`, `delta_wheel_rad`, `gas`, `brake`, `gear`, `rpm`, `cba_actual_pressure_{fl,fr,rl,rr}_pa`, `valid_wheelsspeeds_b` |

When `state_estimation` is absent, the adapter derives `v_mps`, `vx_mps`, `vy_mps`, `wx_radps`, `wy_radps`, `wz_radps`, `ax_mps2`,
`ay_mps2`, `az_mps2`, `beta_rad`, `kappa_radpm` from `/constructor0/validation/twist` and `/accel`; the
filter reads the four named above.

## Provenance

Research decisions behind this page: D-074 (mc7 + borders + poles), D-083 (speed-aware v_max), D-086 (final
model without a learned layer), D-090 (audit laps 6 and 7 scored once), D-091 (configuration freeze),
D-093 (simulated daylight laps), D-094 (batched matching timing). Release-run numbers: the release repository's own test-bag runs (two runs plus a research-driver control).

## Release v2.0.0: final-v2

The default configuration is `final-v2` (research F2, D-097; chosen as the submission model in D-098): the six non-front camera streams come from the S1.22 "E" matcher (`camloc.v2.fastmatch`, one model run) instead of six per-camera probes; the front camera, borders, poles, filter and constants are those of `final-v1`. Accuracy figures for final-v2 and its comparison with final-v1 are research results (PRG-002 F2 and S1.22b) and are not re-measured here. The release only reproduces the pipeline end to end on the test bag (a pipeline check, D-084), see the section final-v2 of `docs/R02B_ADAPTATIONS.md`.


final-v2 (research D-096/D-097; one run per cell, post-hoc selection, not audited; the v2 matcher streams were
computed on an A100), position RMSE in metres, final-v2 vs the final-v1 figures above:

| Evaluation set | Lighting | final-v2 | final-v1 |
|---|---|---|---|
| Held-out laps 1, 3, 5, pooled | night | 0.291 | 0.326 |
| Held-out lap 1 | simulated daylight | 0.768 | 1.104 |
| Dev lap 4 | simulated daylight | 0.535 | 0.590 |
| Dev lap 2, sunnier variant | simulated daylight | 0.544 | 0.699 |
| Dev lap 2, v3 relit | simulated daylight | 0.505 | 0.697 |
| Dev lap 2 | night | 0.308 | 0.276 |

final-v2 was not scored on the audit laps 6 and 7.

**Measured on the test bag, final-v2** (default): one release run on a shared RTX 4090 took 392.3 s in total
(543 ms per output tick), 349.6 s of it image probing; 145 of 145 border fits were valid (`run_summary.json` of
that run). On the same GPU type the release and the research v2 driver differ by a median of 0.2 mm, RMS 8 mm and
a maximum of 7.6 cm. Identical measurement streams are not claimed for final-v2.

**Reproducibility, final-v2.** final-v2 output depends on the GPU type. On the test bag the research v2 driver on
an A100 (research reference run) and the same code on an RTX 4090 differ by median 9.6 cm, RMS 0.55 m and a
maximum of 2.9 m, confined to the start transient (about the first 2 s). On one GPU type the release matches the
research driver (median 0.2 mm, RMS 8 mm). Compare aggregate metrics, not rows.

final-v2 (default since v2.0.0): configuration `configs/final/final-v2.yaml` (release copy SHA-256 `482b9ded85f0881431b137b01dea2390ee7c52cc523591932da6136c61709a2c`; research original `699c35a609030cc8584c51b8169e825d7584a023061286fb2ad730d1cfab903f`, which differs by the `$ASSETS/` path prefix); research decisions D-096 and D-097. The figures in the Results section and the timings above are labelled final-v1 or final-v2 accordingly.
