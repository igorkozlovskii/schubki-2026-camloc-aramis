"""PRG-002 S2.5: sparse geometric landmark map of vertical structures (poles, posts, sign edges) and bearing measurements.

Modules
  detect     GT-free near-vertical line-segment detector (CLAHE + LSD, vanishing-point test) -> per-segment base-frame bearings.
  geometry   bearing model of a vertical 3D line seen from a level car: prediction, Jacobian w.r.t. (x, y, yaw).
  associate  prediction-gated, ambiguity-rejecting association of detections to map landmarks (GT-free).
  mapbuild   offline 2D multi-view triangulation of vertical lines from posed bearings (poses are passed in by the builder script;
             this module never loads ground truth itself).

Nothing in this package reads ground truth, `sn_map_state`, `derived/refline_*` or `app/camloc.py::load_all()`.
"""
