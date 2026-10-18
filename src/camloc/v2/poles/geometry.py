"""Bearing model of vertical 3D lines (GT-free).

Convention: the car base frame is taken as level (roll = pitch = 0; GT median |roll| 1.3 deg, |pitch| 0.3 deg, both ignored in
the builder and online alike, so map and measurement share one model). A vertical line at map (X, Y) seen by camera c projects to
an image line whose every pixel back-projects to the same horizontal azimuth. The measurement is that azimuth in the BASE frame

    beta = atan2(Y - c_y, X - c_x) - yaw,    c = (x, y) + Rz(yaw) t_bc[:2]

with t_bc the camera centre in the base frame. Angles in radians, wrapped to (-pi, pi].
"""
from __future__ import annotations

import math

import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def cam_center(x, y, yaw, t_bc):
    """Camera centre in the map plane (level car). Vectorised over x, y, yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(x) + c * t_bc[0] - s * t_bc[1], np.asarray(y) + s * t_bc[0] + c * t_bc[1]


def predict_bearing(x, y, yaw, t_bc, X, Y):
    """Base-frame azimuth (rad) and range (m, horizontal, from the camera centre) of landmarks (X, Y)."""
    cx, cy = cam_center(x, y, yaw, t_bc)
    dx, dy = np.asarray(X) - cx, np.asarray(Y) - cy
    return wrap(np.arctan2(dy, dx) - yaw), np.hypot(dx, dy)


def bearing_jacobian(x, y, yaw, t_bc, X, Y):
    """d beta / d (x, y, yaw) for one landmark (analytic). Returns np.array(3)."""
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = x + c * t_bc[0] - s * t_bc[1], y + s * t_bc[0] + c * t_bc[1]
    dx, dy = X - cx, Y - cy
    r2 = dx * dx + dy * dy
    dcx = -s * t_bc[0] - c * t_bc[1]
    dcy = c * t_bc[0] - s * t_bc[1]
    return np.array([dy / r2, -dx / r2, -1.0 + (dy * dcx - dx * dcy) / r2])


def map_azimuth(x, y, yaw, t_bc, beta_base):
    """(camera centre x, y, map azimuth) of a base-frame bearing observed at pose (x, y, yaw)."""
    cx, cy = cam_center(x, y, yaw, t_bc)
    return cx, cy, wrap(np.asarray(beta_base) + np.asarray(yaw))
