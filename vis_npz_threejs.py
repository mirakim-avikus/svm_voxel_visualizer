"""
Three.js viewer for gen_semantic_occupancy NPZ files.

Preprocesses a set of NPZ files (200x200x16 `semantics` uint8) into primitives
(cuboids for boats/docks/buildings/persons, cylinders for poles, plane for
water), writes frames.json + index.html, and serves them on localhost.

Usage:
    python vis_npz_threejs.py <npz_or_dir> [...] [--port 8765]
"""

import argparse
import functools
import http.server
import json
import multiprocessing as mp
import os
import re
import socketserver
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_erosion, distance_transform_edt, label as ndlabel,
)

NX, NY, NZ = 200, 200, 16
VOXEL = 0.4
CONN = np.ones((3, 3, 3), dtype=bool)
POLE_RADIUS = 0.375  # thin cylinders, in meters (1.5x of original 0.25)

# The viewer expects gen_semantic_occupancy's scheme
#   (1=pole 2=dock 3=person 4=boat 5=building 6=water).
# occupancy_pred dumps use a different scheme:
#   0=pole 1=dock 2=boat 3=water 4=free 255=ignore.
# We remap on load so the rest of the pipeline stays unchanged. Auto-applied
# when the input has values outside the viewer's scheme.
CLASS_REMAP_OCC_PRED = {
    0: 1, 1: 2, 2: 4, 3: 6, 4: 0, 255: 0,
}

# BEV-footprint straightening: morphological close kernel + Douglas-Peucker
# epsilon, applied per class. Units are voxels (0.4 m). Larger epsilon =
# straighter but coarser edges.
DOCK_CLOSE_KSIZE = 3
DOCK_DP_EPSILON_VOX = 1.2
DOCK_HEIGHT_SCALE = 1.0 / 3.0  # shrink extruded dock prism toward its base
BOAT_CLOSE_KSIZE = 3
BOAT_DP_EPSILON_VOX = 1.2

# Boat instance splitting (BEV, voxel units). The pipeline is:
#   opening -> distance transform -> peak detection -> Voronoi reassign.
# Peak detection is what actually separates touching boats; the knobs below
# each make it more aggressive.
#
#   OPEN_VOX         : morphological-open radius applied before DT. Bigger
#                       kernel kills wider bridges (full edge-to-edge touches).
#                       0 disables pre-opening.
#   PEAK_SEP_VOX     : peaks closer than (2*sep+1) vox in a dilation window
#                       collapse into one seed. 0 disables dilation gating
#                       (every pixel >= MIN_PEAK_DT becomes a seed).
#   MIN_PEAK_DT_VOX  : peaks must be at least this far from the nearest
#                       background pixel (i.e. skinny ridges get rejected).
#                       Raise this to reject saddle-region fake peaks.
BOAT_SPLIT_OPEN_VOX = 2
BOAT_SPLIT_PEAK_SEP_VOX = 2
BOAT_SPLIT_MIN_PEAK_DT_VOX = 2

# Ellipsoid fit for boats (alternate rendering mode). PCA-based with boat
# aspect-ratio constraints; rejects dock-adjacent / vertical / tiny clusters.
# Sizes are in meters and converted to voxel counts internally.
# Faithful port of the reference open3d viewer (boat_ellipsoid_overlay).
# All length thresholds are in VOXEL units (the grid is 200x200x16, voxel
# size 0.4m), matching the reference's VOXEL_SIZE=[1,1,1] convention.
ELL_MIN_VOXELS = 1000
ELL_RATIO_MIN = 4.0          # major:minor in [RATIO_MIN, RATIO_MAX]
ELL_RATIO_MAX = 6.0
ELL_MIN_SEMI_MAJOR = 7.0     # voxels
ELL_MAX_SEMI_MAJOR = 30.0    # voxels
ELL_MAX_TILT_DEG = 10.0
ELL_DOCK_OVERLAP = 0.3
ELL_ARROW_LENGTH_RATIO = 3.5  # bow-arrow length = rx * this (used in JS)
ELL_MEDIAN_DIR_TOL_DEG = 20.0  # drop boats whose bow_dir > this from median

THREE_VERSION = "0.162.0"
VENDOR_FILES = {
    "vendor/three.module.js":
        f"https://unpkg.com/three@{THREE_VERSION}/build/three.module.js",
    "vendor/addons/controls/OrbitControls.js":
        f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js",
    "vendor/addons/loaders/GLTFLoader.js":
        f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/loaders/GLTFLoader.js",
    "vendor/addons/utils/BufferGeometryUtils.js":
        f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/utils/BufferGeometryUtils.js",
    "vendor/addons/geometries/ConvexGeometry.js":
        f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/geometries/ConvexGeometry.js",
    "vendor/addons/math/ConvexHull.js":
        f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/math/ConvexHull.js",
}

# Optional GLB boat asset. If present, copied into the viewer build so the
# HTML can fetch it at ./3dboat.glb. Model convention: origin at boat bottom
# center, +X = head (forward). Scale/yaw per boat are computed from each
# OBB at runtime.
GLB_BOAT_SRC = Path.home() / "workspace" / "3dboat.glb"


def vendor_three(out_dir):
    for rel, url in VENDOR_FILES.items():
        dst = out_dir / rel
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  vendoring {rel}")
        urllib.request.urlretrieve(url, dst)


def _components(mask):
    lbl, n = ndlabel(mask, structure=CONN)
    for i in range(1, n + 1):
        coords = np.argwhere(lbl == i)
        if len(coords):
            yield coords


def _voxel_edge(v):
    """Voxel-index coord (may be fractional edge) -> world meters, grid centered."""
    return (
        (v[0] - NX / 2) * VOXEL,
        (v[1] - NY / 2) * VOXEL,
        (v[2] - NZ / 2) * VOXEL,
    )


def _cuboid_from_coords(coords):
    lo = coords.min(0)
    hi = coords.max(0) + 1  # exclusive edge
    lx, ly, lz = _voxel_edge(lo)
    hx, hy, hz = _voxel_edge(hi)
    return {
        "cx": (lx + hx) / 2, "cy": (ly + hy) / 2, "cz": (lz + hz) / 2,
        "sx": hx - lx, "sy": hy - ly, "sz": hz - lz,
    }


def _prism_from_coords(coords, close_k, dp_eps, height_scale=1.0):
    """Straightened extruded prism from a CC.

    BEV footprint: morphological close -> outer contour -> Douglas-Peucker.
    Height: voxel Z extent of the component, scaled by `height_scale` (kept
    anchored at the base z0).
    """
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    z0 = float((zs.min() - NZ / 2) * VOXEL)
    z1 = float((zs.max() + 1 - NZ / 2) * VOXEL)
    if height_scale != 1.0:
        z1 = z0 + (z1 - z0) * height_scale

    pad = max(2, close_k)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    h = (x_max - x_min + 1) + 2 * pad
    w = (y_max - y_min + 1) + 2 * pad
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[xs - x_min + pad, ys - y_min + pad] = 255

    if close_k > 1:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, dp_eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None

    # cv2 contour points are (col, row); our mask rows = voxel-x axis.
    poly = []
    for col, row in pts.astype(float):
        ix = row + x_min - pad
        iy = col + y_min - pad
        wx = (ix - NX / 2 + 0.5) * VOXEL
        wy = (iy - NY / 2 + 0.5) * VOXEL
        poly.append([float(wx), float(wy)])
    return {"polygon": poly, "z0": z0, "z1": z1}


def _bev_instance_labels(bev, open_vox, peak_sep, min_peak_dt):
    """Watershed-style BEV label with aggressive peak detection.

    Pipeline:
      1. Morphological open (radius = open_vox) on `bev`. Kills thin/wide
         bridges outright so they vanish before DT is computed.
      2. cv2.distanceTransform on the opened mask.
      3. Peaks = pixels where dt equals local max in a (2*peak_sep+1) window
         AND dt >= min_peak_dt. Raising min_peak_dt rejects shallow
         saddle-region peaks.
      4. cv2.connectedComponents(peaks) -> one seed per cluster.
      5. Voronoi: reassign the ORIGINAL bev pixels (not the opened ones)
         to the nearest seed, so each instance keeps its full footprint.

    bev: uint8, 255 inside / 0 outside. Returns int32 labels 0..K.
    """
    opened = bev
    if open_vox > 0:
        k = int(2 * open_vox + 1)
        opened = cv2.morphologyEx(
            bev, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if not opened.any():
        _, lbl = cv2.connectedComponents(bev)
        return lbl

    dt = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    if peak_sep > 0:
        k = int(2 * peak_sep + 1)
        dilated = cv2.dilate(dt, np.ones((k, k), np.uint8))
        peaks = (dt == dilated) & (dt >= min_peak_dt)
    else:
        peaks = dt >= min_peak_dt
    peaks = peaks.astype(np.uint8)

    n_seeds, seed_lbl = cv2.connectedComponents(peaks)
    if n_seeds <= 2:
        _, lbl = cv2.connectedComponents(bev)
        return lbl

    _, (ri, ci) = distance_transform_edt(seed_lbl == 0, return_indices=True)
    nearest = seed_lbl[ri, ci]
    return np.where(bev > 0, nearest, 0).astype(np.int32)


def _boat_instances(boat_mask_3d):
    """Yield per-instance 3D voxel coords after BEV split."""
    bev = boat_mask_3d.any(axis=2).astype(np.uint8) * 255
    lbl2d = _bev_instance_labels(
        bev, BOAT_SPLIT_OPEN_VOX,
        BOAT_SPLIT_PEAK_SEP_VOX, BOAT_SPLIT_MIN_PEAK_DT_VOX)
    for i in range(1, int(lbl2d.max()) + 1):
        pix = lbl2d == i
        if not pix.any():
            continue
        sub_3d = boat_mask_3d & pix[:, :, None]
        coords = np.argwhere(sub_3d)
        if len(coords):
            yield coords


def _obb_from_coords(coords):
    """Oriented cuboid for a boat instance via cv2.minAreaRect on BEV footprint.

    Returns {cx, cy, cz, sx, sy, sz, yaw} in world meters / radians, with yaw
    rotating the box around +Z.
    """
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (bx, by), (bw, bh), angle_deg = cv2.minAreaRect(pts)
    wcx = (bx - NX / 2 + 0.5) * VOXEL
    wcy = (by - NY / 2 + 0.5) * VOXEL
    # +1 voxel so the box covers the full voxel extent (bw,bh are in voxel
    # units, measured between voxel-center points).
    wsx = (bw + 1) * VOXEL
    wsy = (bh + 1) * VOXEL
    z0 = float((zs.min() - NZ / 2) * VOXEL)
    z1 = float((zs.max() + 1 - NZ / 2) * VOXEL)
    return {
        "cx": float(wcx), "cy": float(wcy), "cz": (z0 + z1) / 2,
        "sx": float(wsx), "sy": float(wsy), "sz": z1 - z0,
        "yaw": float(np.deg2rad(angle_deg)),
    }


def _fit_ellipsoid_params(coords):
    """3D PCA fit per reference. Major from PCA; minor/height from PCA but
    clamped so major:minor and major:height stay in [RATIO_MIN, RATIO_MAX].
    Returns (center_voxel, axes_3x3, radii_voxel), axes columns are the
    eigenvectors of the 3D covariance (largest first).
    """
    center = coords.mean(axis=0)
    centered = coords - center

    if coords.shape[0] < 4:
        ratio = (ELL_RATIO_MIN + ELL_RATIO_MAX) / 2
        return center, np.eye(3), np.array([
            ELL_MIN_SEMI_MAJOR,
            ELL_MIN_SEMI_MAJOR / ratio,
            ELL_MIN_SEMI_MAJOR / ratio,
        ])

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # eigh returns orthonormal vecs but not necessarily a right-handed basis.
    # Three.js' Quaternion.setFromRotationMatrix only handles proper rotations
    # (det=+1); a reflection (det=-1) collapses to a 180-deg rotation that
    # SWAPS two columns, so the rendered ellipsoid's major axis ends up
    # orthogonal to v1. Flip the smallest-variance vector to fix the chirality
    # without disturbing v1 / v2.
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] = -eigvecs[:, 2]

    semi_major = np.clip(np.sqrt(max(eigvals[0], 0.01)),
                         ELL_MIN_SEMI_MAJOR, ELL_MAX_SEMI_MAJOR)
    raw = np.sqrt(np.maximum(eigvals, 0.01))
    r1 = np.clip(semi_major / max(raw[1], 0.1),
                 ELL_RATIO_MIN, ELL_RATIO_MAX)
    r2 = np.clip(semi_major / max(raw[2], 0.1),
                 ELL_RATIO_MIN, ELL_RATIO_MAX)
    radii = np.array([semi_major, semi_major / r1, semi_major / r2])
    return center, eigvecs, radii


def _circular_dist(a, b):
    """Smallest unsigned angular gap between two directions, in radians."""
    diff = abs(a - b) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)


def _circular_median(angles):
    """Direction (radians) that minimises the sum of circular distances to
    all input directions. O(N^2) but N is small (max ~25 boats per frame)."""
    if not angles:
        return 0.0
    best_a, best_score = angles[0], float("inf")
    for a in angles:
        score = sum(_circular_dist(a, b) for b in angles)
        if score < best_score:
            best_a, best_score = a, score
    return best_a


def _detect_bow_dir(coords, center, axes):
    """Bow = end of the major axis with the narrower cross-section."""
    major = axes[:, 0]
    centered = coords - center
    proj = centered @ major
    p_min, p_max = proj.min(), proj.max()
    span = p_max - p_min
    if span < 2:
        return major
    tip = 0.25
    pos_m = proj > (p_max - span * tip)
    neg_m = proj < (p_min + span * tip)
    perp = centered - np.outer(proj, major)
    perp_d = np.linalg.norm(perp, axis=1)
    pos_s = perp_d[pos_m].mean() if pos_m.any() else 0
    neg_s = perp_d[neg_m].mean() if neg_m.any() else 0
    return major if pos_s < neg_s else -major


def _extract_boat_ellipsoids(sem):
    """Faithful port of reference extract_boat_ellipsoids.

    Pipeline: XY-erode (1 iter) -> 3D 6-conn CC -> per-CC filters
    (size, z/xy span, dock-adjacency, per-axis tilt) -> 3D PCA fit ->
    bow detection -> greedy overlap pruning.

    Emits world-meter cx/cy/cz, rx/ry/rz, a row-major 9-float `axes`
    = [v1,v2,v3] (each column), and bow_dir unit vector.
    """
    boat_mask = sem == 4
    struct_xy = np.zeros((3, 3, 3), dtype=bool)
    struct_xy[:, :, 1] = True
    boat_mask = binary_erosion(boat_mask, structure=struct_xy, iterations=1)
    if not boat_mask.any():
        return []

    # 6-conn (face neighbors) per reference's generate_binary_structure(3, 1)
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    struct6[1, 1, :] = True
    struct6[1, :, 1] = True
    struct6[:, 1, 1] = True
    cc, n = ndlabel(boat_mask, structure=struct6)

    # BEV (top-down) dock footprint: any-Z collapse. A boat is dropped if
    # more than ELL_DOCK_OVERLAP fraction of its BEV pixels fall on dock.
    dock_bev = (sem == 2).any(axis=2)

    cands = []
    for k in range(1, n + 1):
        coords = np.argwhere(cc == k).astype(np.float64)
        if coords.shape[0] < ELL_MIN_VOXELS:
            continue
        span = coords.max(axis=0) - coords.min(axis=0) + 1
        xy_span = max(span[0], span[1])
        if xy_span > 0 and span[2] / xy_span > 0.5:
            continue
        ci = coords.astype(int)
        boat_xy = np.unique(ci[:, :2], axis=0)
        n_overlap = np.count_nonzero(dock_bev[boat_xy[:, 0], boat_xy[:, 1]])
        if n_overlap / boat_xy.shape[0] > ELL_DOCK_OVERLAP:
            continue

        center, axes, radii = _fit_ellipsoid_params(coords)

        skip = False
        for ax_i in range(3):
            tilt = np.degrees(np.arcsin(np.clip(abs(axes[2, ax_i]), 0, 1)))
            if tilt > ELL_MAX_TILT_DEG and radii[ax_i] > radii.min() * 1.5:
                skip = True
                break
        if skip:
            continue

        bow = _detect_bow_dir(coords, center, axes)

        cx = float((center[0] - NX / 2 + 0.5) * VOXEL)
        cy = float((center[1] - NY / 2 + 0.5) * VOXEL)
        cz = float((center[2] - NZ / 2 + 0.5) * VOXEL)
        rx, ry, rz = (radii * VOXEL).tolist()
        cands.append({
            "cx": cx, "cy": cy, "cz": cz,
            "rx": float(rx), "ry": float(ry), "rz": float(rz),
            "axes": [float(v) for v in axes.T.flatten()],
            "bow_dir": [float(bow[0]), float(bow[1]), float(bow[2])],
            "n": int(coords.shape[0]),
        })

    # Greedy overlap pruning: largest first; drop if center-distance is
    # within 0.8 * (rx + kept_rx) of an already-kept ellipsoid.
    cands.sort(key=lambda e: e["n"], reverse=True)
    kept = []
    for e in cands:
        dup = False
        for kp in kept:
            d = ((e["cx"] - kp["cx"]) ** 2
                 + (e["cy"] - kp["cy"]) ** 2
                 + (e["cz"] - kp["cz"]) ** 2) ** 0.5
            if d < (e["rx"] + kp["rx"]) * 0.8:
                dup = True
                break
        if not dup:
            kept.append(e)

    # Direction-outlier reject in doubled-angle (axis) space:
    #  - Doubling collapses bow/stern flips so a 180-deg ambiguity doesn't
    #    poison the estimator.
    #  - Circular MEAN of doubled angles is continuous in the input, so a
    #    boat appearing or shifting slightly nudges the reference smoothly
    #    instead of snapping it to another cluster (which is what made
    #    whole groups flicker on/off between frames).
    # A 20-deg threshold on axes corresponds to 40 deg in doubled-angle
    # space.
    if len(kept) >= 3:
        doubled = np.array([
            2 * np.arctan2(e["bow_dir"][1], e["bow_dir"][0]) for e in kept
        ])
        cs, sn = np.mean(np.cos(doubled)), np.mean(np.sin(doubled))
        # Bail if the doubled-angle vectors cancel out (boats truly random):
        # leave kept unchanged so the filter doesn't reject everything.
        if cs * cs + sn * sn > 1e-6:
            mean_doubled = np.arctan2(sn, cs)
            thresh = 2 * np.deg2rad(ELL_MEDIAN_DIR_TOL_DEG)
            kept = [e for e, d in zip(kept, doubled)
                    if _circular_dist(d, mean_doubled) <= thresh]
    return kept


def _cylinder_from_coords(coords):
    cx = float((coords[:, 0].mean() - NX / 2 + 0.5) * VOXEL)
    cy = float((coords[:, 1].mean() - NY / 2 + 0.5) * VOXEL)
    z0 = float((coords[:, 2].min() - NZ / 2) * VOXEL)
    z1 = float((coords[:, 2].max() + 1 - NZ / 2) * VOXEL)
    return {"cx": cx, "cy": cy, "z0": z0, "z1": z1, "r": POLE_RADIUS}


def _boat_surface_voxels(boat_mask):
    """Return (N,3) int16 coords of boat voxels that have any empty 6-neighbor.

    Interior voxels aren't visible anyway; dropping them shrinks the payload
    ~10-20x for dense hulls.
    """
    # 6-connectivity structuring element (face neighbors only)
    struct6 = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ], dtype=bool)
    eroded = binary_erosion(boat_mask, structure=struct6, border_value=0)
    surface = boat_mask & ~eroded
    return np.argwhere(surface).astype(np.int16)


def _extract_from_sem(sem):
    """Build primitives + (boat, water) surface voxels from a semantics array."""
    out = {"boats": [], "ellipsoids": [], "docks": [], "buildings": [],
           "persons": [], "poles": [], "water_z": None}

    for cid, key in [(5, "buildings"), (3, "persons")]:
        mask = sem == cid
        if not mask.any():
            continue
        for coords in _components(mask):
            out[key].append(_cuboid_from_coords(coords))

    dock_mask = sem == 2
    if dock_mask.any():
        for coords in _components(dock_mask):
            prism = _prism_from_coords(
                coords, DOCK_CLOSE_KSIZE, DOCK_DP_EPSILON_VOX,
                height_scale=DOCK_HEIGHT_SCALE)
            if prism is not None:
                out["docks"].append(prism)

    boat_mask = sem == 4
    boat_surface = None
    if boat_mask.any():
        for coords in _boat_instances(boat_mask):
            out["boats"].append(_obb_from_coords(coords))
        boat_surface = _boat_surface_voxels(boat_mask)
        out["ellipsoids"] = _extract_boat_ellipsoids(sem)

    pole_mask = sem == 1
    if pole_mask.any():
        for coords in _components(pole_mask):
            out["poles"].append(_cylinder_from_coords(coords))
        # Keep the longer poles: drop anything shorter than the mean height.
        # Short stubs are usually noise; the actual mast/antenna structures
        # we want to visualise cluster at or above the mean.
        if out["poles"]:
            heights = np.array([p["z1"] - p["z0"] for p in out["poles"]])
            mean_h = float(heights.mean())
            out["poles"] = [p for p in out["poles"]
                            if (p["z1"] - p["z0"]) >= mean_h]

    water_mask = sem == 6
    water_surface = None
    if water_mask.any():
        z_med = float(np.median(np.argwhere(water_mask)[:, 2]))
        out["water_z"] = (z_med - NZ / 2 + 0.5) * VOXEL
        # Only keep voxels with at least one empty 6-neighbor — the inside
        # of the water mass is never visible, so dropping it shrinks the
        # per-frame payload massively while keeping the visible silhouette.
        water_surface = _boat_surface_voxels(water_mask)
    return out, boat_surface, water_surface


def extract_frame(npz_path):
    """Return (frame_dict, boat_voxels_off, boat_voxels_on).

    frame_dict has "mask_off" and "mask_on" sub-dicts with primitive lists.
    If the NPZ has no camera-mask array, the two sub-dicts share the same
    content (mask_on == mask_off).
    """
    data = np.load(npz_path)
    # gen_semantic_occupancy writes a `semantics` key; occupancy_pred dumps
    # store the (200,200,16) uint8 grid under the default `np.savez` name
    # `arr_0` AND use a different class scheme — remap those on load.
    if "semantics" in data.files:
        sem = data["semantics"]
    else:
        lut = np.zeros(256, dtype=np.uint8)
        for k, v in CLASS_REMAP_OCC_PRED.items():
            lut[k] = v
        sem = lut[data[data.files[0]]]
    off, b_off, w_off = _extract_from_sem(sem)
    if "mask_camera" in data.files:
        sem_on = np.where(data["mask_camera"], sem, 0).astype(np.uint8)
        on, b_on, w_on = _extract_from_sem(sem_on)
    else:
        on, b_on, w_on = off, b_off, w_off
    return {"mask_off": off, "mask_on": on}, b_off, b_on, w_off, w_on


_TS_RE = re.compile(r"(\d{10,16})")


def _timestamp(p):
    m = _TS_RE.search(Path(p).stem)
    return int(m.group(1)) if m else 0


def _collect(inputs):
    files = []
    for pat in inputs:
        p = Path(pat)
        if p.is_dir():
            files.extend(p.rglob("*.npz"))
        elif any(c in pat for c in "*?["):
            files.extend(Path().glob(pat))
        elif p.is_file():
            files.append(p)
    # Skip `{timestamp}_sym.npz` companions; only the bare `{timestamp}.npz`
    # files contain the unmirrored semantics we want to visualise.
    files = [f for f in files if not f.stem.endswith("_sym")]
    uniq = sorted({str(f) for f in files}, key=_timestamp)
    return uniq


def _process_one(args):
    """Worker: extract one frame and write its voxel .bin files. Returns the
    frame dict (without ts/name; caller fills those)."""
    i, path, out_dir_str = args
    out_dir = Path(out_dir_str)
    fr, b_off, b_on, w_off, w_on = extract_frame(path)

    def _emit(name, off, on):
        # off is always written; on is skipped when identical to off (no
        # mask_camera) so the JS falls back via activeVoxPath.
        if off is not None and len(off):
            rel = f"voxels/{name}_off_{i:05d}.bin"
            (out_dir / rel).write_bytes(off.tobytes())
            fr[f"{name}_voxels_off_file"] = rel
            fr[f"{name}_voxels_off_count"] = int(len(off))
        else:
            fr[f"{name}_voxels_off_file"] = None
            fr[f"{name}_voxels_off_count"] = 0
        same = on is off or (
            on is not None and off is not None
            and on.shape == off.shape and np.array_equal(on, off))
        if same:
            fr[f"{name}_voxels_on_file"] = None
            fr[f"{name}_voxels_on_count"] = fr[f"{name}_voxels_off_count"]
        elif on is not None and len(on):
            rel = f"voxels/{name}_on_{i:05d}.bin"
            (out_dir / rel).write_bytes(on.tobytes())
            fr[f"{name}_voxels_on_file"] = rel
            fr[f"{name}_voxels_on_count"] = int(len(on))
        else:
            fr[f"{name}_voxels_on_file"] = None
            fr[f"{name}_voxels_on_count"] = 0

    _emit("boat", b_off, b_on)
    _emit("water", w_off, w_on)
    # Drop mask_on entirely when identical to mask_off — the JS falls back
    # via activeSet. Halves frames.json on datasets with no camera mask.
    if fr.get("mask_on") is fr.get("mask_off") or fr.get("mask_on") == fr.get("mask_off"):
        fr["mask_on"] = None
    return i, fr


def _write_html(out_dir):
    (out_dir / "index.html").write_text(HTML)
    print(f"wrote {out_dir}/index.html")


def build(inputs, out_dir, workers=None, html_only=False):
    # JS-only edits don't need to touch frames.json or any voxel .bin file.
    if html_only:
        if not (out_dir / "frames.json").exists():
            sys.exit(f"--html-only needs an existing build at {out_dir}")
        _write_html(out_dir)
        return

    files = _collect(inputs)
    if not files:
        sys.exit("no NPZ files found")
    print(f"found {len(files)} npz files")

    out_dir.mkdir(parents=True, exist_ok=True)
    vox_dir = out_dir / "voxels"
    vox_dir.mkdir(exist_ok=True)

    nproc = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"extracting with {nproc} workers")

    tasks = [(i, f, str(out_dir)) for i, f in enumerate(files)]
    frames = [None] * len(files)
    done = 0
    if nproc == 1:
        for t in tasks:
            i, fr = _process_one(t)
            frames[i] = fr
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)}")
    else:
        # chunksize = 4 keeps IPC overhead low without starving short tails.
        with mp.Pool(nproc) as pool:
            for i, fr in pool.imap_unordered(_process_one, tasks, chunksize=4):
                frames[i] = fr
                done += 1
                if done % 20 == 0 or done == len(files):
                    print(f"  processed {done}/{len(files)}")

    for i, f in enumerate(files):
        frames[i]["timestamp"] = _timestamp(f)
        frames[i]["name"] = Path(f).name

    vendor_three(out_dir)
    if GLB_BOAT_SRC.exists():
        dst = out_dir / "3dboat.glb"
        dst.write_bytes(GLB_BOAT_SRC.read_bytes())
        print(f"  copied {GLB_BOAT_SRC} -> {dst}")
    else:
        print(f"  (no GLB boat asset at {GLB_BOAT_SRC}; GLB mode disabled)")
    (out_dir / "frames.json").write_text(json.dumps({
        "grid": {"nx": NX, "ny": NY, "nz": NZ, "voxel_size": VOXEL},
        "frames": frames,
    }))
    _write_html(out_dir)
    print(f"wrote {out_dir}/frames.json")


def serve(out_dir, port, open_browser):
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    try:
        httpd = socketserver.ThreadingTCPServer(("", port), handler)
    except OSError as e:
        if e.errno == 98:  # address already in use
            print(f"port {port} already in use — assuming an existing "
                  f"server is running; just refresh the browser.")
            return
        raise
    with httpd:
        url = f"http://localhost:{port}/"
        print(f"serving {out_dir} at {url}   (Ctrl-C to stop)")
        if open_browser:
            threading.Thread(target=lambda: webbrowser.open(url),
                             daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*",
                    help="NPZ files, directories, or glob patterns "
                         "(omit when using --html-only)")
    ap.add_argument("--out", default="viewer_build", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-serve", action="store_true",
                    help="only build; skip serving")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel extract workers (default: cpu_count - 1)")
    ap.add_argument("--html-only", action="store_true",
                    help="rewrite index.html only; reuse existing "
                         "frames.json + voxel/. Use after JS-only edits.")
    args = ap.parse_args()

    build(args.inputs, args.out,
          workers=args.workers, html_only=args.html_only)
    if not args.no_serve:
        serve(args.out, args.port, not args.no_browser)


HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"><title>NPZ Viewer</title>
<style>
  html,body { margin:0; padding:0; height:100%; background:#0b0f14;
              color:#cdd6e0; font-family: ui-monospace, monospace;
              overflow:hidden; }
  #hud { position:fixed; top:10px; left:10px; padding:10px 12px;
         background:#0009; border:1px solid #2a3340; border-radius:6px;
         z-index:10; user-select:none; }
  #hud button { background:#1e2630; color:#cdd6e0; border:1px solid #2a3340;
                padding:3px 10px; margin-right:4px; cursor:pointer;
                border-radius:3px; font-family:inherit; }
  #hud button:hover { background:#2a3340; }
  #hud button.active { background:#3a4a60; }
  #info { font-size:12px; margin-bottom:6px; color:#8fa0b0; }
  #scrub { width:340px; margin-top:8px; }
  #legend { margin-top:8px; font-size:11px; color:#8fa0b0; line-height:1.6; }
  .sw { display:inline-block; width:10px; height:10px; margin-right:4px;
        vertical-align:middle; border:1px solid #00000080; }
</style>
<script type="importmap">
{ "imports": {
  "three": "./vendor/three.module.js",
  "three/addons/": "./vendor/addons/"
}}
</script>
</head><body>
<div id="hud">
  <div id="info">loading...</div>
  <div>
    <button id="playpause">Pause</button>
    <button id="prev">&laquo;</button>
    <button id="next">&raquo;</button>
    <button id="bev" class="active">BEV</button>
    <button id="persp">Perspective</button>
    <button id="boatmode">Boats: Model</button>
    <button id="watermode" class="active">Water: On</button>
    <button id="maskmode">Mask: Off</button>
  </div>
  <div><input id="scrub" type="range" min="0" max="0" step="1" value="0"></div>
  <div id="legend"></div>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';

const data = await (await fetch('frames.json')).json();
const { grid, frames } = data;
const GX = grid.nx * grid.voxel_size;
const GY = grid.ny * grid.voxel_size;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 2000);
camera.up.set(0, 0, 1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.4;
controls.zoomSpeed = 0.6;
controls.panSpeed = 0.6;

function setBEV() {
  // BEV: looking straight down from +Z; screen-up = world +X (ego forward).
  camera.up.set(1, 0, 0);
  camera.position.set(0, 0, 140);
  controls.target.set(0, 0, 0);
  controls.update();
  document.getElementById('bev').classList.add('active');
  document.getElementById('persp').classList.remove('active');
}
function setPersp() {
  camera.up.set(0, 0, 1);
  camera.position.set(80, -80, 70);
  controls.target.set(0, 0, 0);
  controls.update();
  document.getElementById('bev').classList.remove('active');
  document.getElementById('persp').classList.add('active');
}
setBEV();

scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const dl = new THREE.DirectionalLight(0xffffff, 1.2);
dl.position.set(60, 60, 120);
scene.add(dl);


const COLORS = {
  boats:     0x1166ff,
  docks:     0x8b5a2b,
  buildings: 0x555a60,
  persons:   0xff3030,
  poles:     0xaaaaaa,
  water:     0x2a5a8c,
};

const MATS = {
  boats:     new THREE.MeshLambertMaterial({ color: COLORS.boats }),
  docks:     new THREE.MeshLambertMaterial({ color: COLORS.docks }),
  buildings: new THREE.MeshLambertMaterial({ color: COLORS.buildings }),
  persons:   new THREE.MeshLambertMaterial({ color: COLORS.persons }),
  poles:     new THREE.MeshLambertMaterial({ color: COLORS.poles }),
  water:     new THREE.MeshLambertMaterial({
    color: COLORS.water, transparent: true, opacity: 0.55 }),
};

// Pastel rainbow palette for ellipsoids: HSV(h, 0.9, 1.0) blended 10% with
// 90% white. Matches the reference open3d viewer (ELLIPSOID_ALPHA=0.1).
function _hsv2rgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    case 5: return [v, p, q];
  }
}
const ELL_PALETTE_N = 12;
const ELL_PALETTE = [];
for (let i = 0; i < ELL_PALETTE_N; i++) {
  const hue = (0.1 + i / ELL_PALETTE_N) % 1.0;
  const [r, g, b] = _hsv2rgb(hue, 0.9, 1.0);
  const a = 0.1;  // 10% color, 90% white -> pastel
  ELL_PALETTE.push(new THREE.MeshLambertMaterial({
    color: new THREE.Color(r * a + (1 - a), g * a + (1 - a), b * a + (1 - a)),
  }));
}

const legend = document.getElementById('legend');
for (const [k, c] of Object.entries(COLORS)) {
  const hex = c.toString(16).padStart(6, '0');
  legend.innerHTML += `<span class="sw" style="background:#${hex}"></span>${k}<br>`;
}

const staticGroup = new THREE.Group();
const boatGroup = new THREE.Group();
const waterGroup = new THREE.Group();
scene.add(staticGroup);
scene.add(boatGroup);
scene.add(waterGroup);

function clearGroup(g) {
  while (g.children.length) {
    const c = g.children.pop();
    c.traverse(obj => {
      // Geometries marked `userData.shared` are owned by long-lived
      // templates (e.g. the simple boat template); disposing them per
      // frame would force a needless GPU re-upload of every clone.
      if (obj.geometry && !obj.geometry.userData?.shared) {
        obj.geometry.dispose();
      }
    });
  }
}

function cuboid(info, mat) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(info.sx, info.sy, info.sz), mat);
  m.position.set(info.cx, info.cy, info.cz);
  return m;
}
function obb(info, mat) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(info.sx, info.sy, info.sz), mat);
  m.position.set(info.cx, info.cy, info.cz);
  m.rotation.z = info.yaw || 0;
  return m;
}
const ELL_GEOM = new THREE.SphereGeometry(1, 24, 16);
const _ELL_M3 = new THREE.Matrix3();
const _ELL_M4 = new THREE.Matrix4();
function ellipsoid(info, mat) {
  // axes = [v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z] (columns of R).
  const a = info.axes;
  // Matrix3.set is row-major; build R with columns = v1,v2,v3.
  _ELL_M3.set(
    a[0], a[3], a[6],
    a[1], a[4], a[7],
    a[2], a[5], a[8]);
  _ELL_M4.setFromMatrix3(_ELL_M3);
  const m = new THREE.Mesh(ELL_GEOM, mat);
  m.quaternion.setFromRotationMatrix(_ELL_M4);
  m.scale.set(info.rx, info.ry, info.rz);
  m.position.set(info.cx, info.cy, info.cz);
  return m;
}
function bowArrow(info) {
  const d = info.bow_dir;
  const dir = new THREE.Vector3(d[0], d[1], d[2]).normalize();
  const origin = new THREE.Vector3(info.cx, info.cy, info.cz);
  // Reference: arrow_len = radii[0] * VOXEL * ARROW_LENGTH_RATIO = rx * 3.5
  const len = info.rx * 3.5;
  return new THREE.ArrowHelper(dir, origin, len, 0x00cc66,
                               len * 0.3, len * 0.25);
}

// Detected boats in 'model' mode share the ego GLB shape, but each clone
// has its mesh materials overridden by one of the ELL_PALETTE colors so
// the boat picks up the same pastel hue as its ellipsoid would have used.
// Non-ego clones use a simplified geometry to keep the scene light.
let egoTemplate = null;          // full-detail template (ego only)
let boatCloneTemplate = null;    // decimated template (non-ego boats)
let egoTemplateLen = 1.0;

// Stride into the source positions array when collecting convex-hull seeds.
// Larger stride = faster hull computation on large GLBs; 1 uses every vertex.
const BOAT_HULL_STRIDE = 1;

const gltfLoader = new GLTFLoader();
gltfLoader.load('3dboat.glb', (gltf) => {
  egoTemplate = gltf.scene;
  // Mark every geometry as shared so clearGroup() won't dispose it when
  // boat clones get cleared each frame.
  egoTemplate.traverse(obj => {
    if (obj.geometry) obj.geometry.userData.shared = true;
  });
  // Length along +X (head axis). Y-up rotation keeps X invariant so this
  // measurement is valid for both the ego and the cloned boats.
  egoTemplate.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(egoTemplate);
  egoTemplateLen = Math.max(0.01, box.max.x - box.min.x);

  // Non-ego boats: replace each mesh's geometry with its convex hull so the
  // silhouette stays recognisable but internal/curved detail collapses to a
  // handful of flat faces. Fast (O(n log n)) and keeps ego full-detail.
  try {
    boatCloneTemplate = egoTemplate.clone();
    boatCloneTemplate.traverse(obj => {
      if (!obj.isMesh || !obj.geometry) return;
      const src = obj.geometry;
      const pos = src.attributes.position;
      if (!pos || pos.count < 4) return;
      try {
        const pts = [];
        for (let i = 0; i < pos.count; i += BOAT_HULL_STRIDE) {
          pts.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)));
        }
        const hull = new ConvexGeometry(pts);
        hull.userData.shared = true;
        obj.geometry = hull;
      } catch (e) {
        console.warn('boat convex hull failed, using full mesh:', e);
      }
    });
  } catch (e) {
    console.warn('boat clone template build failed:', e);
    boatCloneTemplate = null;
  }

  // Ego instance — shifted slightly forward along +X (head direction).
  const egoWrap = new THREE.Group();
  const ego = egoTemplate.clone();
  ego.rotation.x = Math.PI / 2;  // Y-up -> Z-up
  egoWrap.add(ego);
  egoWrap.position.set(2, 0, 0);
  scene.add(egoWrap);

  if (typeof renderFrame === 'function') renderFrame(cur);
}, undefined, (err) => {
  console.warn('3dboat.glb load failed:', err);
});

function boatModel(info, idx) {
  const template = boatCloneTemplate || egoTemplate;
  if (!template) return null;
  const wrapper = new THREE.Group();
  const inner = template.clone();
  // Override every Mesh's material so the boat is rendered in a single
  // unified palette color (matches what the ellipsoid mode would show).
  const mat = ELL_PALETTE[idx % ELL_PALETTE_N];
  inner.traverse(obj => {
    if (obj.isMesh) obj.material = mat;
  });
  inner.rotation.x = Math.PI / 2;  // Y-up -> Z-up
  const scale = (info.rx * 2) / egoTemplateLen;  // model length -> 2 * rx
  inner.scale.set(scale, scale, scale);
  wrapper.add(inner);
  wrapper.rotation.z = Math.atan2(info.bow_dir[1], info.bow_dir[0]);
  wrapper.position.set(info.cx, info.cy, info.cz - info.rz);
  return wrapper;
}
function prism(info, mat) {
  const poly = info.polygon;
  const shape = new THREE.Shape();
  shape.moveTo(poly[0][0], poly[0][1]);
  for (let i = 1; i < poly.length; i++) shape.lineTo(poly[i][0], poly[i][1]);
  shape.closePath();
  const geom = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(1e-3, info.z1 - info.z0),
    bevelEnabled: false,
    curveSegments: 1,
    steps: 1,
  });
  const m = new THREE.Mesh(geom, mat);
  m.position.z = info.z0;
  return m;
}

const VOXEL_BOX = new THREE.BoxGeometry(
  grid.voxel_size, grid.voxel_size, grid.voxel_size);
function voxelInstancedMesh(int16arr, mat) {
  const n = int16arr.length / 3;
  const mesh = new THREE.InstancedMesh(VOXEL_BOX, mat, n);
  const m = new THREE.Matrix4();
  for (let i = 0; i < n; i++) {
    const ix = int16arr[3*i], iy = int16arr[3*i+1], iz = int16arr[3*i+2];
    m.makeTranslation(
      (ix - grid.nx/2 + 0.5) * grid.voxel_size,
      (iy - grid.ny/2 + 0.5) * grid.voxel_size,
      (iz - grid.nz/2 + 0.5) * grid.voxel_size);
    mesh.setMatrixAt(i, m);
  }
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

// LRU-ish cache of decoded Int16Arrays, keyed by file path.
const VOX_CACHE = new Map();
const VOX_CACHE_MAX = 120;
async function loadVoxels(path) {
  if (VOX_CACHE.has(path)) {
    const v = VOX_CACHE.get(path);
    VOX_CACHE.delete(path); VOX_CACHE.set(path, v);  // bump LRU
    return v;
  }
  const buf = await (await fetch(path)).arrayBuffer();
  const arr = new Int16Array(buf);
  VOX_CACHE.set(path, arr);
  if (VOX_CACHE.size > VOX_CACHE_MAX) {
    const k = VOX_CACHE.keys().next().value;
    VOX_CACHE.delete(k);
  }
  return arr;
}

// Render-time pole radius override (meters). info.r in frames.json is baked
// at extract time; override here so the radius can be tuned without a full
// rebuild. Keep in sync with POLE_RADIUS in the Python side for new builds.
const POLE_R = 0.375;
function cylinder(info, mat) {
  const h = info.z1 - info.z0;
  const g = new THREE.CylinderGeometry(POLE_R, POLE_R, h, 16);
  const m = new THREE.Mesh(g, mat);
  m.rotation.x = Math.PI / 2;  // align cylinder axis (Y) -> world Z
  m.position.set(info.cx, info.cy, (info.z0 + info.z1) / 2);
  return m;
}

const BOAT_MODES = ['voxels', 'obb', 'ellipsoid', 'model'];
const BOAT_LABELS = { voxels: 'Voxels', obb: 'Cuboid',
                      ellipsoid: 'Ellipsoid', model: 'Model' };
let boatMode = 'model';
let maskOn = false;
let waterOn = true;
let renderedIdx = -1;
let boatIdx = -1;
let waterIdx = -1;

// `mask_on` / `*_voxels_on_file` are omitted when identical to the `off`
// variant (the source NPZ had no `mask_camera`); fall back so the mask
// toggle is a no-op in that case instead of clearing the scene.
function activeSet(f) { return (maskOn && f.mask_on) ? f.mask_on : f.mask_off; }
function activeBoatVoxPath(f) {
  return (maskOn && f.boat_voxels_on_file)
    ? f.boat_voxels_on_file : f.boat_voxels_off_file;
}
function activeWaterVoxPath(f) {
  return (maskOn && f.water_voxels_on_file)
    ? f.water_voxels_on_file : f.water_voxels_off_file;
}

function renderFrame(idx) {
  const f = frames[idx];
  const s = activeSet(f);
  renderedIdx = idx;

  // Static layer: rebuilt synchronously each frame.
  clearGroup(staticGroup);
  for (const b of s.docks)     staticGroup.add(prism(b, MATS.docks));
  for (const b of s.buildings) staticGroup.add(cuboid(b, MATS.buildings));
  for (const b of s.persons)   staticGroup.add(cuboid(b, MATS.persons));
  for (const p of s.poles)     staticGroup.add(cylinder(p, MATS.poles));

  // Boat layer: only clear+replace when the new data is ready.
  if (boatMode === 'voxels') {
    const path = activeBoatVoxPath(f);
    if (path) {
      loadVoxels(path).then(arr => {
        if (renderedIdx !== idx || boatMode !== 'voxels') return;
        clearGroup(boatGroup);
        boatGroup.add(voxelInstancedMesh(arr, MATS.boats));
        boatIdx = idx;
      });
      for (let k = 1; k <= 10; k++) {
        const j = (idx + k) % frames.length;
        const fr = frames[j];
        if (fr.boat_voxels_on_file)  loadVoxels(fr.boat_voxels_on_file).catch(() => {});
        if (fr.boat_voxels_off_file) loadVoxels(fr.boat_voxels_off_file).catch(() => {});
      }
    } else {
      clearGroup(boatGroup);
      boatIdx = idx;
    }
  } else if (boatMode === 'obb') {
    clearGroup(boatGroup);
    for (const b of s.boats) boatGroup.add(obb(b, MATS.boats));
    boatIdx = idx;
  } else if (boatMode === 'ellipsoid') {
    clearGroup(boatGroup);
    const ells = s.ellipsoids || [];
    for (let i = 0; i < ells.length; i++) {
      const e = ells[i];
      const mat = ELL_PALETTE[i % ELL_PALETTE_N];
      boatGroup.add(ellipsoid(e, mat));
      boatGroup.add(bowArrow(e));
    }
    boatIdx = idx;
  } else {  // model: clone the ego GLB per ellipsoid, palette-coloured
    clearGroup(boatGroup);
    const ells = s.ellipsoids || [];
    for (let i = 0; i < ells.length; i++) {
      const m = boatModel(ells[i], i);
      if (m) boatGroup.add(m);
    }
    boatIdx = idx;
  }

  // Water layer: optional InstancedMesh of surface voxels, async-loaded.
  if (waterOn) {
    const wpath = activeWaterVoxPath(f);
    if (wpath) {
      loadVoxels(wpath).then(arr => {
        if (renderedIdx !== idx || !waterOn) return;
        clearGroup(waterGroup);
        waterGroup.add(voxelInstancedMesh(arr, MATS.water));
        waterIdx = idx;
      });
    } else {
      clearGroup(waterGroup);
      waterIdx = idx;
    }
  } else if (waterIdx !== -1) {
    clearGroup(waterGroup);
    waterIdx = -1;
  }

  document.getElementById('info').textContent =
    `[${idx+1}/${frames.length}] ${f.name}  ts=${f.timestamp}`;
  document.getElementById('scrub').value = idx;
}

let cur = 0;
let playing = true;
let acc = 0;
const FRAME_DT = 1 / 10;  // 10 FPS
const clock = new THREE.Clock();

function tick() {
  controls.update();
  const dt = clock.getDelta();
  if (playing && frames.length > 1) {
    acc += dt;
    while (acc >= FRAME_DT) {
      acc -= FRAME_DT;
      cur = (cur + 1) % frames.length;
      renderFrame(cur);
    }
  }
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

const pp = document.getElementById('playpause');
pp.onclick = () => {
  playing = !playing;
  pp.textContent = playing ? 'Pause' : 'Play';
};
document.getElementById('prev').onclick = () => {
  playing = false; pp.textContent = 'Play';
  cur = (cur - 1 + frames.length) % frames.length; renderFrame(cur);
};
document.getElementById('next').onclick = () => {
  playing = false; pp.textContent = 'Play';
  cur = (cur + 1) % frames.length; renderFrame(cur);
};
document.getElementById('bev').onclick = setBEV;
document.getElementById('persp').onclick = setPersp;
const bm = document.getElementById('boatmode');
bm.onclick = () => {
  const i = BOAT_MODES.indexOf(boatMode);
  boatMode = BOAT_MODES[(i + 1) % BOAT_MODES.length];
  bm.textContent = 'Boats: ' + BOAT_LABELS[boatMode];
  renderFrame(cur);
};
const mm = document.getElementById('maskmode');
mm.onclick = () => {
  maskOn = !maskOn;
  mm.textContent = 'Mask: ' + (maskOn ? 'On' : 'Off');
  mm.classList.toggle('active', maskOn);
  renderFrame(cur);
};
const wm = document.getElementById('watermode');
wm.onclick = () => {
  waterOn = !waterOn;
  wm.textContent = 'Water: ' + (waterOn ? 'On' : 'Off');
  wm.classList.toggle('active', waterOn);
  renderFrame(cur);
};

const scrub = document.getElementById('scrub');
scrub.max = Math.max(0, frames.length - 1);
scrub.oninput = (e) => {
  playing = false; pp.textContent = 'Play';
  cur = parseInt(e.target.value); renderFrame(cur);
};

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Numpad orbit (Blender-style). Yaw rotates around world +Z; pitch rotates
// around the camera-right vector (current_up × back). camera.up is rotated
// alongside the offset so screen orientation stays consistent — without
// this, BEV (camera.up = +X, offset purely along +Z) lands in the
// up-parallel-to-forward degeneracy on the first yaw and the view flips.
//   Numpad 4 / 6  : yaw left / right
//   Numpad 8 / 2  : pitch up / down
//   Numpad + / -  : zoom in / out
//   Numpad 5      : reset camera.up to world +Z (useful from a tilted view)
const KEY_ROT_DEG = 5;
const KEY_ZOOM = 1.1;
const _Z_AXIS = new THREE.Vector3(0, 0, 1);
function _keyOrbit(yawDeg, pitchDeg) {
  const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
  if (yawDeg) {
    const rad = THREE.MathUtils.degToRad(yawDeg);
    offset.applyAxisAngle(_Z_AXIS, rad);
    camera.up.applyAxisAngle(_Z_AXIS, rad);
  }
  if (pitchDeg) {
    const back = offset.clone().normalize();
    let right = new THREE.Vector3().crossVectors(camera.up, back);
    if (right.lengthSq() < 1e-6) right.set(0, 1, 0);  // up || back fallback
    right.normalize();
    const rad = THREE.MathUtils.degToRad(pitchDeg);
    offset.applyAxisAngle(right, rad);
    camera.up.applyAxisAngle(right, rad);
  }
  camera.position.copy(controls.target).add(offset);
  camera.lookAt(controls.target);
  controls.update();
}
addEventListener('keydown', (ev) => {
  if (ev.target?.matches?.('input,textarea')) return;
  let handled = true;
  switch (ev.code) {
    case 'Numpad4': _keyOrbit(-KEY_ROT_DEG, 0); break;
    case 'Numpad6': _keyOrbit( KEY_ROT_DEG, 0); break;
    case 'Numpad8': _keyOrbit(0, -KEY_ROT_DEG); break;
    case 'Numpad2': _keyOrbit(0,  KEY_ROT_DEG); break;
    case 'NumpadAdd':
      camera.position.lerpVectors(controls.target, camera.position, 1/KEY_ZOOM);
      controls.update();
      break;
    case 'NumpadSubtract':
      camera.position.lerpVectors(controls.target, camera.position, KEY_ZOOM);
      controls.update();
      break;
    case 'Numpad5':
      camera.up.set(0, 0, 1); controls.update();
      break;
    default: handled = false;
  }
  if (handled) ev.preventDefault();
});

renderFrame(0);
tick();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
