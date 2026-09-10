# -*- coding: utf-8 -*-
"""
02_generate_orientation_pools.py

Generate BCC/HCP orientation pools in Rodrigues fundamental regions (FRs),
using the same Rodrigues/FR/Euler conversion logic as u_to_rodrigues.py.

Purpose in the RVE workflow
---------------------------
01_build_periodic_neighbor_map.py
    -> builds the fixed 20^3 periodic neighbor map.
02_generate_orientation_pools.py
    -> builds BCC/HCP basic orientation pools.
03_adaptive_basis_rve_builder.py
    -> reads neighbor map + these pools, then adaptively builds basis-RVE info.

Important sampling convention
-----------------------------
The paper says the sampling step is approximately 15 degrees in Rodrigues space.
To reproduce the paper-level counts (BCC=311, HCP=325), this script samples a
component-wise angular grid q_i with 15-degree spacing, and maps it to Rodrigues
components by

    r_i = tan(q_i / 2).

Then the exact BCC/HCP FR inequalities are applied.

This is different from using a uniform Cartesian Rodrigues spacing of
    dr = tan(15 deg / 2),
which gives slightly different counts.

Outputs
-------
OUTPUT_ROOT/
    BCC_orientation_pool.csv
    HCP_orientation_pool.csv
    orientation_pool_summary.csv
    orientation_pool_summary.json
    BCC_rodrigues_pool_3d.png       optional
    HCP_rodrigues_pool_3d.png       optional

CSV columns
-----------
Orientation_ID, Crystal_Type, Index, Step_Deg,
r1, r2, r3, r_norm, theta_deg,
phi1, Phi, phi2,
g11, g12, ..., g33,
In_FR

Author: ChatGPT
"""
from __future__ import print_function

import os
import csv
import json
import math
import numpy as np

# =============================================================================
# 0. CONFIGURATION AREA
# =============================================================================

# Output directory. Change this to your own project path on Windows, e.g.
# OUTPUT_ROOT = r"F:\1_job\...\orientation_pools"
OUTPUT_ROOT = r"F:\1_job\20251124\CPFE_PARAMATER_TEST_PBC\2_RVE_BUILD\02_generate_orientation_pools_file"

# Sampling step in the component-wise angular grid q_i, in degrees.
SAMPLE_STEP_DEG = 15.0

# Expected counts according to the current manuscript logic.
EXPECTED_COUNTS = {
    "BCC": 311,
    "HCP": 325,
}

# Floating-point tolerance for FR boundary checks.
FR_EPS = 1.0e-7

# Rounding precision for output CSV.
ROUND_DIGITS = 12

# Also save 3D scatter plots of Rodrigues points.
MAKE_PLOTS = True

# If True, raise an error when the generated counts differ from EXPECTED_COUNTS.
# Keep False during debugging if you edit the sampling rule.
STRICT_EXPECTED_COUNT = False

# =============================================================================
# 1. Symmetry and FR logic: same style as u_to_rodrigues.py
# =============================================================================

def get_symmetries(crystal_type="BCC"):
    """
    Return rotation symmetry matrices.
    BCC/Cubic: 24 operators of rotational octahedral group O.
    HCP/Hexagonal: 12 operators of rotational D6 group.
    """
    crystal_type = crystal_type.upper()
    symmetries = []

    if crystal_type == "BCC":
        perms = [
            [0, 1, 2], [0, 2, 1],
            [1, 0, 2], [1, 2, 0],
            [2, 0, 1], [2, 1, 0],
        ]
        signs = [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ]
        for p in perms:
            for s in signs:
                m = np.zeros((3, 3), dtype=float)
                for i in range(3):
                    m[i, p[i]] = s[i]
                if np.isclose(np.linalg.det(m), 1.0):
                    symmetries.append(m)

    elif crystal_type == "HCP":
        for i in range(6):
            angle = np.radians(i * 60.0)
            c, s = np.cos(angle), np.sin(angle)
            sz = np.array([
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=float)
            symmetries.append(sz)

            flip_x = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0],
            ], dtype=float)
            symmetries.append(np.dot(sz, flip_x))

    else:
        raise ValueError("Unsupported crystal_type: %s" % crystal_type)

    return symmetries


def is_in_fr(r, crystal_type="BCC", eps=FR_EPS):
    """Check whether a Rodrigues vector is inside the selected FR."""
    crystal_type = crystal_type.upper()
    r = np.asarray(r, dtype=float)

    if crystal_type == "BCC":
        # max(|rx|, |ry|, |rz|) <= tan(pi/8)
        if np.any(np.abs(r) > math.tan(math.pi / 8.0) + eps):
            return False
        # |rx| + |ry| + |rz| <= 1
        if np.sum(np.abs(r)) > 1.0 + eps:
            return False
        return True

    if crystal_type == "HCP":
        # |rz| <= tan(pi/12)
        if abs(r[2]) > math.tan(math.pi / 12.0) + eps:
            return False

        # Hexagonal prism side limits:
        # max(|ry|, |sqrt(3)/2 rx + 1/2 ry|,
        #     |sqrt(3)/2 rx - 1/2 ry|) <= sqrt(3)/3
        rx, ry = r[0], r[1]
        lim = math.sqrt(3.0) / 3.0
        if abs(ry) > lim + eps:
            return False
        if abs((math.sqrt(3.0) / 2.0) * rx + 0.5 * ry) > lim + eps:
            return False
        if abs((math.sqrt(3.0) / 2.0) * rx - 0.5 * ry) > lim + eps:
            return False
        return True

    raise ValueError("Unsupported crystal_type: %s" % crystal_type)

# =============================================================================
# 2. Rodrigues / matrix / Euler conversions: same style as u_to_rodrigues.py
# =============================================================================

def rodrigues_to_mat(r):
    """Rodrigues vector -> rotation matrix."""
    r = np.asarray(r, dtype=float)
    r2 = float(np.dot(r, r))
    rx, ry, rz = r
    R = np.array([
        [1.0 + rx**2 - ry**2 - rz**2, 2.0 * (rx * ry - rz),          2.0 * (rx * rz + ry)],
        [2.0 * (rx * ry + rz),          1.0 - rx**2 + ry**2 - rz**2, 2.0 * (ry * rz - rx)],
        [2.0 * (rx * rz - ry),          2.0 * (ry * rz + rx),        1.0 - rx**2 - ry**2 + rz**2],
    ], dtype=float)
    return R / (1.0 + r2)


def mat_to_euler(R):
    """Rotation matrix -> Bunge Euler angles in degrees."""
    R = np.asarray(R, dtype=float)
    g33 = np.clip(R[2, 2], -1.0, 1.0)

    if abs(g33 - 1.0) < 1.0e-10:
        phi1 = np.arctan2(R[0, 1], R[0, 0])
        Phi = 0.0
        phi2 = 0.0
    else:
        Phi = np.arccos(g33)
        sP = np.sin(Phi)
        phi1 = np.arctan2(R[2, 0] / sP, -R[2, 1] / sP)
        phi2 = np.arctan2(R[0, 2] / sP,  R[1, 2] / sP)

    return np.degrees([
        phi1 % (2.0 * np.pi),
        Phi,
        phi2 % (2.0 * np.pi),
    ])


def angular_component_to_rodrigues_component(q_deg):
    """q_i in degrees -> r_i = tan(q_i/2)."""
    return math.tan(math.radians(q_deg) / 2.0)

# =============================================================================
# 3. Sampling logic
# =============================================================================

def _angle_grid(start_deg, end_deg, step_deg):
    """Inclusive angular grid, robust against floating-point endpoint errors."""
    n = int(round((end_deg - start_deg) / step_deg))
    values = [start_deg + i * step_deg for i in range(n + 1)]
    return np.array(values, dtype=float)


def get_sampling_angle_ranges(crystal_type):
    """
    Return q_i ranges for component-wise angular sampling.

    BCC:
        |r_i| <= tan(pi/8) = tan(22.5 deg)
        Since r_i = tan(q_i/2), q_i ranges in [-45, 45] deg.

    HCP:
        |rz| <= tan(pi/12) = tan(15 deg) -> q_z in [-30, 30] deg.
        Hex side bound sqrt(3)/3 = tan(30 deg), so q_x/q_y bounding ranges
        are [-60, 60] deg before applying the exact hexagonal FR filter.
    """
    crystal_type = crystal_type.upper()
    if crystal_type == "BCC":
        return [(-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0)]
    if crystal_type == "HCP":
        return [(-60.0, 60.0), (-60.0, 60.0), (-30.0, 30.0)]
    raise ValueError("Unsupported crystal_type: %s" % crystal_type)


def generate_rodrigues_points(crystal_type, step_deg=SAMPLE_STEP_DEG):
    """Generate Rodrigues points inside the selected FR."""
    crystal_type = crystal_type.upper()
    ranges = get_sampling_angle_ranges(crystal_type)
    q_arrays = [_angle_grid(a, b, step_deg) for (a, b) in ranges]

    points = []
    q_points = []
    for qx in q_arrays[0]:
        for qy in q_arrays[1]:
            for qz in q_arrays[2]:
                r = np.array([
                    angular_component_to_rodrigues_component(qx),
                    angular_component_to_rodrigues_component(qy),
                    angular_component_to_rodrigues_component(qz),
                ], dtype=float)
                if is_in_fr(r, crystal_type):
                    points.append(r)
                    q_points.append(np.array([qx, qy, qz], dtype=float))

    if not points:
        raise RuntimeError("No Rodrigues points generated for %s." % crystal_type)

    points = np.vstack(points)
    q_points = np.vstack(q_points)

    # Stable order: low norm first, then qx, qy, qz.
    order = sorted(
        range(points.shape[0]),
        key=lambda i: (
            float(np.linalg.norm(points[i])),
            float(q_points[i, 0]),
            float(q_points[i, 1]),
            float(q_points[i, 2]),
        ),
    )
    return points[order], q_points[order]

# =============================================================================
# 4. Output helpers
# =============================================================================

def _round_float(x):
    return round(float(x), ROUND_DIGITS)


def make_orientation_rows(crystal_type, points, q_points):
    """Build rows for CSV output."""
    rows = []
    crystal_type = crystal_type.upper()
    prefix = crystal_type

    for idx, (r, q) in enumerate(zip(points, q_points), start=1):
        R = rodrigues_to_mat(r)
        euler = mat_to_euler(R)
        r_norm = float(np.linalg.norm(r))
        theta_deg = math.degrees(2.0 * math.atan(r_norm))

        # Sanity checks.
        detR = float(np.linalg.det(R))
        ortho_err = float(np.linalg.norm(np.dot(R.T, R) - np.eye(3)))
        in_fr = is_in_fr(r, crystal_type)

        row = {
            "Orientation_ID": "%s_%04d" % (prefix, idx),
            "Crystal_Type": crystal_type,
            "Index": idx,
            "Step_Deg": _round_float(SAMPLE_STEP_DEG),
            "q1_deg": _round_float(q[0]),
            "q2_deg": _round_float(q[1]),
            "q3_deg": _round_float(q[2]),
            "r1": _round_float(r[0]),
            "r2": _round_float(r[1]),
            "r3": _round_float(r[2]),
            "r_norm": _round_float(r_norm),
            "theta_deg": _round_float(theta_deg),
            "phi1": _round_float(euler[0]),
            "Phi": _round_float(euler[1]),
            "phi2": _round_float(euler[2]),
            "det_g": _round_float(detR),
            "orthogonality_error": _round_float(ortho_err),
            "In_FR": bool(in_fr),
        }

        # Flatten rotation matrix in row-major order: g11 g12 ... g33.
        for a in range(3):
            for b in range(3):
                row["g%d%d" % (a + 1, b + 1)] = _round_float(R[a, b])

        rows.append(row)

    return rows


def write_csv(path, rows):
    if len(rows) == 0:
        raise ValueError("No rows to write: %s" % path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary(path_csv, path_json, summaries):
    rows = []
    for s in summaries:
        rows.append({
            "Crystal_Type": s["Crystal_Type"],
            "Num_Orientations": s["Num_Orientations"],
            "Expected_Count": s["Expected_Count"],
            "Count_Match": s["Count_Match"],
            "Step_Deg": s["Step_Deg"],
            "Sampling_Mode": s["Sampling_Mode"],
            "Output_CSV": s["Output_CSV"],
        })
    write_csv(path_csv, rows)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)


def plot_rodrigues_points(crystal_type, points, output_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[WARN] matplotlib is not available, skip plot:", exc)
        return

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=14, alpha=0.80)
    ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(r"$r_2$")
    ax.set_zlabel(r"$r_3$")
    ax.set_title("%s Rodrigues Orientation Pool, step = %.1f deg, N = %d" % (
        crystal_type.upper(), SAMPLE_STEP_DEG, points.shape[0]
    ))
    ax.view_init(elev=22, azim=42)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# =============================================================================
# 5. Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 78)
    print("02_generate_orientation_pools.py")
    print("Output root:", OUTPUT_ROOT)
    print("Sampling mode: component angular grid q_i, r_i = tan(q_i/2)")
    print("Sampling step: %.6g deg" % SAMPLE_STEP_DEG)
    print("=" * 78)

    summaries = []

    for crystal in ("BCC", "HCP"):
        points, q_points = generate_rodrigues_points(crystal, SAMPLE_STEP_DEG)
        rows = make_orientation_rows(crystal, points, q_points)

        out_csv = os.path.join(OUTPUT_ROOT, "%s_orientation_pool.csv" % crystal)
        write_csv(out_csv, rows)

        if MAKE_PLOTS:
            out_png = os.path.join(OUTPUT_ROOT, "%s_rodrigues_pool_3d.png" % crystal)
            plot_rodrigues_points(crystal, points, out_png)
        else:
            out_png = ""

        expected = EXPECTED_COUNTS.get(crystal, None)
        count_match = (expected is None) or (len(rows) == expected)

        print("[%s] generated orientations: %d" % (crystal, len(rows)))
        if expected is not None:
            print("[%s] expected count       : %d  -> %s" % (
                crystal, expected, "MATCH" if count_match else "MISMATCH"
            ))
        print("[%s] CSV                  : %s" % (crystal, out_csv))
        if MAKE_PLOTS:
            print("[%s] PNG                  : %s" % (crystal, out_png))

        if STRICT_EXPECTED_COUNT and not count_match:
            raise RuntimeError(
                "%s count mismatch: got %d, expected %d" % (crystal, len(rows), expected)
            )

        summaries.append({
            "Crystal_Type": crystal,
            "Num_Orientations": len(rows),
            "Expected_Count": expected,
            "Count_Match": bool(count_match),
            "Step_Deg": SAMPLE_STEP_DEG,
            "Sampling_Mode": "component angular grid q_i, r_i = tan(q_i/2)",
            "FR_Check": "u_to_rodrigues-compatible BCC/HCP inequalities",
            "Output_CSV": out_csv,
            "Output_PNG": out_png,
        })

    summary_csv = os.path.join(OUTPUT_ROOT, "orientation_pool_summary.csv")
    summary_json = os.path.join(OUTPUT_ROOT, "orientation_pool_summary.json")
    save_summary(summary_csv, summary_json, summaries)

    print("-" * 78)
    print("Summary CSV :", summary_csv)
    print("Summary JSON:", summary_json)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
