# -*- coding: utf-8 -*-
"""
Build a periodic 26-neighbor map for a structured Abaqus C3D8 RVE mesh.

Purpose
-------
Read *Node and *Element sections from a 20x20x20 (or general Nx x Ny x Nz)
Abaqus .inp file, compute element centroids, infer the logical grid index
(i, j, k), and output the periodic neighbor relation used for N_diff calculation.

Outputs
-------
1) element_periodic_neighbor_map.csv
   One row per element, including centroid, grid index, and JSON neighbor list.

2) element_periodic_neighbor_edges.csv
   One row per element-neighbor pair, including offset direction and whether the
   neighbor crosses a periodic boundary.

3) element_periodic_neighbor_map.json
   Compact dictionary form for direct reuse in Python.

Notes
-----
The PBC equations in the inp file are NOT parsed to define neighbors. The PBC
condition is reflected here by periodic indexing of the structured element grid:
(i+di) mod Nx, (j+dj) mod Ny, (k+dk) mod Nz.
"""

import os
import csv
import json
import math
from collections import OrderedDict

# ========================== CONFIGURATION AREA =============================
INP_PATH = r"F:\1_job\20251124\CPFE_PARAMATER_TEST_PBC\2_RVE_BUILD\01_neibor_read_file\input\01_001_20.inp"
OUTPUT_DIR = r"F:\1_job\20251124\CPFE_PARAMATER_TEST_PBC\2_RVE_BUILD\01_neibor_read_file"

# Usually leave these as None. The script infers them from centroid coordinates.
NX = None
NY = None
NZ = None

# Coordinate rounding used when grouping coordinates. Abaqus coordinates often
# contain small float noise such as 0.100000001.
ROUND_DIGITS = 8

# Whether to write the long edge table. It has N_elem * 26 rows.
WRITE_EDGE_TABLE = True
# ===========================================================================


def _is_keyword(line):
    return line.lstrip().startswith("*")


def _split_abaqus_numbers(line):
    return [x.strip() for x in line.replace("\t", " ").split(",") if x.strip()]


def parse_inp_nodes_elements(inp_path, part_name=None):
    """Parse *Node and C3D8 *Element blocks from the first/selected *Part only.

    Abaqus inp files with PBCs may contain additional *Node sections in the
    Assembly for reference points. Those assembly nodes can reuse low labels and
    must not overwrite the RVE part mesh nodes. Therefore this parser restricts
    reading to the part block.
    """
    nodes = OrderedDict()
    elements = OrderedDict()
    mode = None
    in_part = False
    selected_part_found = False
    reading_c3d8 = False

    with open(inp_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            lower = line.lower()

            if _is_keyword(line):
                if lower.startswith("*part"):
                    # Example: *Part, name=DREAM3D
                    this_name = None
                    for token in line.split(","):
                        token = token.strip()
                        if token.lower().startswith("name="):
                            this_name = token.split("=", 1)[1].strip()
                    if part_name is None or (this_name and this_name.lower() == part_name.lower()):
                        in_part = True
                        selected_part_found = True
                    else:
                        in_part = False
                    mode = None
                    reading_c3d8 = False
                    continue

                if lower.startswith("*end part"):
                    if in_part:
                        break
                    mode = None
                    in_part = False
                    reading_c3d8 = False
                    continue

                if not in_part:
                    mode = None
                    reading_c3d8 = False
                    continue

                if lower.startswith("*node"):
                    mode = "node"
                    reading_c3d8 = False
                elif lower.startswith("*element"):
                    if "type=c3d8" in lower:
                        mode = "element"
                        reading_c3d8 = True
                    else:
                        mode = None
                        reading_c3d8 = False
                else:
                    mode = None
                    reading_c3d8 = False
                continue

            if not in_part:
                continue

            if mode == "node":
                parts = _split_abaqus_numbers(line)
                if len(parts) >= 4:
                    try:
                        nid = int(float(parts[0]))
                        xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
                        # Keep first definition in the selected part.
                        if nid not in nodes:
                            nodes[nid] = xyz
                    except ValueError:
                        pass

            elif mode == "element" and reading_c3d8:
                parts = _split_abaqus_numbers(line)
                if len(parts) >= 9:
                    try:
                        eid = int(float(parts[0]))
                        conn = [int(float(x)) for x in parts[1:9]]
                        if len(conn) == 8 and eid not in elements:
                            elements[eid] = conn
                    except ValueError:
                        pass

    if part_name is not None and not selected_part_found:
        raise RuntimeError("Part not found in inp: %s" % part_name)
    if not nodes:
        raise RuntimeError("No part nodes were parsed from inp: %s" % inp_path)
    if not elements:
        raise RuntimeError("No C3D8 part elements were parsed from inp: %s" % inp_path)
    return nodes, elements


def compute_centroids(nodes, elements):
    centroids = OrderedDict()
    for eid, conn in elements.items():
        coords = []
        for nid in conn:
            if nid not in nodes:
                raise RuntimeError("Element %s references missing node %s" % (eid, nid))
            coords.append(nodes[nid])
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        centroids[eid] = (cx, cy, cz)
    return centroids


def _unique_sorted_rounded(values, digits):
    vals = sorted(set(round(v, digits) for v in values))
    return vals


def infer_grid_indices(centroids, nx=None, ny=None, nz=None, round_digits=8):
    xs = _unique_sorted_rounded([c[0] for c in centroids.values()], round_digits)
    ys = _unique_sorted_rounded([c[1] for c in centroids.values()], round_digits)
    zs = _unique_sorted_rounded([c[2] for c in centroids.values()], round_digits)

    if nx is not None and len(xs) != nx:
        raise RuntimeError("Inferred Nx=%d, but configured NX=%d" % (len(xs), nx))
    if ny is not None and len(ys) != ny:
        raise RuntimeError("Inferred Ny=%d, but configured NY=%d" % (len(ys), ny))
    if nz is not None and len(zs) != nz:
        raise RuntimeError("Inferred Nz=%d, but configured NZ=%d" % (len(zs), nz))

    nx, ny, nz = len(xs), len(ys), len(zs)
    expected = nx * ny * nz
    if expected != len(centroids):
        raise RuntimeError(
            "Centroid grid is not complete: Nx*Ny*Nz=%d but element count=%d" %
            (expected, len(centroids))
        )

    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}
    z_to_k = {z: k for k, z in enumerate(zs)}

    eid_to_ijk = OrderedDict()
    ijk_to_eid = {}
    for eid, (x, y, z) in centroids.items():
        key = (round(x, round_digits), round(y, round_digits), round(z, round_digits))
        ijk = (x_to_i[key[0]], y_to_j[key[1]], z_to_k[key[2]])
        if ijk in ijk_to_eid:
            raise RuntimeError("Duplicate element grid index %s for elements %s and %s" %
                               (str(ijk), ijk_to_eid[ijk], eid))
        eid_to_ijk[eid] = ijk
        ijk_to_eid[ijk] = eid

    return nx, ny, nz, xs, ys, zs, eid_to_ijk, ijk_to_eid


def build_periodic_neighbors(nx, ny, nz, eid_to_ijk, ijk_to_eid):
    offsets = [(di, dj, dk)
               for dk in (-1, 0, 1)
               for dj in (-1, 0, 1)
               for di in (-1, 0, 1)
               if not (di == 0 and dj == 0 and dk == 0)]

    neighbor_info = OrderedDict()
    for eid, (i, j, k) in eid_to_ijk.items():
        nbrs = []
        for di, dj, dk in offsets:
            ni = (i + di) % nx
            nj = (j + dj) % ny
            nk = (k + dk) % nz
            nbr_eid = ijk_to_eid[(ni, nj, nk)]
            wrapped = (i + di < 0 or i + di >= nx or
                       j + dj < 0 or j + dj >= ny or
                       k + dk < 0 or k + dk >= nz)
            nbrs.append({
                "neighbor_id": nbr_eid,
                "offset": [di, dj, dk],
                "neighbor_ijk": [ni, nj, nk],
                "periodic_wrap": bool(wrapped)
            })
        neighbor_info[eid] = nbrs
    return neighbor_info


def write_outputs(out_dir, inp_path, nodes, elements, centroids, nx, ny, nz,
                  eid_to_ijk, neighbor_info, write_edge_table=True):
    os.makedirs(out_dir, exist_ok=True)

    map_csv = os.path.join(out_dir, "element_periodic_neighbor_map.csv")
    edge_csv = os.path.join(out_dir, "element_periodic_neighbor_edges.csv")
    map_json = os.path.join(out_dir, "element_periodic_neighbor_map.json")
    summary_json = os.path.join(out_dir, "summary.json")

    with open(map_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Element_ID", "i", "j", "k", "cx", "cy", "cz",
                         "N_total", "Neighbor_List", "Neighbor_Info_JSON"])
        for eid in sorted(elements.keys()):
            i, j, k = eid_to_ijk[eid]
            cx, cy, cz = centroids[eid]
            nbr_ids = [x["neighbor_id"] for x in neighbor_info[eid]]
            writer.writerow([eid, i, j, k, cx, cy, cz, len(nbr_ids),
                             json.dumps(nbr_ids, separators=(",", ":")),
                             json.dumps(neighbor_info[eid], separators=(",", ":"))])

    if write_edge_table:
        with open(edge_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Element_ID", "i", "j", "k", "cx", "cy", "cz",
                             "Neighbor_ID", "ni", "nj", "nk", "ncx", "ncy", "ncz",
                             "di", "dj", "dk", "periodic_wrap"])
            for eid in sorted(elements.keys()):
                i, j, k = eid_to_ijk[eid]
                cx, cy, cz = centroids[eid]
                for nbr in neighbor_info[eid]:
                    nid = nbr["neighbor_id"]
                    ni, nj, nk = nbr["neighbor_ijk"]
                    ncx, ncy, ncz = centroids[nid]
                    di, dj, dk = nbr["offset"]
                    writer.writerow([eid, i, j, k, cx, cy, cz,
                                     nid, ni, nj, nk, ncx, ncy, ncz,
                                     di, dj, dk, int(nbr["periodic_wrap"])])

    json_obj = OrderedDict()
    for eid in sorted(elements.keys()):
        i, j, k = eid_to_ijk[eid]
        cx, cy, cz = centroids[eid]
        json_obj[str(eid)] = {
            "ijk": [i, j, k],
            "centroid": [cx, cy, cz],
            "neighbors": neighbor_info[eid]
        }
    with open(map_json, "w") as f:
        json.dump(json_obj, f, indent=2)

    summary = {
        "source_inp": inp_path,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "neighbors_per_element": 26,
        "map_csv": map_csv,
        "edge_csv": edge_csv if write_edge_table else None,
        "map_json": map_json
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    nodes, elements = parse_inp_nodes_elements(INP_PATH)
    centroids = compute_centroids(nodes, elements)
    nx, ny, nz, xs, ys, zs, eid_to_ijk, ijk_to_eid = infer_grid_indices(
        centroids, nx=NX, ny=NY, nz=NZ, round_digits=ROUND_DIGITS
    )
    neighbor_info = build_periodic_neighbors(nx, ny, nz, eid_to_ijk, ijk_to_eid)
    summary = write_outputs(OUTPUT_DIR, INP_PATH, nodes, elements, centroids,
                            nx, ny, nz, eid_to_ijk, neighbor_info,
                            write_edge_table=WRITE_EDGE_TABLE)

    print("Periodic neighbor map generated successfully.")
    print("Nodes    :", summary["n_nodes"])
    print("Elements :", summary["n_elements"])
    print("Grid     : {nx} x {ny} x {nz}".format(**summary["grid"]))
    print("N_total  :", summary["neighbors_per_element"])
    print("Map CSV  :", summary["map_csv"])
    if summary["edge_csv"]:
        print("Edge CSV :", summary["edge_csv"])
    print("JSON     :", summary["map_json"])


if __name__ == "__main__":
    main()
