#!/usr/bin/env python3
"""
Virtual Road Network Generator

Generates a SUMO network directly from geometry parameters
(lanes, avg_block_m, etc.) output by the fine-tuned model.
Enables simulation without OSM data.

Supported road types:
  - intersection: Single intersection (4-way)
  - straight: Straight road (with N intersections)
  - curve: Curved road
  - grid: Grid network

Usage:
    from tools.network_generator import generate_network
    net_path = generate_network(params, output_dir)
"""

import math
import os
import subprocess
import xml.etree.ElementTree as ET

from src.config import get_netconvert_bin


def generate_network(params: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    net_type = params.get("network_type", "straight")
    lanes = int(params.get("lanes", 2))
    speed_limit = float(params.get("speed_limit_kmh", 50)) / 3.6
    block_m = int(params.get("avg_block_m", 200))
    total_length = int(params.get("total_length_m", 0))
    name = params.get("name", "generated")

    # Default total_length: depends on network type
    if not total_length:
        if net_type == "intersection":
            total_length = block_m  # single block for intersection
        elif net_type == "curve":
            total_length = 500
        elif net_type == "grid":
            total_length = block_m * 3
        else:
            total_length = block_m * 4  # straight: 3-4 intersections

    nod_path = os.path.join(output_dir, f"{name}.nod.xml")
    edg_path = os.path.join(output_dir, f"{name}.edg.xml")
    net_path = os.path.join(output_dir, f"{name}.net.xml")

    if net_type == "intersection":
        _gen_intersection(nod_path, edg_path, lanes, speed_limit, block_m)
    elif net_type == "curve":
        radius = float(params.get("curve_radius_m", 500))
        _gen_curve(nod_path, edg_path, lanes, speed_limit, total_length, radius)
    elif net_type == "grid":
        _gen_grid(nod_path, edg_path, lanes, speed_limit, block_m, total_length)
    else:
        _gen_straight(nod_path, edg_path, lanes, speed_limit, block_m, total_length)

    cmd = [
        get_netconvert_bin(),
        "--node-files", nod_path,
        "--edge-files", edg_path,
        "--output-file", net_path,
        "--no-turnarounds", "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if not os.path.exists(net_path):
        raise RuntimeError(f"netconvert failed: {result.stderr}")

    print(f"Network generation complete: {net_path}")
    return net_path


def _gen_intersection(nod_path, edg_path, lanes, speed, arm_length):
    """Single cross intersection."""
    arm = max(arm_length, 100)
    cross_lanes = max(1, lanes - 1)
    nodes = [
        {"id": "center", "x": 0, "y": 0, "type": "traffic_light"},
        {"id": "north", "x": 0, "y": arm, "type": "priority"},
        {"id": "south", "x": 0, "y": -arm, "type": "priority"},
        {"id": "east", "x": arm, "y": 0, "type": "priority"},
        {"id": "west", "x": -arm, "y": 0, "type": "priority"},
    ]
    edges = []
    for direction, node in [("north", "north"), ("south", "south"), ("east", "east"), ("west", "west")]:
        is_main = direction in ("east", "west")
        ln = lanes if is_main else cross_lanes
        edges.append({"id": f"{direction}_in", "from": node, "to": "center", "lanes": ln, "speed": speed})
        edges.append({"id": f"{direction}_out", "from": "center", "to": node, "lanes": ln, "speed": speed})

    _write_nod_xml(nod_path, nodes)
    _write_edg_xml(edg_path, edges)


def _gen_straight(nod_path, edg_path, lanes, speed, block_m, total_length):
    """Straight road with intersections."""
    block_m = max(block_m, 50)
    n_intersections = max(1, min(total_length // block_m, 10))  # max 10
    nodes = []
    edges = []

    for i in range(n_intersections + 1):
        x = i * block_m
        nodes.append({"id": f"n{i}", "x": x, "y": 0,
                       "type": "traffic_light" if 0 < i < n_intersections else "priority"})

    for i in range(n_intersections):
        edges.append({"id": f"main_{i}", "from": f"n{i}", "to": f"n{i+1}",
                       "lanes": lanes, "speed": speed})
        edges.append({"id": f"main_{i}_rev", "from": f"n{i+1}", "to": f"n{i}",
                       "lanes": lanes, "speed": speed})

    # Cross streets (only at internal intersections)
    cross_lanes = max(1, lanes - 1)
    arm_len = block_m * 0.5
    for i in range(1, n_intersections):
        x = i * block_m
        nodes.append({"id": f"s{i}", "x": x, "y": -arm_len, "type": "priority"})
        nodes.append({"id": f"nn{i}", "x": x, "y": arm_len, "type": "priority"})
        edges.append({"id": f"cross_{i}_in_s", "from": f"s{i}", "to": f"n{i}",
                       "lanes": cross_lanes, "speed": speed * 0.8})
        edges.append({"id": f"cross_{i}_out_s", "from": f"n{i}", "to": f"s{i}",
                       "lanes": cross_lanes, "speed": speed * 0.8})
        edges.append({"id": f"cross_{i}_in_n", "from": f"nn{i}", "to": f"n{i}",
                       "lanes": cross_lanes, "speed": speed * 0.8})
        edges.append({"id": f"cross_{i}_out_n", "from": f"n{i}", "to": f"nn{i}",
                       "lanes": cross_lanes, "speed": speed * 0.8})

    _write_nod_xml(nod_path, nodes)
    _write_edg_xml(edg_path, edges)


def _gen_curve(nod_path, edg_path, lanes, speed, total_length, radius):
    """Curved road."""
    angle_rad = total_length / radius
    n_segments = max(5, int(total_length / 50))
    nodes = []
    edges = []

    for i in range(n_segments + 1):
        t = i / n_segments
        theta = t * angle_rad
        x = radius * math.sin(theta)
        y = radius * (1 - math.cos(theta))
        nodes.append({"id": f"c{i}", "x": round(x, 1), "y": round(y, 1), "type": "priority"})

    for i in range(n_segments):
        edges.append({"id": f"curve_{i}", "from": f"c{i}", "to": f"c{i+1}",
                       "lanes": lanes, "speed": speed})
        edges.append({"id": f"curve_{i}_rev", "from": f"c{i+1}", "to": f"c{i}",
                       "lanes": lanes, "speed": speed})

    _write_nod_xml(nod_path, nodes)
    _write_edg_xml(edg_path, edges)


def _gen_grid(nod_path, edg_path, lanes, speed, block_m, total_length):
    """Grid network."""
    grid_size = max(2, min(int(math.sqrt(total_length / block_m)), 5))  # max 5x5
    nodes = []
    edges = []

    for r in range(grid_size):
        for c in range(grid_size):
            ntype = "traffic_light" if (0 < r < grid_size-1 and 0 < c < grid_size-1) else "priority"
            nodes.append({"id": f"g{r}_{c}", "x": c * block_m, "y": r * block_m, "type": ntype})

    cross_lanes = max(1, lanes - 1)
    for r in range(grid_size):
        for c in range(grid_size):
            if c < grid_size - 1:
                edges.append({"id": f"h{r}_{c}", "from": f"g{r}_{c}", "to": f"g{r}_{c+1}",
                               "lanes": lanes, "speed": speed})
                edges.append({"id": f"h{r}_{c}_r", "from": f"g{r}_{c+1}", "to": f"g{r}_{c}",
                               "lanes": lanes, "speed": speed})
            if r < grid_size - 1:
                edges.append({"id": f"v{r}_{c}", "from": f"g{r}_{c}", "to": f"g{r+1}_{c}",
                               "lanes": cross_lanes, "speed": speed * 0.8})
                edges.append({"id": f"v{r}_{c}_r", "from": f"g{r+1}_{c}", "to": f"g{r}_{c}",
                               "lanes": cross_lanes, "speed": speed * 0.8})

    _write_nod_xml(nod_path, nodes)
    _write_edg_xml(edg_path, edges)


def _write_nod_xml(path, nodes):
    root = ET.Element("nodes")
    for n in nodes:
        ET.SubElement(root, "node", id=n["id"],
                      x=str(n["x"]), y=str(n["y"]), type=n["type"])
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, xml_declaration=True, encoding="UTF-8")


def _write_edg_xml(path, edges):
    root = ET.Element("edges")
    for e in edges:
        ET.SubElement(root, "edge", id=e["id"],
                      **{"from": e["from"], "to": e["to"],
                         "numLanes": str(e["lanes"]),
                         "speed": str(round(e["speed"], 2))})
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, xml_declaration=True, encoding="UTF-8")
