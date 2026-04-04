"""
SUMO XML File Generator

Generates .rou.xml, .add.xml, and .sumocfg files based on a .net.xml file.
Accepts traffic parameters (demand, vehicle type composition, simulation time, etc.)
to produce a complete simulation set.
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class VehicleType:
    """Vehicle type definition"""
    id: str
    vclass: str  # passenger, truck, bus
    accel: float = 2.6
    decel: float = 4.5
    sigma: float = 0.5
    length: float = 5.0
    max_speed: float = 50.0  # m/s
    speed_factor: float = 1.0
    speed_dev: float = 0.1
    tau: float = 1.0  # headway time (s)
    color: str = "1,1,0"  # yellow


@dataclass
class TrafficDemand:
    """Traffic demand definition"""
    total_vehicles_per_hour: int = 1000
    # Vehicle type ratios (sum = 1.0)
    passenger_ratio: float = 0.85
    truck_ratio: float = 0.10
    bus_ratio: float = 0.05


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    begin_time: int = 0  # seconds
    end_time: int = 3600  # seconds (1 hour)
    warmup_seconds: int = 600  # initial warm-up period excluded from evaluation
    step_length: float = 1.0
    seed: int = 42


# Default vehicle types for Korean urban areas
DEFAULT_VTYPES = [
    VehicleType(
        id="passenger",
        vclass="passenger",
        accel=2.6, decel=4.5, sigma=0.7,
        length=4.5, max_speed=33.3,  # 120km/h
        speed_factor=1.0, speed_dev=0.1,
        tau=1.2, color="1,1,0",
    ),
    VehicleType(
        id="truck",
        vclass="truck",
        accel=1.3, decel=4.0, sigma=0.5,
        length=12.0, max_speed=25.0,  # 90km/h
        speed_factor=0.9, speed_dev=0.05,
        tau=1.5, color="0.5,0.5,0.5",
    ),
    VehicleType(
        id="bus",
        vclass="bus",
        accel=1.2, decel=4.0, sigma=0.5,
        length=12.0, max_speed=22.2,  # 80km/h
        speed_factor=0.85, speed_dev=0.05,
        tau=1.5, color="0,0.5,1",
    ),
]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def build_vtypes_from_ft(
    ft: dict = None,
    speed_limit_kmh: float = None,
    base_vtypes: list[VehicleType] = None,
) -> list[VehicleType]:
    """
    Build runtime vehicle types from FT output.

    The FT layer predicts scene-wide driver-behavior parameters (`sigma`, `tau`).
    We reflect those values into the SUMO vType definitions rather than leaving
    every run on the same hard-coded defaults.
    """
    vtypes = deepcopy(base_vtypes or DEFAULT_VTYPES)
    ft = ft or {}

    sigma = ft.get("sigma")
    tau = ft.get("tau")

    try:
        sigma = _clamp(float(sigma), 0.0, 1.0) if sigma is not None else None
    except (TypeError, ValueError):
        sigma = None

    try:
        tau = _clamp(float(tau), 0.5, 3.0) if tau is not None else None
    except (TypeError, ValueError):
        tau = None

    limit_ms = None
    try:
        if speed_limit_kmh is not None:
            limit_ms = max(float(speed_limit_kmh) / 3.6, 1.0)
    except (TypeError, ValueError):
        limit_ms = None

    for vt in vtypes:
        if sigma is not None:
            vt.sigma = sigma
        if tau is not None:
            vt.tau = tau
        if limit_ms is not None:
            # Keep class-specific caps while preventing implausibly high desired speeds.
            vt.max_speed = min(vt.max_speed, round(limit_ms * 1.05, 2))

    return vtypes


def parse_net_edges(net_path: str) -> list[dict]:
    """
    Extracts edge information from a .net.xml file.
    Internal edges (starting with ':') are excluded.
    """
    tree = ET.parse(net_path)
    root = tree.getroot()

    edges = []
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if edge_id.startswith(":"):
            continue  # Exclude junction internal edges

        lanes = edge.findall("lane")
        edge_info = {
            "id": edge_id,
            "from": edge.get("from", ""),
            "to": edge.get("to", ""),
            "num_lanes": len(lanes),
            "length": float(lanes[0].get("length", 0)) if lanes else 0,
            "speed": float(lanes[0].get("speed", 13.89)) if lanes else 13.89,
        }
        edges.append(edge_info)

    return edges


def find_entry_exit_edges(edges: list[dict]) -> tuple[list[str], list[str]]:
    """
    Finds entry/exit edges of the network.
    Edges whose 'from' node is not in any other edge's 'to' -> entry
    Edges whose 'to' node is not in any other edge's 'from' -> exit
    """
    from_nodes = {e["from"] for e in edges}
    to_nodes = {e["to"] for e in edges}

    entry_edges = [e["id"] for e in edges if e["from"] not in to_nodes]
    exit_edges = [e["id"] for e in edges if e["to"] not in from_nodes]

    return entry_edges, exit_edges


def _write_vtypes_xml(vtypes: list[VehicleType], output_path: str) -> str:
    """Saves vType definitions to a separate XML file (referenced by randomTrips)."""
    root = ET.Element("routes")
    for vt in vtypes:
        ET.SubElement(root, "vType",
            id=vt.id,
            vClass=vt.vclass,
            accel=str(vt.accel),
            decel=str(vt.decel),
            sigma=str(vt.sigma),
            length=str(vt.length),
            maxSpeed=str(vt.max_speed),
            speedFactor=f"normc({vt.speed_factor},{vt.speed_dev},{vt.speed_factor-0.2},{vt.speed_factor+0.2})",
            tau=str(vt.tau),
            color=vt.color,
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _find_random_trips_py() -> str:
    """Finds the path to randomTrips.py."""
    import glob

    # 1) Search in the project venv's sumo package
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_pattern = os.path.join(project_root, ".venv", "lib", "python*",
                                "site-packages", "sumo", "tools", "randomTrips.py")
    matches = glob.glob(venv_pattern)
    if matches:
        return matches[0]

    # 2) Search via importlib
    try:
        import importlib.util
        spec = importlib.util.find_spec("sumo")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                candidate = os.path.join(loc, "tools", "randomTrips.py")
                if os.path.exists(candidate):
                    return candidate
    except (ImportError, ModuleNotFoundError):
        pass

    # 3) SUMO_HOME fallback
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = os.path.join(sumo_home, "tools", "randomTrips.py")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("Could not find randomTrips.py. Please check your SUMO installation.")


def generate_rou_xml(
    net_path: str,
    output_path: str,
    demand: TrafficDemand = None,
    vtypes: list[VehicleType] = None,
    sim_config: SimulationConfig = None,
) -> str:
    """
    Generates a .rou.xml file with valid routes using randomTrips.py.
    """
    if demand is None:
        demand = TrafficDemand()
    if vtypes is None:
        vtypes = DEFAULT_VTYPES
    if sim_config is None:
        sim_config = SimulationConfig()

    output_dir = os.path.dirname(output_path) or "."
    base_name = os.path.basename(output_path).replace(".rou.xml", "")

    # 1) Generate vType file
    vtype_path = os.path.join(output_dir, f"{base_name}.vtype.xml")
    _write_vtypes_xml(vtypes, vtype_path)

    # 2) Generate trips using randomTrips.py
    random_trips_py = _find_random_trips_py()
    trip_path = os.path.join(output_dir, f"{base_name}.trips.xml")

    duration = sim_config.end_time - sim_config.begin_time
    total_vehicles = int(demand.total_vehicles_per_hour * duration / 3600)

    # period = duration / total_vehicles (interval in seconds)
    period = duration / max(total_vehicles, 1)

    # Vehicle type ratio string: "passenger 0.85 truck 0.10 bus 0.05"
    trip_attributes = []
    for vt, ratio in [("passenger", demand.passenger_ratio),
                       ("truck", demand.truck_ratio),
                       ("bus", demand.bus_ratio)]:
        if ratio > 0:
            trip_attributes.append(f'{vt} {ratio}')

    cmd = [
        "python3", random_trips_py,
        "-n", net_path,
        "-o", trip_path,
        "-b", str(sim_config.begin_time),
        "-e", str(sim_config.end_time),
        "-p", str(period),
        "--seed", str(sim_config.seed),
        "--validate",
        "--additional-file", vtype_path,
        "--trip-attributes", f'type="passenger"',
        "-r", output_path,
    ]

    print(f"Running randomTrips.py... (vehicles ~{total_vehicles}, period={period:.2f}s)")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"randomTrips warnings:\n{result.stderr[:500]}")

    if not os.path.exists(output_path):
        raise RuntimeError(f"Route file generation failed:\n{result.stderr[:500]}")

    # 3) Merge vTypes into the route file
    _merge_vtypes_into_routes(vtype_path, output_path, demand)

    # Cleanup
    for tmp in [trip_path, vtype_path]:
        if os.path.exists(tmp):
            os.remove(tmp)
    # Clean up alt file generated by randomTrips
    alt_path = output_path.replace(".rou.xml", ".rou.alt.xml")
    if os.path.exists(alt_path):
        os.remove(alt_path)

    print(f"Route file created: {output_path}")
    return output_path


def _merge_vtypes_into_routes(vtype_path: str, rou_path: str, demand: TrafficDemand):
    """
    Inserts vType definitions at the beginning of the route file
    and reassigns vehicle type attributes according to the specified ratios.
    """
    vtype_tree = ET.parse(vtype_path)
    rou_tree = ET.parse(rou_path)
    rou_root = rou_tree.getroot()

    # Remove existing vTypes (if any)
    for existing_vt in rou_root.findall("vType"):
        rou_root.remove(existing_vt)

    # Insert vTypes at the beginning
    vtypes = list(vtype_tree.getroot().findall("vType"))
    for i, vt in enumerate(vtypes):
        rou_root.insert(i, vt)

    # Reassign vehicle types according to ratios
    import random
    random.seed(42)
    type_choices = []
    for vt, ratio in [("passenger", demand.passenger_ratio),
                       ("truck", demand.truck_ratio),
                       ("bus", demand.bus_ratio)]:
        type_choices.extend([vt] * int(ratio * 100))

    vehicles = rou_root.findall("vehicle")
    for veh in vehicles:
        veh.set("type", random.choice(type_choices))

    ET.indent(rou_tree, space="    ")
    rou_tree.write(rou_path, encoding="utf-8", xml_declaration=True)


def generate_add_xml(
    net_path: str,
    output_path: str,
) -> str:
    """
    Generates an .add.xml file.
    Places detectors (e1Detector) on major edges.
    """
    edges = parse_net_edges(net_path)

    # Place detectors on top edges sorted by length
    main_edges = sorted(edges, key=lambda e: e["length"], reverse=True)[:20]

    root = ET.Element("additional")

    for i, edge in enumerate(main_edges):
        # Place detector on the first lane of the edge
        lane_id = f"{edge['id']}_0"
        pos = min(edge["length"] * 0.5, edge["length"] - 1)  # midpoint

        ET.SubElement(root, "e1Detector",
            id=f"det_{i}",
            lane=lane_id,
            pos=f"{pos:.1f}",
            freq="300",  # 5-minute interval
            file="detector_output.xml",
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"Additional file created: {output_path} ({len(main_edges)} detectors)")
    return output_path


def generate_sumocfg(
    net_path: str,
    rou_path: str,
    add_path: str,
    output_path: str,
    sim_config: SimulationConfig = None,
) -> str:
    """
    Generates a .sumocfg file.
    """
    if sim_config is None:
        sim_config = SimulationConfig()

    # Convert to relative paths
    cfg_dir = os.path.dirname(output_path) or "."
    net_rel = os.path.relpath(net_path, cfg_dir)
    rou_rel = os.path.relpath(rou_path, cfg_dir)
    add_rel = os.path.relpath(add_path, cfg_dir)

    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=net_rel)
    ET.SubElement(inp, "route-files", value=rou_rel)
    ET.SubElement(inp, "additional-files", value=add_rel)

    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin", value=str(sim_config.begin_time))
    ET.SubElement(time, "end", value=str(sim_config.end_time))
    ET.SubElement(time, "step-length", value=str(sim_config.step_length))

    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "time-to-teleport", value="300")

    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-step-log", value="true")

    seed_elem = ET.SubElement(root, "random_number")
    ET.SubElement(seed_elem, "seed", value=str(sim_config.seed))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"Configuration file created: {output_path}")
    return output_path


def generate_all(
    net_path: str,
    output_dir: str = None,
    demand: TrafficDemand = None,
    vtypes: list[VehicleType] = None,
    sim_config: SimulationConfig = None,
) -> dict[str, str]:
    """
    Generates all SUMO files based on a .net.xml file.

    Returns:
        Dictionary of generated file paths
    """
    if output_dir is None:
        output_dir = os.path.dirname(net_path) or "output"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(net_path).replace(".net.xml", "")

    rou_path = os.path.join(output_dir, f"{base_name}.rou.xml")
    add_path = os.path.join(output_dir, f"{base_name}.add.xml")
    cfg_path = os.path.join(output_dir, f"{base_name}.sumocfg")

    generate_rou_xml(net_path, rou_path, demand=demand, vtypes=vtypes, sim_config=sim_config)
    generate_add_xml(net_path, add_path)
    generate_sumocfg(net_path, rou_path, add_path, cfg_path, sim_config=sim_config)

    return {
        "net": net_path,
        "rou": rou_path,
        "add": add_path,
        "cfg": cfg_path,
    }


if __name__ == "__main__":
    import sys
    net_file = sys.argv[1] if len(sys.argv) > 1 else "output/강남역.net.xml"
    result = generate_all(net_file)
    print(f"\nGenerated files:")
    for k, v in result.items():
        print(f"  {k}: {v}")
