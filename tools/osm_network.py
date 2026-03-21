"""
OSM to SUMO Network Conversion Tool

Downloads road network data via the Overpass API,
and generates a SUMO .net.xml file using netconvert.
"""

import os
import subprocess
import urllib.request
import urllib.parse
import json
from dataclasses import dataclass


@dataclass
class BBox:
    """Lat/lon bounding box (south, west, north, east)"""
    south: float
    west: float
    north: float
    east: float

    def to_overpass(self) -> str:
        return f"{self.south},{self.west},{self.north},{self.east}"

    def to_netconvert(self) -> str:
        """netconvert --osm.bounding-box format: west,south,east,north"""
        return f"{self.west},{self.south},{self.east},{self.north}"


# Major area center coordinates (latitude, longitude)
KNOWN_LOCATIONS = {
    "강남역": (37.4979, 127.0276),
    "강남대로": (37.5000, 127.0270),
    "여의도": (37.5219, 126.9245),
    "서울역": (37.5547, 126.9707),
    "홍대입구": (37.5573, 126.9237),
    "잠실역": (37.5133, 127.1001),
    "올림픽대로": (37.5180, 126.9400),
    "테헤란로": (37.5053, 127.0390),
    "방배동": (37.4813, 126.9827),
    "서초동": (37.4920, 127.0090),
}


def geocode_location(location_name: str) -> tuple[float, float]:
    """
    Returns latitude/longitude coordinates for a given location name.
    First searches KNOWN_LOCATIONS, then falls back to the Nominatim API.
    """
    # 1) Search in known locations
    if location_name in KNOWN_LOCATIONS:
        return KNOWN_LOCATIONS[location_name]

    # Partial matching
    for name, coords in KNOWN_LOCATIONS.items():
        if name in location_name or location_name in name:
            return coords

    # 2) Search via Nominatim API
    query = f"{location_name}, 서울, 대한민국"
    url = (
        "https://nominatim.openstreetmap.org/search?"
        + urllib.parse.urlencode({"q": query, "format": "json", "limit": 1})
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SUMO-Agent/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    if not data:
        raise ValueError(f"Could not find location '{location_name}'.")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon


def make_bbox(lat: float, lon: float, radius_m: float = 500) -> BBox:
    """
    Creates a bounding box from center coordinates and radius (in meters).
    """
    # 1 degree of latitude ~ 111,320m, 1 degree of longitude ~ 111,320m * cos(lat)
    import math
    lat_offset = radius_m / 111320
    lon_offset = radius_m / (111320 * math.cos(math.radians(lat)))
    return BBox(
        south=lat - lat_offset,
        west=lon - lon_offset,
        north=lat + lat_offset,
        east=lon + lon_offset,
    )


def download_osm(bbox: BBox, output_path: str) -> str:
    """
    Downloads road network OSM data within the bounding box using the Overpass API.
    """
    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    query = f"""
    [out:xml][timeout:90][bbox:{bbox.to_overpass()}];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|living_street)(_link)?$"];
    (._;>;);
    out body;
    """

    print(f"Downloading OSM data... (bbox: {bbox.to_overpass()})")
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    osm_data = None
    for server in servers:
        try:
            print(f"  Trying server: {server}")
            req = urllib.request.Request(server, data=data, headers={"User-Agent": "SUMO-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                osm_data = resp.read()
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if osm_data is None:
        raise RuntimeError("Download failed from all Overpass servers. Please check your network connection.")

    with open(output_path, "wb") as f:
        f.write(osm_data)

    size_kb = len(osm_data) / 1024
    print(f"OSM data saved: {output_path} ({size_kb:.1f} KB)")
    return output_path


def convert_osm_to_net(
    osm_path: str,
    net_path: str,
    netconvert_bin: str = "netconvert",
) -> str:
    """
    Converts an OSM file to a SUMO network file using netconvert.
    """
    cmd = [
        netconvert_bin,
        "--osm-files", osm_path,
        "--output-file", net_path,
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--tls.default-type", "actuated",
        "--edges.join",
        "--no-turnarounds",
        "--proj.utm",
    ]

    print(f"Running netconvert...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"netconvert warnings/errors:\n{result.stderr[:500]}")

    if not os.path.exists(net_path):
        raise RuntimeError(f"Network file generation failed: {result.stderr}")

    size_kb = os.path.getsize(net_path) / 1024
    print(f"Network file created: {net_path} ({size_kb:.1f} KB)")
    return net_path


def build_network(location_name: str, radius_m: float = 500, output_dir: str = "output") -> str:
    """
    Main function that generates a SUMO network file from a location name.

    Args:
        location_name: Location name (e.g., "강남역", "테헤란로")
        radius_m: Radius from center (in meters)
        output_dir: Output directory

    Returns:
        Path to the generated .net.xml file
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Look up coordinates
    lat, lon = geocode_location(location_name)
    print(f"Location: {location_name} -> ({lat:.4f}, {lon:.4f})")

    # 2) Create bounding box
    bbox = make_bbox(lat, lon, radius_m)

    # 3) Download OSM data
    safe_name = location_name.replace(" ", "_")
    osm_path = os.path.join(output_dir, f"{safe_name}.osm.xml")
    net_path = os.path.join(output_dir, f"{safe_name}.net.xml")

    try:
        download_osm(bbox, osm_path)
    except Exception as e:
        if os.path.exists(osm_path):
            print(f"OSM download failed -> using existing OSM cache: {e}")
        elif os.path.exists(net_path):
            print(f"OSM download failed -> using existing network cache: {e}")
            return net_path
        else:
            raise

    # 4) Convert with netconvert
    from src.config import get_netconvert_bin
    convert_osm_to_net(osm_path, net_path, netconvert_bin=get_netconvert_bin())

    return net_path


if __name__ == "__main__":
    import sys
    location = sys.argv[1] if len(sys.argv) > 1 else "강남역"
    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 500
    result = build_network(location, radius)
    print(f"\nDone: {result}")
