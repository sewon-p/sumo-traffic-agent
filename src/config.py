"""
Project configuration management

Auto-discovers environment variables and SUMO binary paths.
Defers binary requirements until runtime so that module imports
succeed even in environments where SUMO is not installed.
"""

import os
import shutil


def _find_binary(name: str) -> str:
    """Find the executable path. Returns an empty string if not found."""
    # 1) Environment variable
    env_key = f"{name.upper()}_BIN"
    if os.environ.get(env_key):
        return os.environ[env_key]

    # 2) Project venv (highest priority)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_bin = os.path.join(project_root, ".venv", "bin", name)
    if os.path.isfile(project_bin):
        return project_bin

    # 3) System PATH and common locations
    candidates = [
        shutil.which(name),
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    return ""


def _require_binary(path: str, name: str) -> str:
    """Raise an explicit error if a required binary is missing."""
    if path:
        return path

    env_key = f"{name.upper()}_BIN"
    raise FileNotFoundError(
        f"'{name}' not found. "
        f"Install SUMO or set the {env_key} environment variable."
    )


def get_sumo_bin(required: bool = True) -> str:
    path = _find_binary("sumo")
    return _require_binary(path, "sumo") if required else path


def get_netconvert_bin(required: bool = True) -> str:
    path = _find_binary("netconvert")
    return _require_binary(path, "netconvert") if required else path


def get_sumo_gui_bin(required: bool = True) -> str:
    sumo_path = get_sumo_bin(required=required)
    if not sumo_path:
        return ""
    gui_path = sumo_path.replace("/sumo", "/sumo-gui")
    if os.path.isfile(gui_path):
        return gui_path
    return _require_binary(_find_binary("sumo-gui"), "sumo-gui") if required else _find_binary("sumo-gui")


# SUMO binary paths
SUMO_BIN = get_sumo_bin(required=False)
NETCONVERT_BIN = get_netconvert_bin(required=False)

# API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TOPIS_API_KEY = os.environ.get("TOPIS_API_KEY", "")
DATA_GO_KR_API_KEY = os.environ.get("DATA_GO_KR_API_KEY", "")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
