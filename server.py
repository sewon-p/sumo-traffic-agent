#!/usr/bin/env python3
"""
Local web server - interactive simulation UI backend

Streams simulation progress in real-time via SSE (Server-Sent Events).

Run: python server.py
Access: http://localhost:8080
"""

import http.server
import importlib.util
import json
import os
import sys
import re
import subprocess
import threading
import urllib.parse
import tempfile
import uuid
from io import BytesIO

# Re-exec under project venv if available (skipped in Docker / production)
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")

if (
    __name__ == "__main__"
    and os.path.exists(VENV_PYTHON)
    and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON)
    and os.environ.get("SUMO_SERVER_SKIP_VENV_REEXEC") != "1"
):
    os.execve(
        VENV_PYTHON,
        [VENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]],
        {**os.environ, "SUMO_SERVER_SKIP_VENV_REEXEC": "1"},
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add venv site-packages to path if running locally (for sumolib, etc.)
import glob
_venv_sp = glob.glob(os.path.join(os.path.dirname(__file__), ".venv", "lib", "python*", "site-packages"))
if _venv_sp:
    sys.path.insert(0, _venv_sp[0])

# Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.config import SUMO_BIN, NETCONVERT_BIN, get_sumo_bin, get_sumo_gui_bin, get_netconvert_bin
from src.llm_client import detect_provider

PORT = 8080
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Session state: last simulation info (for modification requests)
_session = {
    "output_dir": None,
    "net_path": None,
    "params": None,
    "ft": {},
    "history": [],  # Conversation history
    "pending_modification": None,
    "sim_stats": None,
}


def _snapshot_state(params, ft):
    """Serialize current parameter/FT state for retraining."""
    return {
        "params": {
            "location": params.location,
            "radius_m": params.radius_m,
            "time_start": params.time_start,
            "time_end": params.time_end,
            "vehicles_per_hour": params.vehicles_per_hour,
            "speed_limit_kmh": params.speed_limit_kmh,
            "weather": params.weather,
            "incident": params.incident,
            "lane_closure": params.lane_closure,
        },
        "ft": {
            "speed_kmh": ft.get("speed_kmh"),
            "sigma": ft.get("sigma"),
            "tau": ft.get("tau"),
            "lanes": ft.get("lanes"),
            "avg_block_m": ft.get("avg_block_m"),
            "reasoning": ft.get("reasoning", ""),
        },
    }


def _apply_parameter_changes(params, ft, changes):
    """Apply modified parameters and return before/after snapshots."""
    before = _snapshot_state(params, ft)

    if "volume_vph" in changes:
        params.vehicles_per_hour = int(changes["volume_vph"])
    if "speed_limit_kmh" in changes:
        params.speed_limit_kmh = float(changes["speed_limit_kmh"])
    if "avg_block_m" in changes:
        ft["avg_block_m"] = float(changes["avg_block_m"])
    if "sigma" in changes:
        ft["sigma"] = float(changes["sigma"])
    if "tau" in changes:
        ft["tau"] = float(changes["tau"])

    after = _snapshot_state(params, ft)
    return before, after


def _ft_runtime_status():
    ft_model = os.environ.get("OPENAI_FT_MODEL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_ready = importlib.util.find_spec("openai") is not None
    return {
        "configured": bool(ft_model and api_key),
        "openai_installed": openai_ready,
        "ready": bool(ft_model and api_key and openai_ready),
        "ft_model": ft_model,
    }


def _snapshot_xml_state():
    return {
        "nod_path": _session.get("nod_path"),
        "edg_path": _session.get("edg_path"),
    }


def _compute_final_parameter_changes(before_state, after_state):
    changes = {}
    before_params = (before_state or {}).get("params", {})
    after_params = (after_state or {}).get("params", {})
    before_ft = (before_state or {}).get("ft", {})
    after_ft = (after_state or {}).get("ft", {})

    if before_params.get("vehicles_per_hour") != after_params.get("vehicles_per_hour"):
        changes["volume_vph"] = after_params.get("vehicles_per_hour")
    if before_params.get("speed_limit_kmh") != after_params.get("speed_limit_kmh"):
        changes["speed_limit_kmh"] = after_params.get("speed_limit_kmh")
    for field in ("sigma", "tau", "lanes", "avg_block_m", "speed_kmh"):
        if before_ft.get(field) != after_ft.get(field):
            changes[field] = after_ft.get(field)
    return changes


def _start_pending_modification(intent):
    pending = {
        "session_id": str(uuid.uuid4()),
        "started_at": datetime.now().isoformat(),
        "sim_id": _session.get("sim_id"),
        "intent": intent,
        "base_state": _snapshot_state(_session["params"], _session["ft"]),
        "base_xml": _snapshot_xml_state(),
        "events": [],
    }
    _session["pending_modification"] = pending
    return pending


def _record_pending_modification_event(user_input, modification_type, modification_details):
    pending = _session.get("pending_modification")
    if not pending:
        pending = _start_pending_modification((modification_details or {}).get("intent"))

    pending["events"].append({
        "user_input": user_input,
        "modification_type": modification_type,
        "details": modification_details,
        "recorded_at": datetime.now().isoformat(),
    })


def _finalize_pending_modification():
    pending = _session.get("pending_modification")
    if not pending or not pending.get("events") or not pending.get("sim_id"):
        _session["pending_modification"] = None
        return None

    from src.session_db import save_modification

    before_state = pending.get("base_state") or {}
    after_state = _snapshot_state(_session["params"], _session["ft"])
    before_xml = pending.get("base_xml") or {}
    after_xml = _snapshot_xml_state()
    modification_types = {e.get("modification_type") for e in pending["events"] if e.get("modification_type")}
    final_type = modification_types.pop() if len(modification_types) == 1 else "mixed"
    last_prompt = pending["events"][-1]["user_input"]
    sim_speed = ((_session.get("sim_stats") or {}).get("avg_speed_kmh"))

    details = {
        "kind": final_type,
        "intent": pending.get("intent"),
        "session_id": pending.get("session_id"),
        "started_at": pending.get("started_at"),
        "ended_at": datetime.now().isoformat(),
        "event_count": len(pending["events"]),
        "events": pending["events"],
        "before": before_state,
        "after": after_state,
        "changes": _compute_final_parameter_changes(before_state, after_state),
        "xml": {
            "old_nod_path": before_xml.get("nod_path"),
            "old_edg_path": before_xml.get("edg_path"),
            "new_nod_path": after_xml.get("nod_path"),
            "new_edg_path": after_xml.get("edg_path"),
        },
    }

    save_modification(
        pending["sim_id"],
        last_prompt,
        final_type,
        json.dumps(before_state, ensure_ascii=False),
        json.dumps(after_state, ensure_ascii=False),
        sim_speed,
        modification_type=final_type,
        edit_intent=pending.get("intent"),
        trainable=(pending.get("intent") == "correction"),
        details=details,
    )
    _session["pending_modification"] = None
    return {
        "status": "success",
        "saved": True,
        "session_id": pending.get("session_id"),
        "event_count": len(pending["events"]),
        "modification_type": final_type,
        "intent": pending.get("intent"),
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/status":
            self.send_json(self._get_status())

        elif parsed.path == "/api/admin/summary":
            self._handle_admin_summary()

        elif parsed.path == "/api/admin/simulations":
            params = urllib.parse.parse_qs(parsed.query)
            limit = int(params.get("limit", ["100"])[0])
            self._handle_admin_simulations(limit)

        elif parsed.path == "/api/admin/modifications":
            params = urllib.parse.parse_qs(parsed.query)
            limit = int(params.get("limit", ["200"])[0])
            self._handle_admin_modifications(limit)

        elif parsed.path == "/api/admin/llm-eval":
            self._handle_admin_llm_eval()

        elif parsed.path == "/api/admin/export/corrections":
            self._handle_admin_export_corrections()

        elif parsed.path == "/api/admin/export/corrections/download":
            self._handle_admin_export_corrections_download()

        elif parsed.path == "/api/admin/report/evaluation":
            self._handle_admin_report_evaluation()

        elif parsed.path == "/api/admin/report/evaluation/download":
            self._handle_admin_report_evaluation_download()

        elif parsed.path == "/api/admin/report/llm-evaluation/download":
            self._handle_admin_report_llm_evaluation_download()

        elif parsed.path == "/api/traffic":
            params = urllib.parse.parse_qs(parsed.query)
            location = params.get("location", ["강남역"])[0]
            self._handle_traffic(location)

        elif parsed.path == "/api/network":
            params = urllib.parse.parse_qs(parsed.query)
            net_path = params.get("path", [None])[0]
            location = params.get("location", [None])[0]
            self._handle_network_geometry(net_path, location)

        elif parsed.path == "/api/view":
            params = urllib.parse.parse_qs(parsed.query)
            cfg_path = params.get("path", [None])[0]
            location = params.get("location", [None])[0]
            self._handle_sumo_gui(cfg_path, location)

        elif parsed.path == "/api/download":
            self._handle_download()

        elif parsed.path == "/admin":
            self.path = "/admin.html"
            super().do_GET()

        elif parsed.path == "/about":
            self.path = "/about.html"
            super().do_GET()

        elif parsed.path == "/" or parsed.path == "/index.html":
            # Prevent caching: no-cache header
            self.path = "/index.html"
            super().do_GET()

        else:
            super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def do_POST(self):
        if self.path == "/api/simulate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            user_input = body.get("input", "")
            provider = body.get("provider", None)
            modify_intent = body.get("modify_intent", None)
            api_key = body.get("api_key", None)
            base_llm = body.get("base_llm", "default")
            self._handle_simulate_sse(user_input, provider=provider, modify_intent=modify_intent, api_key=api_key, base_llm=base_llm)
        else:
            self.send_error(404)

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_sse(self, event_data):
        line = f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        self.wfile.write(line.encode())
        self.wfile.flush()

    def _get_status(self):
        import shutil
        provider, key = detect_provider()
        clis = [n for n in ["gemini", "claude", "codex"] if shutil.which(n)]
        mode = "API" if key and key != "CLI" else ("CLI" if clis else "")

        ft_status = _ft_runtime_status()
        ft_model = ft_status["ft_model"]
        has_ft = ft_status["ready"]

        return {
            "sumo": bool(SUMO_BIN and os.path.isfile(SUMO_BIN)),
            "llm": has_ft or bool(provider),
            "llm_name": f"Fine-tuned ({ft_model.split(':')[-1]})" if has_ft else (f"{provider} ({mode})" if provider else "Rule-based"),
            "llm_provider": "openai-ft" if has_ft else (provider or ""),
            "llm_mode": "Fine-tuned" if has_ft else mode,
            "available_cli": clis,
            "traffic_data": bool(os.environ.get("TOPIS_API_KEY")),
            "ft_configured": ft_status["configured"],
            "ft_ready": ft_status["ready"],
            "python_executable": sys.executable,
        }

    def _handle_network_geometry(self, net_path, location):
        """Return network edge coordinates as JSON (for Canvas rendering)."""
        if not net_path and location:
            safe = location.replace(" ", "_")
            net_path = os.path.join("output", safe, f"{safe}.net.xml")

        if not net_path or not os.path.exists(net_path):
            self.send_json({"error": "Network file not found", "path": net_path})
            return

        try:
            sys.path.insert(0, os.path.join(
                os.path.dirname(__file__), ".venv", "lib"))
            import sumolib
            net = sumolib.net.readNet(net_path)
        except ImportError:
            # If sumolib is not available, parse XML directly
            self.send_json(self._parse_net_xml_fallback(net_path))
            return

        edges_data = []
        for edge in net.getEdges():
            if edge.getID().startswith(":"):
                continue
            shape = edge.getShape()
            edges_data.append({
                "id": edge.getID(),
                "name": edge.getName() or "",
                "lanes": edge.getLaneNumber(),
                "length": round(edge.getLength(), 1),
                "speed_limit": round(edge.getSpeed() * 3.6, 1),
                "points": [[round(x, 1), round(y, 1)] for x, y in shape],
            })

        junctions_data = []
        for junc in net.getNodes():
            if junc.getType() == "internal":
                continue
            x, y = junc.getCoord()
            junctions_data.append({
                "id": junc.getID(),
                "x": round(x, 1),
                "y": round(y, 1),
                "type": junc.getType(),
            })

        bbox = net.getBoundary()
        self.send_json({
            "status": "success",
            "bbox": {"x_min": bbox[0], "y_min": bbox[1],
                     "x_max": bbox[2], "y_max": bbox[3]},
            "edges": edges_data,
            "junctions": junctions_data,
            "stats": {
                "edge_count": len(edges_data),
                "junction_count": len(junctions_data),
                "road_names": sorted(set(e["name"] for e in edges_data if e["name"]))[:20],
            },
        })

    def _parse_net_xml_fallback(self, net_path):
        """Fallback XML parsing when sumolib is not available."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(net_path)
        root = tree.getroot()
        edges = []
        for edge in root.findall("edge"):
            eid = edge.get("id", "")
            if eid.startswith(":"):
                continue
            lanes = edge.findall("lane")
            if not lanes:
                continue
            shape_str = lanes[0].get("shape", "")
            points = []
            for pair in shape_str.split():
                parts = pair.split(",")
                if len(parts) == 2:
                    points.append([round(float(parts[0]), 1), round(float(parts[1]), 1)])
            edges.append({
                "id": eid,
                "name": edge.get("name", ""),
                "lanes": len(lanes),
                "length": round(float(lanes[0].get("length", 0)), 1),
                "points": points,
            })
        return {"status": "success", "edges": edges, "junctions": [],
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 1000, "y_max": 1000},
                "stats": {"edge_count": len(edges)}}

    def _handle_sumo_gui(self, cfg_path, location):
        """Launch sumo-gui."""
        if not cfg_path and location:
            safe = location.replace(" ", "_")
            cfg_path = os.path.join("output", safe, f"{safe}.sumocfg")

        if not cfg_path or not os.path.exists(cfg_path):
            self.send_json({"error": "Configuration file not found", "path": cfg_path})
            return

        sumo_gui = get_sumo_gui_bin()
        subprocess.Popen([sumo_gui, "-c", cfg_path])
        self.send_json({"status": "launched", "path": cfg_path})

    def _handle_traffic(self, location):
        from tools.topis_api import query_realtime_traffic
        result = query_realtime_traffic(location)
        self.send_json(result)

    def _handle_admin_summary(self):
        from src.session_db import build_evaluation_summary
        self.send_json(build_evaluation_summary())

    def _handle_admin_simulations(self, limit):
        from src.session_db import list_simulations
        self.send_json({"items": list_simulations(limit=limit)})

    def _handle_admin_modifications(self, limit):
        from src.session_db import list_modifications
        self.send_json({"items": list_modifications(limit=limit)})

    def _handle_admin_llm_eval(self):
        from src.session_db import build_llm_evaluation_summary
        self.send_json(build_llm_evaluation_summary())

    def _handle_admin_export_corrections(self):
        from src.session_db import export_corrections_for_training
        path = export_corrections_for_training()
        self.send_json({"status": "success", "path": path})

    def _handle_admin_export_corrections_download(self):
        from src.session_db import export_corrections_for_training
        with tempfile.NamedTemporaryFile(prefix="corrections_", suffix=".jsonl", delete=False) as tmp:
            temp_path = tmp.name
        path = export_corrections_for_training(temp_path)
        self._send_file_download(path, "application/jsonl", os.path.basename(path))

    def _handle_admin_report_evaluation(self):
        from src.session_db import export_evaluation_report
        path = export_evaluation_report()
        self.send_json({"status": "success", "path": path})

    def _handle_admin_report_evaluation_download(self):
        from src.session_db import export_evaluation_report
        with tempfile.NamedTemporaryFile(prefix="evaluation_", suffix=".txt", delete=False) as tmp:
            temp_path = tmp.name
        path = export_evaluation_report(temp_path)
        self._send_file_download(path, "text/plain; charset=utf-8", os.path.basename(path))

    def _handle_admin_report_llm_evaluation_download(self):
        from src.session_db import export_llm_evaluation_report
        with tempfile.NamedTemporaryFile(prefix="llm_evaluation_", suffix=".txt", delete=False) as tmp:
            temp_path = tmp.name
        path = export_llm_evaluation_report(temp_path)
        self._send_file_download(path, "text/plain; charset=utf-8", os.path.basename(path))

    def _send_file_download(self, path, content_type, download_name=None):
        if not path or not os.path.exists(path):
            self.send_json({"error": "File does not exist.", "path": path})
            return

        with open(path, "rb") as f:
            data = f.read()

        filename = download_name or os.path.basename(path)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _handle_download(self):
        """Download the last simulation output folder as a zip file."""
        import zipfile
        output_dir = _session.get("output_dir")
        if not output_dir or not os.path.isdir(output_dir):
            self.send_json({"error": "No simulation available for download."})
            return

        zip_path = output_dir.rstrip("/") + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    arcname = os.path.relpath(fpath, os.path.dirname(output_dir))
                    zf.write(fpath, arcname)

        zip_name = os.path.basename(zip_path)
        with open(zip_path, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{zip_name}"')
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _handle_simulate_sse(self, user_input, provider=None, modify_intent=None, api_key=None, base_llm="default"):
        """Stream simulation progress via SSE."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            self._run_simulation(user_input, provider=provider, modify_intent=modify_intent, api_key=api_key, base_llm=base_llm)
        except Exception as e:
            self.send_sse({"type": "error", "text": str(e)})

    def _run_simulation(self, user_input, provider=None, modify_intent=None, api_key=None, base_llm="default"):
        """
        Full pipeline:
        1) If modification request, modify existing session parameters
        2) If new request, determine parameters with fine-tuned model
        3) Build/reuse road network
        4) Run SUMO + validate
        """
        from src.llm_parser import parse_user_input, SimulationParams, get_prompt_metadata
        from tools.osm_network import build_network
        from src.base_llm import generate_network_xml, modify_network_xml
        from tools.sumo_generator import generate_all, TrafficDemand, SimulationConfig
        from src.validator import validate
        from datetime import datetime

        # Set base LLM mode (reflect the value selected from the frontend)
        if base_llm.startswith("cli:"):
            os.environ["BASE_LLM_MODE"] = "cli"
            os.environ["BASE_LLM_CLI"] = base_llm.split(":")[1]
        elif base_llm.startswith("api:"):
            parts = base_llm.split(":")
            os.environ["BASE_LLM_MODE"] = "api"
            # Temporarily set user-provided key
            if len(parts) >= 3:
                provider_name = parts[1]
                key = ":".join(parts[2:])
                os.environ["BASE_LLM_CUSTOM_KEY"] = key
                os.environ["BASE_LLM_CUSTOM_PROVIDER"] = provider_name
        else:
            os.environ["BASE_LLM_MODE"] = "api"

        # -- Modification mode: frontend sends provider="modify" --
        is_modify = False
        is_param_only = False
        modification_type = ""
        modification_details = None
        modify_intent = modify_intent or ""
        if provider == "modify" and not (_session["net_path"] and _session["params"]):
            self.send_sse({"type": "error", "text": "No previous simulation to modify. Please run a simulation first."})
            return

        if provider == "modify" and _session["net_path"] and _session["params"]:
            is_modify = True
            intent_label = "Correction" if modify_intent == "correction" else "Alternative request" if modify_intent == "alternative" else "Modification"
            self.send_sse({"type": "message", "text": f"{intent_label} of previous simulation..."})
            self.send_sse({"type": "message", "text": "Determining modification type..."})
            params = _session["params"]
            ft = _session["ft"]

            # Ask base LLM to classify geometry vs parameter modification
            from src.base_llm import classify_modification, modify_parameters, extract_ft_training_hints
            mod_type = classify_modification(user_input)
            modification_type = mod_type
            mod_label = {"geometry": "Geometry", "parameter": "Parameter", "mixed": "Mixed modification"}.get(mod_type, mod_type)
            self.send_sse({"type": "message", "text": f"Modification type: {mod_label}"})

            if mod_type in {"parameter", "mixed"}:
                parameter_applied = False
                current_params = json.dumps({
                    "volume_vph": params.vehicles_per_hour,
                    "speed_limit_kmh": params.speed_limit_kmh,
                    "sigma": ft.get("sigma"),
                    "tau": ft.get("tau"),
                }, ensure_ascii=False)
                changes = modify_parameters(user_input, current_params)
                if changes:
                    before_state, after_state = _apply_parameter_changes(params, ft, changes)
                    modification_details = {
                        "kind": "parameter" if mod_type == "parameter" else "mixed",
                        "intent": modify_intent,
                        "changes": changes,
                        "before": before_state,
                        "after": after_state,
                        "parameter_applied": True,
                    }
                    parameter_applied = True
                    self.send_sse({"type": "message", "text": f"Parameter modification: {json.dumps(changes, ensure_ascii=False)}"})
                    if mod_type == "parameter":
                        is_param_only = True
                else:
                    parameter_applied = False
                    if mod_type == "parameter":
                        self.send_sse({"type": "error", "text": "Failed to interpret parameter modification. Please rephrase within the same modification context."})
                        return
                    self.send_sse({"type": "message", "text": "Skipping parameter modification, continuing with geometry modification only."})

            if mod_type in {"geometry", "mixed"}:
                geometry_applied = False
                ft_train_hints = extract_ft_training_hints(user_input, json.dumps({
                    "lanes": ft.get("lanes"),
                    "avg_block_m": ft.get("avg_block_m"),
                    "speed_limit_kmh": params.speed_limit_kmh,
                    "sigma": ft.get("sigma"),
                    "tau": ft.get("tau"),
                }, ensure_ascii=False))
                # Geometry modification -- XML editing
                nod_path = _session.get("nod_path", "")
                edg_path = _session.get("edg_path", "")
                if nod_path and edg_path and os.path.exists(nod_path) and os.path.exists(edg_path):
                    with open(nod_path) as f:
                        current_nod = f.read()
                    with open(edg_path) as f:
                        current_edg = f.read()
                    try:
                        self.send_sse({"type": "tool", "name": "modify_xml", "summary": "LLM is modifying network XML..."})
                        new_nod, new_edg = modify_network_xml(user_input, current_nod, current_edg)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = os.path.join("output", f"modified_{ts}")
                        os.makedirs(output_dir, exist_ok=True)
                        new_nod_path = os.path.join(output_dir, "modified.nod.xml")
                        new_edg_path = os.path.join(output_dir, "modified.edg.xml")
                        with open(new_nod_path, "w") as f:
                            f.write(new_nod)
                        with open(new_edg_path, "w") as f:
                            f.write(new_edg)
                        geometry_details = {
                            "kind": "geometry" if mod_type == "geometry" else "mixed",
                            "intent": modify_intent,
                            "before": modification_details.get("before", _snapshot_state(params, ft)) if modification_details else _snapshot_state(params, ft),
                            "after": _snapshot_state(params, ft),
                            "xml": {
                                "old_nod_path": nod_path,
                                "old_edg_path": edg_path,
                                "new_nod_path": new_nod_path,
                                "new_edg_path": new_edg_path,
                            },
                            "ft_train_fields": ft_train_hints,
                            "geometry_applied": True,
                        }
                        geometry_applied = True
                        if modification_details:
                            modification_details.update({
                                "kind": "mixed",
                                "after": geometry_details["after"],
                                "xml": geometry_details["xml"],
                                "ft_train_fields": ft_train_hints,
                                "geometry_applied": True,
                            })
                        else:
                            modification_details = geometry_details
                        self.send_sse({"type": "message", "text": "XML modification complete"})
                    except Exception as e:
                        self.send_sse({"type": "error", "text": f"XML modification failed: {e}"})
                        return
                else:
                    self.send_sse({"type": "error", "text": "No previous XML available for modification. Please start a new simulation."})
                    return

            if mod_type == "mixed":
                if modification_details is None:
                    modification_details = {"kind": "mixed", "intent": modify_intent}
                modification_details["result_status"] = (
                    "success" if modification_details.get("parameter_applied") and modification_details.get("geometry_applied")
                    else "partial_success" if modification_details.get("parameter_applied") or modification_details.get("geometry_applied")
                    else "failed"
                )

        if not is_modify:
            # -- 1) Extract parameters with fine-tuned model --
            self.send_sse({"type": "message", "text": "Extracting parameters..."})
            params = parse_user_input(user_input)

            ft = {}
            if params.notes:
                try:
                    ft = json.loads(params.notes)
                except:
                    pass

        self.send_sse({"type": "params", "data": {
            "location": params.location or "(virtual road)",
            "time": f"{params.time_start}~{params.time_end}",
            "speed_kmh": ft.get("speed_kmh", "-"),
            "volume_vph": params.vehicles_per_hour,
            "lanes": ft.get("lanes", "-"),
            "speed_limit_kmh": params.speed_limit_kmh,
            "sigma": ft.get("sigma", "-"),
            "tau": ft.get("tau", "-"),
            "avg_block_m": ft.get("avg_block_m", "-"),
            "reasoning": ft.get("reasoning", ""),
            "source": modification_type if is_modify else ("fine-tuned" if ft else "rule-based"),
        }})

        # -- 2) Road network --
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        loc_name = (params.location or "generated").replace(" ", "_")
        output_dir = os.path.join("output", f"{loc_name}_{ts}")
        os.makedirs(output_dir, exist_ok=True)
        net_path = None

        # Parameter-only modification -> reuse existing network
        if is_modify and is_param_only and _session.get("net_path") and os.path.exists(_session["net_path"]):
            net_path = _session["net_path"]
            self.send_sse({"type": "message", "text": "Reusing road network (parameters only changed)"})

        # Geometry modification: if LLM modified XML, run netconvert
        elif is_modify and 'new_nod_path' in dir() and os.path.exists(new_nod_path):
            net_path = os.path.join(output_dir, "modified.net.xml")
            cmd = [get_netconvert_bin(), "--node-files", new_nod_path,
                   "--edge-files", new_edg_path, "--output-file", net_path,
                   "--no-turnarounds", "true"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if os.path.exists(net_path):
                _session["nod_path"] = new_nod_path
                _session["edg_path"] = new_edg_path
            else:
                err_msg = result.stderr[:200] if result.stderr else "Unknown error"
                self.send_sse({"type": "message", "text": f"netconvert failed: {err_msg}\nRetrying XML generation..."})
                # On failure, retry: include error message and ask LLM again
                try:
                    with open(new_nod_path) as f:
                        bad_nod = f.read()
                    with open(new_edg_path) as f:
                        bad_edg = f.read()
                    retry_input = f"{user_input}\n\nnetconvert error occurred with previous XML: {err_msg}\nPlease fix and regenerate."
                    new_nod2, new_edg2 = modify_network_xml(retry_input, bad_nod, bad_edg)
                    with open(new_nod_path, "w") as f:
                        f.write(new_nod2)
                    with open(new_edg_path, "w") as f:
                        f.write(new_edg2)
                    result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if os.path.exists(net_path):
                        _session["nod_path"] = new_nod_path
                        _session["edg_path"] = new_edg_path
                        self.send_sse({"type": "message", "text": "Retry succeeded"})
                    else:
                        self.send_sse({"type": "error", "text": f"Retry also failed: {result2.stderr[:100]}"})
                        return
                except Exception as e:
                    self.send_sse({"type": "error", "text": f"Retry failed: {e}"})
                    return

        if not net_path:
            if params.location:
                self.send_sse({"type": "tool", "name": "osm_network",
                               "summary": f"{params.location} OSM download"})
                try:
                    net_path = build_network(params.location, params.radius_m, output_dir)
                except Exception as e:
                    self.send_sse({"type": "message",
                                   "text": f"OSM failed -> LLM road generation: {e}"})

            if not net_path:
                # LLM generates XML directly
                self.send_sse({"type": "tool", "name": "generate_xml",
                               "summary": "LLM is generating road network XML..."})
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    nod_xml, edg_xml = generate_network_xml(user_input, {
                        "lanes": ft.get("lanes", 2),
                        "speed_limit_kmh": params.speed_limit_kmh,
                        "avg_block_m": ft.get("avg_block_m", 200),
                    })
                    nod_path = os.path.join(output_dir, f"{loc_name}.nod.xml")
                    edg_path = os.path.join(output_dir, f"{loc_name}.edg.xml")
                    with open(nod_path, "w") as f:
                        f.write(nod_xml)
                    with open(edg_path, "w") as f:
                        f.write(edg_xml)
                    # netconvert
                    net_path = os.path.join(output_dir, f"{loc_name}.net.xml")
                    cmd = [get_netconvert_bin(), "--node-files", nod_path,
                           "--edge-files", edg_path, "--output-file", net_path,
                           "--no-turnarounds", "true"]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if not os.path.exists(net_path):
                        raise RuntimeError("netconvert failed")
                    _session["nod_path"] = nod_path
                    _session["edg_path"] = edg_path
                except Exception as e:
                    self.send_sse({"type": "error", "text": f"Road generation failed: {e}"})
                    return

        self._send_network_info(net_path)

        # -- 3) Generate SUMO files + run simulation --
        self.send_sse({"type": "tool", "name": "sumo_generate",
                       "summary": f"{params.vehicles_per_hour} veh/h simulation setup"})
        demand = TrafficDemand(
            total_vehicles_per_hour=params.vehicles_per_hour,
            passenger_ratio=params.passenger_ratio,
            truck_ratio=params.truck_ratio,
            bus_ratio=params.bus_ratio,
        )
        h1 = int(params.time_start.split(":")[0])
        h2 = int(params.time_end.split(":")[0])
        evaluation_duration = max((h2 - h1) * 3600, 3600)
        warmup_seconds = max(int(os.environ.get("SIM_WARMUP_SECONDS", "600") or 0), 0)
        total_duration = evaluation_duration + warmup_seconds
        if warmup_seconds:
            self.send_sse({"type": "message", "text": (
                f"Simulation time setup: warmup {warmup_seconds // 60} min + "
                f"evaluation {evaluation_duration // 60} min"
            )})
        files = generate_all(net_path, output_dir,
                             demand=demand,
                             sim_config=SimulationConfig(
                                 begin_time=0,
                                 end_time=total_duration,
                                 warmup_seconds=warmup_seconds,
                             ))

        self.send_sse({"type": "tool", "name": "sumo_run",
                       "summary": "Running SUMO simulation..."})
        sim_stats = self._run_sumo(files["cfg"], warmup_seconds=warmup_seconds)
        sim_stats["evaluation_duration_s"] = evaluation_duration
        sim_stats["total_duration_s"] = total_duration
        self.send_sse({"type": "result", "data": sim_stats})

        # -- 4) Validation: FT prediction vs SUMO result --
        ft_speed = ft.get("speed_kmh") if ft else None
        if ft_speed and sim_stats.get("avg_speed_kmh"):
            v = validate(sim_stats, real_speed_kmh=ft_speed,
                         real_volume_vph=params.vehicles_per_hour)
            self.send_sse({"type": "validation", "data": {
                "grade": v.grade,
                "speed_error_pct": v.speed_error_pct,
                "ft_speed": ft_speed,
                "sim_speed": sim_stats["avg_speed_kmh"],
                "issues": v.issues,
            }})
            self.send_sse({"type": "message", "text": (
                f"\nValidation: FT prediction {ft_speed}km/h vs SUMO {sim_stats['avg_speed_kmh']}km/h"
                f" -> error {v.speed_error_pct:+.1f}% (grade {v.grade})"
            )})

        self.send_sse({"type": "message", "text": f"\nFiles saved: {output_dir}/"})

        # -- DB save --
        from src.session_db import save_simulation, save_modification
        sim_speed = sim_stats.get("avg_speed_kmh")
        ft_speed_val = ft.get("speed_kmh") if ft else None
        err = None
        grd = None
        if ft_speed_val and sim_speed:
            err = round((sim_speed - ft_speed_val) / ft_speed_val * 100, 1)
            grd = "A" if abs(err) < 10 else "B" if abs(err) < 20 else "C" if abs(err) < 30 else "D" if abs(err) < 50 else "F"

        if is_modify and _session.get("sim_id"):
            from src.session_db import update_simulation_params
            save_modification(
                _session["sim_id"], user_input,
                modification_type or "modify",
                json.dumps(modification_details.get("before", {}), ensure_ascii=False) if modification_details else "",
                json.dumps(modification_details.get("after", {}), ensure_ascii=False) if modification_details else "",
                sim_speed,
                modification_type=modification_type or "modify",
                edit_intent=modify_intent or None,
                trainable=(modify_intent == "correction"),
                details=modification_details,
            )
            # Update simulation record with final parameters
            update_simulation_params(
                _session["sim_id"],
                {"location": params.location, "vehicles_per_hour": params.vehicles_per_hour,
                 "speed_limit_kmh": params.speed_limit_kmh},
                ft, sim_speed, err, grd,
            )
        else:
            params_dict = {
                "location": params.location, "time_start": params.time_start,
                "time_end": params.time_end, "vehicles_per_hour": params.vehicles_per_hour,
                "speed_limit_kmh": params.speed_limit_kmh,
            }
            _session["sim_id"] = save_simulation(
                user_input, params_dict, ft, net_type if 'net_type' in dir() else "osm",
                output_dir, sim_speed, ft_speed_val, err, grd,
                prompt_meta=get_prompt_metadata(ft_used=bool(ft)),
            )

        # -- Save session --
        _session["output_dir"] = output_dir
        _session["net_path"] = net_path
        _session["params"] = params
        _session["ft"] = ft

        # -- Buttons: Adjust / New Simulation / Download --
        self.send_sse({"type": "buttons", "data": [
            {"label": "Adjust", "action": "modify-menu"},
            {"label": "New Simulation", "action": "new"},
            {"label": "Download", "action": "download"},
            {"label": "Correction", "action": "modify-correction"},
            {"label": "Tuning", "action": "modify-alternative"},
        ]})
        self.send_sse({"type": "done"})

    def _send_network_info(self, net_path):
        """Send network coordinate data via SSE (for chat Canvas rendering)."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(net_path)
            root = tree.getroot()

            edges_data = []
            for edge in root.findall("edge"):
                eid = edge.get("id", "")
                if eid.startswith(":"):
                    continue
                lanes = edge.findall("lane")
                if not lanes:
                    continue
                shape_str = lanes[0].get("shape", "")
                points = []
                for pair in shape_str.split():
                    parts = pair.split(",")
                    if len(parts) == 2:
                        points.append([round(float(parts[0]), 1), round(float(parts[1]), 1)])
                if points:
                    length = float(lanes[0].get("length", 0))
                    edges_data.append({
                        "lanes": len(lanes),
                        "length": round(length, 1),
                        "points": points,
                    })

            junctions = [j for j in root.findall("junction") if j.get("type") != "internal"]
            junc_data = []
            for j in junctions:
                x = j.get("x")
                y = j.get("y")
                if x and y:
                    junc_data.append({
                        "x": round(float(x), 1),
                        "y": round(float(y), 1),
                        "type": j.get("type", ""),
                    })

            # Calculate bbox
            all_x = [p[0] for e in edges_data for p in e["points"]]
            all_y = [p[1] for e in edges_data for p in e["points"]]
            bbox = {
                "x_min": min(all_x) if all_x else 0,
                "y_min": min(all_y) if all_y else 0,
                "x_max": max(all_x) if all_x else 1000,
                "y_max": max(all_y) if all_y else 1000,
            }

            self.send_sse({
                "type": "network",
                "data": {
                    "edge_count": len(edges_data),
                    "junction_count": len(junc_data),
                    "bbox": bbox,
                    "edges": edges_data,
                    "junctions": junc_data,
                },
            })
        except Exception as e:
            self.send_sse({"type": "message", "text": f"Network info extraction error: {e}"})

    def _run_sumo(self, cfg_path, warmup_seconds=0):
        from src.validator import parse_sumo_statistics

        cfg_dir = os.path.dirname(cfg_path)
        tripinfo_path = os.path.join(cfg_dir, "tripinfo.xml")
        cmd = [
            get_sumo_bin(), "-c", cfg_path,
            "--duration-log.statistics", "--no-step-log",
            "--tripinfo-output", tripinfo_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr

        stats = {}
        for line in output.split("\n"):
            line = line.strip()
            if "Inserted:" in line:
                m = re.search(r'Inserted:\s*(\d+)', line)
                if m: stats["vehicles_inserted"] = int(m.group(1))
            elif "Speed:" in line and "Statistics" not in line:
                m = re.search(r'Speed:\s*([\d.]+)', line)
                if m:
                    stats["avg_speed_ms"] = float(m.group(1))
                    stats["avg_speed_kmh"] = round(float(m.group(1)) * 3.6, 1)
            elif "WaitingTime:" in line:
                m = re.search(r'WaitingTime:\s*([\d.]+)', line)
                if m: stats["avg_waiting_time_s"] = float(m.group(1))
            elif "TimeLoss:" in line:
                m = re.search(r'TimeLoss:\s*([\d.]+)', line)
                if m: stats["avg_time_loss_s"] = float(m.group(1))
        detailed_stats = parse_sumo_statistics(cfg_path, warmup_seconds=warmup_seconds)
        stats.update(detailed_stats)

        if "detector_avg_speed_kmh" in detailed_stats:
            stats["avg_speed_kmh"] = detailed_stats["detector_avg_speed_kmh"]
            stats["avg_speed_ms"] = detailed_stats.get("detector_avg_speed_ms", round(stats["avg_speed_kmh"] / 3.6, 2))
        elif "trip_avg_speed_kmh" in detailed_stats:
            stats["avg_speed_kmh"] = detailed_stats["trip_avg_speed_kmh"]
            stats["avg_speed_ms"] = detailed_stats.get("trip_avg_speed_ms", round(stats["avg_speed_kmh"] / 3.6, 2))

        if "trip_avg_waiting_s" in detailed_stats:
            stats["avg_waiting_time_s"] = detailed_stats["trip_avg_waiting_s"]
        if "trip_avg_timeloss_s" in detailed_stats:
            stats["avg_time_loss_s"] = detailed_stats["trip_avg_timeloss_s"]
        return stats

    def log_message(self, format, *args):
        # Hide static file request logs
        if args and ".html" not in str(args[0]) and "/api" not in str(args[0]):
            return
        super().log_message(format, *args)


def main():
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n{'='*50}")
    print(f"  🚗 SUMO Traffic Simulation Agent")
    print(f"  http://localhost:{PORT}")
    print(f"{'='*50}")

    provider, _ = detect_provider()
    print(f"  LLM: {provider.upper() if provider else 'Rule-based'}")
    print(f"  SUMO: {SUMO_BIN or 'Not installed'}")
    print(f"  Data: {'✓ Seoul API' if os.environ.get('TOPIS_API_KEY') else 'Stats only'}")
    print(f"  Python: {sys.executable}")
    print(f"\n  Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
