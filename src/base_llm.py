#!/usr/bin/env python3
"""
General-purpose LLM calls (for geometry XML generation/modification)

Priority: OPENAI_API_KEY -> CLI (gemini/claude/codex)
Uses base models, not fine-tuned models.
"""

import json
import os
import re
import shutil
import subprocess
import time
from threading import local

from src.token_tracker import tracker as _tracker


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_env()
_REQUEST_OVERRIDE = local()


def set_request_base_llm_override(mode: str = "", cli: str = "", provider: str = "", api_key: str = "", model: str = ""):
    _REQUEST_OVERRIDE.config = {
        "mode": mode,
        "cli": cli,
        "provider": provider,
        "api_key": api_key,
        "model": model,
    }


def clear_request_base_llm_override():
    if hasattr(_REQUEST_OVERRIDE, "config"):
        delattr(_REQUEST_OVERRIDE, "config")


def _request_base_llm_override():
    return getattr(_REQUEST_OVERRIDE, "config", {}) or {}


def _shared_base_llm_settings():
    provider = os.environ.get("BASE_LLM_SHARED_PROVIDER") or os.environ.get("BASE_LLM_CUSTOM_PROVIDER", "")
    api_key = os.environ.get("BASE_LLM_SHARED_KEY") or os.environ.get("BASE_LLM_CUSTOM_KEY", "")
    model = os.environ.get("BASE_LLM_SHARED_MODEL") or os.environ.get("BASE_LLM_CUSTOM_MODEL", "")
    return provider, api_key, model


def _cli_timeout_sec() -> int:
    """Timeout for CLI-based base LLM calls."""
    try:
        return max(60, int(os.environ.get("BASE_LLM_CLI_TIMEOUT_SEC", "180")))
    except ValueError:
        return 180


def ask_base_llm(prompt: str, system: str = "") -> str:
    """
    Ask a general-purpose LLM a question.
    Uses CLI if BASE_LLM_MODE=cli, otherwise uses API.
    """
    override = _request_base_llm_override()
    mode = override.get("mode") or os.environ.get("BASE_LLM_MODE", "api")

    print(f"\n[BASE_LLM] mode={mode}")
    print(f"[BASE_LLM] prompt: {prompt[:100]}...")

    if mode == "cli":
        preferred = override.get("cli") or os.environ.get("BASE_LLM_CLI", "")
        if preferred and shutil.which(preferred):
            return _ask_cli(prompt, system, preferred)
        for cli in ["gemini", "claude", "codex"]:
            if shutil.which(cli):
                return _ask_cli(prompt, system, cli)
        mode = "api"

    if mode == "api":
        request_key = override.get("api_key", "")
        request_provider = override.get("provider", "")
        request_model = override.get("model", "")
        if request_key and request_provider:
            if request_provider == "gpt":
                return _ask_openai(prompt, system, request_key, request_model)
            if request_provider == "gemini":
                return _ask_gemini(prompt, system, request_key, request_model)
            if request_provider == "claude":
                return _ask_claude(prompt, system, request_key, request_model)

        shared_provider, shared_key, shared_model = _shared_base_llm_settings()
        if shared_key and shared_provider:
            if shared_provider == "gpt":
                return _ask_openai(prompt, system, shared_key, shared_model)
            if shared_provider == "gemini":
                return _ask_gemini(prompt, system, shared_key, shared_model)
            if shared_provider == "claude":
                return _ask_claude(prompt, system, shared_key, shared_model)

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return _ask_openai(prompt, system, api_key, "")

    raise RuntimeError("Cannot use LLM. Check BASE_LLM_MODE and API keys in .env.")


def _log_response(text):
    print(f"[BASE_LLM] response: {text[:200]}...")
    return text


def _ask_openai(prompt: str, system: str, api_key: str, model_override: str = "") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = model_override or os.environ.get("BASE_LLM_OPENAI_MODEL") or os.environ.get("BASE_LLM_MODEL", "gpt-5.4")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    print(f"[BASE_LLM] OpenAI call: {model}")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, max_completion_tokens=2000,
    )
    latency = (time.perf_counter() - t0) * 1000
    result = resp.choices[0].message.content.strip()
    if resp.usage:
        _tracker.record("openai", model, resp.usage.prompt_tokens, resp.usage.completion_tokens, latency, caller="base_llm")
    print(f"[BASE_LLM] response length: {len(result)} chars")
    return result


def _ask_gemini(prompt: str, system: str, api_key: str, model_override: str = "") -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    full = f"{system}\n\n{prompt}" if system else prompt
    model = model_override or os.environ.get("BASE_LLM_GEMINI_MODEL", "gemini-2.5-flash")
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=full,
    )
    latency = (time.perf_counter() - t0) * 1000
    um = getattr(resp, "usage_metadata", None)
    if um:
        _tracker.record("gemini", model, getattr(um, "prompt_token_count", 0), getattr(um, "candidates_token_count", 0), latency, caller="base_llm")
    return resp.text.strip()


def _ask_claude(prompt: str, system: str, api_key: str, model_override: str = "") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    model = model_override or os.environ.get("BASE_LLM_CLAUDE_MODEL", "claude-sonnet-4-5")
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    latency = (time.perf_counter() - t0) * 1000
    _tracker.record("claude", model, msg.usage.input_tokens, msg.usage.output_tokens, latency, caller="base_llm")
    return msg.content[0].text.strip()


def _ask_cli(prompt: str, system: str, cli_name: str) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    timeout_sec = _cli_timeout_sec()

    if cli_name == "gemini":
        result = subprocess.run(
            [shutil.which("gemini")],
            input=full.encode(), capture_output=True, timeout=timeout_sec,
        )
        return result.stdout.decode().strip()
    elif cli_name == "claude":
        result = subprocess.run(
            [shutil.which("claude"), "-p", full],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        return result.stdout.strip()
    elif cli_name == "codex":
        result = subprocess.run(
            [shutil.which("codex"), "exec", full],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        return result.stdout.strip()

    return ""


def classify_modification(user_input: str) -> str:
    """
    Determine whether the modification request is a geometry (XML) change
    or a parameter change.

    Returns: "geometry", "parameter", or "mixed"
    """
    system = (
        "The user wants to modify a traffic simulation. "
        "Determine whether the modification request is a 'road structure (geometry) change' "
        "or a 'traffic parameter change'.\n"
        "If both are included, answer mixed.\n"
        "Answer with only one word: geometry, parameter, or mixed.\n\n"
        "geometry examples: bend the road, add/remove intersection, add lanes, lengthen/shorten road, make curved\n"
        "parameter examples: increase/decrease traffic volume, raise/lower speed, change time period, change block length/intersection spacing\n"
        "mixed examples: set speed limit to 70 and make road straight / set speed limit to 70 and block to 2km"
    )
    try:
        result = ask_base_llm(user_input, system)
        lowered = result.lower()
        if "mixed" in lowered:
            return "mixed"
        if "geometry" in lowered:
            return "geometry"
        return "parameter"
    except (ValueError, Exception):
        # On failure, use keyword heuristics only
        has_geometry = any(token in user_input for token in ["교차로", "사거리", "삼거리", "직선", "곡선", "휘", "로터리", "원형"])
        has_parameter = any(token in user_input for token in ["제한속도", "교통량", "속도", "tau", "sigma", "vph", "블록", "간격", "km", "m"])
        if has_geometry and has_parameter:
            return "mixed"
        if has_geometry:
            return "geometry"
        return "parameter"


def modify_parameters(user_input: str, params_json: str) -> dict:
    """
    Have the base LLM interpret parameter modifications.

    Returns: dict containing only the fields to change (e.g., {"volume_vph": 3000})
    """
    system = (
        "The user wants to modify traffic simulation parameters.\n"
        f"Current parameters: {params_json}\n"
        "Interpret the user's modification request and return only the fields to change as JSON.\n"
        "Do not include fields that are not being changed. Output only JSON.\n"
        "Available fields: volume_vph, speed_limit_kmh, sigma, tau, avg_block_m\n"
        "Examples:\n"
        "  'set traffic volume to 3000' -> {\"volume_vph\": 3000}\n"
        "  'lower speed' -> {\"speed_limit_kmh\": current*0.8}\n"
        "  'make it more congested' -> {\"volume_vph\": current*1.3, \"sigma\": current+0.1}\n"
        "  'set block to 2km' -> {\"avg_block_m\": 2000}\n"
        "Understand both m and km distance units and return values in meters."
    )
    try:
        result = ask_base_llm(user_input, system)
        m = re.search(r'\{[\s\S]*\}', result)
        if m:
            return json.loads(m.group())
    except (json.JSONDecodeError, ValueError, Exception):
        pass
    return {}


def extract_ft_training_hints(user_input: str, ft_json: str) -> dict:
    """
    Extract numeric hints (lanes, avg_block_m, etc.) for FT retraining
    even from geometry modification requests.
    The execution path remains geometry, but retraining signals are recorded separately.
    """
    system = (
        "Extract only numeric hints usable for FT parameter calibration from the user's modification request.\n"
        f"Current FT parameters: {ft_json}\n"
        "Even if the execution is a geometry modification, return hints as JSON if available.\n"
        "Available fields: lanes, avg_block_m, speed_limit_kmh, volume_vph, sigma, tau\n"
        "If none, return only {}. Output only JSON without any other text.\n"
        "Examples:\n"
        "  'set block to 2km' -> {\"avg_block_m\": 2000}\n"
        "  'set lanes to 3' -> {\"lanes\": 3}\n"
        "  'make it a straight road' -> {}\n"
    )
    try:
        result = ask_base_llm(user_input, system)
        m = re.search(r'\{[\s\S]*\}', result)
        if m:
            parsed = json.loads(m.group())
            if parsed:
                return parsed
    except Exception:
        pass

    hints = {}
    lane_match = re.search(r'(\d+)\s*(?:차로|차선|lane)', user_input, re.IGNORECASE)
    lane_match_after = re.search(r'(?:차로|차선|lane)(?:은|는|을|를)?\s*(\d+)', user_input, re.IGNORECASE)
    if lane_match:
        hints["lanes"] = int(lane_match.group(1))
    elif lane_match_after:
        hints["lanes"] = int(lane_match_after.group(1))

    km_match = re.search(r'(\d+(?:\.\d+)?)\s*km', user_input, re.IGNORECASE)
    m_match = re.search(r'(\d+(?:\.\d+)?)\s*m(?![a-z])', user_input, re.IGNORECASE)
    if "블록" in user_input or "간격" in user_input:
        if km_match:
            hints["avg_block_m"] = float(km_match.group(1)) * 1000
        elif m_match:
            hints["avg_block_m"] = float(m_match.group(1))

    return hints


def generate_network_xml(user_input: str, params: dict) -> tuple:
    """
    Have the LLM directly generate SUMO .nod.xml + .edg.xml.

    Returns: (nod_xml_str, edg_xml_str)
    """
    system = (
        "You are a SUMO traffic simulation network design expert.\n"
        "Given the user's road conditions, generate .nod.xml and .edg.xml for SUMO netconvert.\n\n"
        "Rules:\n"
        "- Bidirectional roads: create an edge for each direction (e.g., e0_fwd, e0_rev)\n"
        "- Intersections: type='traffic_light', endpoints: type='priority'\n"
        "- speed is in m/s units (km/h / 3.6)\n"
        "- Coordinates are in meters, based on origin (0,0)\n"
        "- Include cross streets (short north-south roads at intersections)\n\n"
        "You must output in the format below. Only XML, no other text:\n"
        "===NOD===\n<nodes>...</nodes>\n===EDG===\n<edges>...</edges>"
    )

    prompt = (
        f"Road conditions: {user_input}\n"
        f"Parameters: {params.get('lanes', 2)} lanes per direction, "
        f"speed limit {params.get('speed_limit_kmh', 50)}km/h, "
        f"block spacing approx. {params.get('avg_block_m', 200)}m"
    )

    response = ask_base_llm(prompt, system)

    # Split by ===NOD=== and ===EDG===
    nod_xml = ""
    edg_xml = ""

    if "===NOD===" in response and "===EDG===" in response:
        parts = response.split("===EDG===")
        nod_part = parts[0].split("===NOD===")[-1].strip()
        edg_part = parts[1].strip()

        # Extract XML tags only
        nod_match = re.search(r'<nodes[\s\S]*?</nodes>', nod_part)
        edg_match = re.search(r'<edges[\s\S]*?</edges>', edg_part)
        if nod_match:
            nod_xml = nod_match.group()
        if edg_match:
            edg_xml = edg_match.group()
    else:
        # Fallback: find <nodes> and <edges> directly
        nod_match = re.search(r'<nodes[\s\S]*?</nodes>', response)
        edg_match = re.search(r'<edges[\s\S]*?</edges>', response)
        if nod_match:
            nod_xml = nod_match.group()
        if edg_match:
            edg_xml = edg_match.group()

    if not nod_xml or not edg_xml:
        raise ValueError(f"XML generation failed. LLM response:\n{response[:500]}")

    return nod_xml, edg_xml


def modify_network_xml(user_input: str, current_nod: str, current_edg: str) -> tuple:
    """
    Have the LLM modify existing XML.

    Returns: (nod_xml_str, edg_xml_str)
    """
    system = (
        "You are a SUMO network XML modification expert.\n"
        "Modify the existing .nod.xml and .edg.xml according to the user's modification request.\n"
        "Output the complete modified XML. Only XML, no other text:\n"
        "===NOD===\n<nodes>...</nodes>\n===EDG===\n<edges>...</edges>"
    )

    prompt = (
        f"Modification request: {user_input}\n\n"
        f"Current .nod.xml:\n{current_nod}\n\n"
        f"Current .edg.xml:\n{current_edg}"
    )

    response = ask_base_llm(prompt, system)

    nod_xml = ""
    edg_xml = ""

    if "===NOD===" in response and "===EDG===" in response:
        parts = response.split("===EDG===")
        nod_part = parts[0].split("===NOD===")[-1].strip()
        edg_part = parts[1].strip()
        nod_match = re.search(r'<nodes[\s\S]*?</nodes>', nod_part)
        edg_match = re.search(r'<edges[\s\S]*?</edges>', edg_part)
        if nod_match:
            nod_xml = nod_match.group()
        if edg_match:
            edg_xml = edg_match.group()
    else:
        nod_match = re.search(r'<nodes[\s\S]*?</nodes>', response)
        edg_match = re.search(r'<edges[\s\S]*?</edges>', response)
        if nod_match:
            nod_xml = nod_match.group()
        if edg_match:
            edg_xml = edg_match.group()

    if not nod_xml or not edg_xml:
        raise ValueError(f"XML modification failed. LLM response:\n{response[:500]}")

    return nod_xml, edg_xml
