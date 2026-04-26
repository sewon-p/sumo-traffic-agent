"""
Multi-LLM Client Abstraction Layer

Works with Gemini, Claude, or GPT through a unified tool-calling interface
as long as the API key is configured.

Usage:
    client = create_client()  # Auto-detect from environment variables
    client = create_client(provider="gemini", api_key="...")
"""

import json
import os
import time
from abc import ABC, abstractmethod

from src.token_tracker import tracker as _tracker


class LLMClient(ABC):
    """Abstract LLM client interface"""

    @abstractmethod
    def chat(self, messages: list, system: str, tools: list) -> dict:
        """
        Send messages to the LLM and receive a response.

        Returns:
            {
                "text": "Response text (if any)",
                "tool_calls": [
                    {"id": "...", "name": "tool_name", "input": {...}},
                    ...
                ],
                "stop_reason": "end_turn" | "tool_use"
            }
        """
        pass

    @abstractmethod
    def provider_name(self) -> str:
        pass


# ──────────────────────────────────────────────
# Claude (Anthropic)
# ──────────────────────────────────────────────

class ClaudeClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def provider_name(self) -> str:
        return f"Claude ({self.model})"

    def chat(self, messages: list, system: str, tools: list) -> dict:
        t0 = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )
        latency = (time.perf_counter() - t0) * 1000
        _tracker.record("claude", self.model, response.usage.input_tokens, response.usage.output_tokens, latency, caller="agent")

        result = {"text": "", "tool_calls": [], "stop_reason": response.stop_reason}

        for block in response.content:
            if block.type == "text":
                result["text"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Preserve Claude's raw content (for message history)
        result["_raw_content"] = response.content
        return result

    def format_tool_result(self, tool_call_id: str, result: str) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result,
        }


# ──────────────────────────────────────────────
# Gemini (Google)
# ──────────────────────────────────────────────

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite-preview"):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def provider_name(self) -> str:
        return f"Gemini ({self.model})"

    def _convert_tools_to_gemini(self, tools: list) -> list:
        """Convert Claude/OpenAI format tool schema to Gemini format"""
        from google.genai import types

        declarations = []
        for tool in tools:
            # input_schema -> parameters
            schema = tool.get("input_schema", {})
            # Gemini requires non-empty properties
            props = schema.get("properties", {})
            if not props:
                continue

            declarations.append(types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=schema,
            ))

        return [types.Tool(function_declarations=declarations)]

    def chat(self, messages: list, system: str, tools: list) -> dict:
        from google.genai import types

        gemini_tools = self._convert_tools_to_gemini(tools)

        # Convert messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    gemini_contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)]
                    ))
                elif isinstance(content, list):
                    # tool_result list
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            parts.append(types.Part.from_function_response(
                                name=item.get("_tool_name", "unknown"),
                                response={"result": item["content"]},
                            ))
                    if parts:
                        gemini_contents.append(types.Content(role="user", parts=parts))

            elif role == "assistant":
                if isinstance(content, str):
                    gemini_contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)]
                    ))
                elif isinstance(content, list):
                    # Restore tool_call response
                    parts = []
                    for item in content:
                        if hasattr(item, 'type'):
                            if item.type == "text":
                                parts.append(types.Part.from_text(text=item.text))
                            elif item.type == "tool_use":
                                parts.append(types.Part.from_function_call(
                                    name=item.name,
                                    args=item.input,
                                ))
                        elif isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append(types.Part.from_text(text=item["text"]))
                            elif item.get("type") == "function_call":
                                parts.append(types.Part.from_function_call(
                                    name=item["name"],
                                    args=item.get("args", {}),
                                ))
                    if parts:
                        gemini_contents.append(types.Content(role="model", parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=gemini_tools,
            temperature=0.3,
        )

        t0 = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )
        latency = (time.perf_counter() - t0) * 1000
        um = getattr(response, "usage_metadata", None)
        if um:
            _tracker.record("gemini", self.model, getattr(um, "prompt_token_count", 0), getattr(um, "candidates_token_count", 0), latency, caller="agent")

        result = {"text": "", "tool_calls": [], "stop_reason": "end_turn"}

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    result["text"] += part.text
                elif part.function_call:
                    fc = part.function_call
                    result["tool_calls"].append({
                        "id": f"gemini_{fc.name}",
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {},
                    })
                    result["stop_reason"] = "tool_use"

        # Preserve Gemini raw content
        if response.candidates and response.candidates[0].content:
            result["_raw_content"] = response.candidates[0].content.parts

        return result

    def format_tool_result(self, tool_call_id: str, result: str) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result,
            "_tool_name": tool_call_id.replace("gemini_", ""),
        }


# ──────────────────────────────────────────────
# GPT (OpenAI)
# ──────────────────────────────────────────────

class GPTClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-5.4"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def provider_name(self) -> str:
        return f"GPT ({self.model})"

    def _convert_tools_to_openai(self, tools: list) -> list:
        """Claude format -> OpenAI function calling format"""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }
            })
        return openai_tools

    def chat(self, messages: list, system: str, tools: list) -> dict:
        openai_tools = self._convert_tools_to_openai(tools)

        # Convert messages
        openai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    openai_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # tool results
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"],
                            })

            elif role == "assistant":
                if isinstance(content, str):
                    openai_messages.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    # Assistant message with tool_calls
                    tool_calls_openai = []
                    text_parts = []
                    for item in content:
                        if hasattr(item, 'type'):
                            if item.type == "text":
                                text_parts.append(item.text)
                            elif item.type == "tool_use":
                                tool_calls_openai.append({
                                    "id": item.id,
                                    "type": "function",
                                    "function": {
                                        "name": item.name,
                                        "arguments": json.dumps(item.input),
                                    }
                                })
                    msg_dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
                    if tool_calls_openai:
                        msg_dict["tool_calls"] = tool_calls_openai
                    openai_messages.append(msg_dict)

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            temperature=0.3,
        )
        latency = (time.perf_counter() - t0) * 1000
        if response.usage:
            _tracker.record("openai", self.model, response.usage.prompt_tokens, response.usage.completion_tokens, latency, caller="agent")

        choice = response.choices[0]
        result = {"text": choice.message.content or "", "tool_calls": [], "stop_reason": "end_turn"}

        if choice.message.tool_calls:
            result["stop_reason"] = "tool_use"
            for tc in choice.message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        result["_raw_content"] = choice.message
        return result

    def format_tool_result(self, tool_call_id: str, result: str) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result,
        }


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# CLI mode (uses locally installed CLI tools)
# ──────────────────────────────────────────────

class CLIClient(LLMClient):
    """Calls locally installed CLI tools (gemini, claude, codex) via subprocess."""

    def __init__(self, cli_name: str):
        import shutil
        self.cli_name = cli_name
        self.cli_path = shutil.which(cli_name)
        if not self.cli_path:
            raise FileNotFoundError(f"'{cli_name}' CLI not found.")

    def provider_name(self) -> str:
        return f"{self.cli_name} (CLI)"

    def chat(self, messages: list, system: str, tools: list) -> dict:
        import subprocess

        # Extract the last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user" and isinstance(msg["content"], str):
                user_msg = msg["content"]
                break

        # Include tool info in prompt
        tool_names = [t["name"] for t in tools]
        full_prompt = f"{system}\n\nAvailable tools: {', '.join(tool_names)}\n\nUser request: {user_msg}"

        # Build command per CLI
        if self.cli_name == "gemini":
            cmd = [self.cli_path]
            input_data = full_prompt.encode()
        elif self.cli_name == "claude":
            cmd = [self.cli_path, "-p", full_prompt]
            input_data = None
        elif self.cli_name == "codex":
            cmd = [self.cli_path, "exec", full_prompt]
            input_data = None
        else:
            cmd = [self.cli_path]
            input_data = full_prompt.encode()

        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=(input_data is None),
                timeout=120,
            )
            output = result.stdout if isinstance(result.stdout, str) else result.stdout.decode()
        except subprocess.TimeoutExpired:
            output = "CLI response timed out (120 seconds)"
        except Exception as e:
            output = f"CLI execution error: {e}"

        return {
            "text": output.strip(),
            "tool_calls": [],  # CLI mode cannot do tool-calling, text response only
            "stop_reason": "end_turn",
        }

    def format_tool_result(self, tool_call_id: str, result: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": result}


def detect_cli() -> str:
    """Detect locally installed LLM CLI tools."""
    import shutil
    for name in ["gemini", "claude", "codex"]:
        if shutil.which(name):
            return name
    return ""


def detect_provider() -> tuple[str, str]:
    """Auto-detect available LLM provider from environment variables."""
    # Load .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

    providers = [
        ("claude", "ANTHROPIC_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("gpt", "OPENAI_API_KEY"),
    ]

    for name, env_key in providers:
        key = os.environ.get(env_key, "")
        if key:
            return name, key

    # If no API keys, detect CLI
    cli = detect_cli()
    if cli:
        return cli, "CLI"

    return "", ""


def create_client(provider: str = None, api_key: str = None) -> LLMClient:
    """
    Create an LLM client.

    Args:
        provider: "claude", "gemini", "gpt" (None for auto-detection)
        api_key: API key (None to use environment variable)

    Returns:
        LLMClient instance
    """
    if provider is None or api_key is None:
        detected_provider, detected_key = detect_provider()
        provider = provider or detected_provider
        if detected_key != "CLI":
            api_key = api_key or detected_key

    # If API key is available, use API client
    if provider and api_key and api_key != "CLI":
        if provider == "claude":
            return ClaudeClient(api_key)
        elif provider == "gemini":
            return GeminiClient(api_key)
        elif provider == "gpt":
            return GPTClient(api_key)

    # If no API key, try CLI mode
    cli_name = provider or detect_cli()
    if cli_name:
        try:
            return CLIClient(cli_name)
        except FileNotFoundError:
            pass

    raise ValueError(
        "Cannot use LLM.\n"
        "Option 1: Install a local CLI (gemini, claude, codex)\n"
        "Option 2: Set API keys in .env (GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)"
    )
