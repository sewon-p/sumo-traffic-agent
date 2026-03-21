#!/usr/bin/env python3
"""
Fine-tuning runner.

Fine-tunes models via the OpenAI / Gemini API,
and can also leverage training data locally through CLI tools (gemini, codex).

Usage:
    python -m training.fine_tune --provider openai --api-key sk-xxx
    python -m training.fine_tune --provider gemini --api-key AIza-xxx
    python -m training.fine_tune --provider cli    # CLI-based (no API key required)
"""

import argparse
import json
import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fine_tune_openai(api_key: str, data_path: str, model: str = "gpt-5.4"):
    """Fine-tune using the OpenAI fine-tuning API."""
    try:
        import openai
    except ImportError:
        print("openai package required: pip install openai")
        return

    client = openai.OpenAI(api_key=api_key)

    # 1) Upload file
    print(f"[1/3] Uploading training data: {data_path}")
    with open(data_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"  File ID: {file_obj.id}")

    # 2) Create fine-tuning job
    print(f"[2/3] Starting fine-tuning (base: {model})")
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=model,
        suffix="sumo-traffic",
    )
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")

    # 3) Status check instructions
    print(f"\n[3/3] May take minutes to hours to complete")
    print(f"  Check: openai api fine_tuning.jobs.retrieve -i {job.id}")
    print(f"  Or: python -m training.fine_tune --check-openai {job.id} --api-key {api_key[:10]}...")

    return job.id


def fine_tune_gemini(api_key: str, data_path: str, model: str = "models/gemini-2.5-flash"):
    """Fine-tune using the Gemini tuning API."""
    try:
        from google import genai
    except ImportError:
        print("google-genai package required: pip install google-genai")
        return

    client = genai.Client(api_key=api_key)

    # Load training data
    print(f"[1/2] Loading training data: {data_path}")
    training_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            training_data.append(json.loads(line))
    print(f"  {len(training_data)} records")

    # Create tuning job
    print(f"[2/2] Starting tuning (base: {model})")
    job = client.tunings.tune(
        base_model=model,
        training_dataset=training_data,
        config={
            "tuned_model_display_name": "sumo-traffic-agent",
            "epoch_count": 3,
        },
    )
    print(f"  Job: {job}")
    return job


def cli_based_training(provider: str = "gemini", data_path: str = None):
    """
    CLI-based 'training' -- not actual fine-tuning, but a method that
    injects training data as few-shot examples into the system prompt.

    CLI tools (gemini, codex, claude) use OAuth authentication,
    so they work without an API key.
    """
    if not data_path:
        data_path = os.path.join(DATA_DIR, "train_raw.jsonl")

    if not os.path.exists(data_path):
        print(f"Training data not found: {data_path}")
        print("Run first: python -m training.generate_dataset")
        return

    # Extract few-shot examples from training data
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            examples.append(d)

    # Sample evenly from different types
    import random
    random.seed(42)

    abstract = [e for e in examples if e.get("meta", {}).get("type", "").startswith("abstract")]
    specific = [e for e in examples if not e.get("meta", {}).get("type", "").startswith("abstract")]

    selected = random.sample(abstract, min(5, len(abstract)))
    selected += random.sample(specific, min(10, len(specific)))

    # Generate few-shot prompt
    few_shot_prompt = "다음은 교통 시뮬레이션 파라미터 예시다:\n\n"
    for i, ex in enumerate(selected):
        p = ex["params"]
        few_shot_prompt += f"[예시 {i+1}]\n"
        few_shot_prompt += f"입력: {ex['prompt']}\n"
        few_shot_prompt += f"출력: {json.dumps(p, ensure_ascii=False)}\n\n"

    # Save prompt file
    prompt_path = os.path.join(DATA_DIR, "few_shot_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(few_shot_prompt)

    print(f"Few-shot prompt generated: {prompt_path}")
    print(f"  {len(selected)} examples included")
    print(f"\n  Usage:")
    print(f"  cat {prompt_path} | gemini  # Gemini CLI")
    print(f"  claude -p \"$(cat {prompt_path}) 강남역 퇴근\"  # Claude CLI")

    # Test run
    test_input = "판교 출근시간 4차로 도로"
    print(f"\n  Test: '{test_input}'")

    full_prompt = few_shot_prompt + f"\n위 예시를 참고하여, 다음 입력에 대한 파라미터를 JSON으로만 답해:\n입력: {test_input}"

    if provider == "gemini":
        result = subprocess.run(
            ["gemini"], input=full_prompt, capture_output=True, text=True, timeout=60
        )
    elif provider == "claude":
        result = subprocess.run(
            ["claude", "-p", full_prompt], capture_output=True, text=True, timeout=60
        )
    elif provider == "codex":
        result = subprocess.run(
            ["codex", "exec", full_prompt], capture_output=True, text=True, timeout=60
        )
    else:
        print(f"  Unsupported CLI: {provider}")
        return

    print(f"  Result:\n{result.stdout.strip()[:500]}")
    return prompt_path


def check_openai_job(job_id: str, api_key: str):
    """Check the status of an OpenAI fine-tuning job."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"  Status: {job.status}")
    if job.fine_tuned_model:
        print(f"  Model ID: {job.fine_tuned_model}")
        print(f"\n  Use this model ID in llm_client.py:")
        print(f'  GPTClient(api_key="...", model="{job.fine_tuned_model}")')
    return job


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning runner")
    parser.add_argument("--provider", choices=["openai", "gemini", "cli"], required=True)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--cli-provider", type=str, default="gemini", help="CLI to use in CLI mode (gemini/claude/codex)")
    parser.add_argument("--check-openai", type=str, default=None, help="Check OpenAI job ID status")
    args = parser.parse_args()

    if args.check_openai:
        check_openai_job(args.check_openai, args.api_key)
        return

    if args.provider == "openai":
        if not args.api_key:
            args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            print("--api-key or OPENAI_API_KEY environment variable required")
            return
        data_path = args.data or os.path.join(DATA_DIR, "train_openai.jsonl")
        fine_tune_openai(args.api_key, data_path, model=args.model or "gpt-5.4")

    elif args.provider == "gemini":
        if not args.api_key:
            args.api_key = os.environ.get("GEMINI_API_KEY")
        if not args.api_key:
            print("--api-key or GEMINI_API_KEY environment variable required")
            return
        data_path = args.data or os.path.join(DATA_DIR, "train_gemini.jsonl")
        fine_tune_gemini(args.api_key, data_path, model=args.model or "models/gemini-2.5-flash")

    elif args.provider == "cli":
        cli_based_training(provider=args.cli_provider, data_path=args.data)


if __name__ == "__main__":
    main()
