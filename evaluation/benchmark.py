#!/usr/bin/env python3
"""
LLM Benchmark: Fine-tuned model vs Base model comparison.

Two benchmark modes:
  1. Quality check: JSON compliance, field completeness, consistency (CV)
  2. Accuracy (ground truth): Compare model output against validation dataset
     using MAE/MAPE per field

Usage:
    python -m evaluation.benchmark                          # Quality check only
    python -m evaluation.benchmark --accuracy               # Ground truth comparison
    python -m evaluation.benchmark --accuracy --samples 30  # Use 30 samples
    python -m evaluation.benchmark --all                    # Run both
"""

import json
import os
import re
import sys
import time
import argparse
import statistics
from datetime import datetime
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.llm_parser import FT_SYSTEM_PROMPT

# ── Test prompts ──────────────────────────────────────

TEST_PROMPTS = [
    "강남대로 퇴근시간 시뮬레이션",
    "테헤란로 출근시간 비오는 날",
    "올림픽대로 심야 시간대",
    "홍대입구역 주변 주말 오후",
    "경부고속도로 서울-수원 금요일 저녁",
    "서울역 앞 점심시간 눈오는 날",
    "판교테크노밸리 내부도로 출근시간",
    "강변북로 퇴근시간 사고 발생",
    "세종대로 낮 시간대 맑은 날",
    "이면도로 골목길 등하교 시간",
]

# ── Expected parameter ranges ─────────────────────────

VALID_RANGES = {
    "speed_kmh": (5, 120),
    "volume_vph": (50, 5000),
    "lanes": (1, 8),
    "speed_limit_kmh": (20, 120),
    "sigma": (0.0, 1.0),
    "tau": (0.3, 5.0),
    "avg_block_m": (30, 2000),
}

REQUIRED_FIELDS = ["speed_kmh", "volume_vph", "lanes", "speed_limit_kmh", "sigma", "tau"]


@dataclass
class SingleResult:
    """Result of a single LLM call."""
    prompt: str
    model_type: str  # "ft" or "base"
    raw_response: str = ""
    json_valid: bool = False
    parsed: dict = field(default_factory=dict)
    fields_present: list = field(default_factory=list)
    fields_missing: list = field(default_factory=list)
    range_violations: list = field(default_factory=list)
    latency_ms: int = 0
    error: str = ""


def call_model(prompt: str, model: str, system_prompt: str, api_key: str) -> tuple[str, int]:
    """Call OpenAI model and return (response_text, latency_ms)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    latency = int((time.time() - start) * 1000)
    text = resp.choices[0].message.content.strip()
    return text, latency


def evaluate_response(prompt: str, model_type: str, raw: str, latency: int) -> SingleResult:
    """Evaluate a single model response."""
    result = SingleResult(prompt=prompt, model_type=model_type, raw_response=raw, latency_ms=latency)

    # 1. JSON parsing
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        result.error = "No JSON found in response"
        return result

    try:
        data = json.loads(json_match.group())
        result.json_valid = True
        result.parsed = data
    except json.JSONDecodeError as e:
        result.error = f"JSON parse error: {e}"
        return result

    # 2. Field completeness
    for f in REQUIRED_FIELDS:
        if f in data and data[f] is not None and str(data[f]).strip() not in ("", "-"):
            result.fields_present.append(f)
        else:
            result.fields_missing.append(f)

    # 3. Range validity
    for f, (lo, hi) in VALID_RANGES.items():
        if f in data and data[f] is not None:
            try:
                val = float(data[f])
                if val < lo or val > hi:
                    result.range_violations.append(f"{f}={val} (valid: {lo}-{hi})")
            except (ValueError, TypeError):
                result.range_violations.append(f"{f}='{data[f]}' (not numeric)")

    return result


def run_benchmark(runs_per_prompt: int = 3, output_dir: str = None):
    """Run the full benchmark."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    ft_model = os.environ.get(
        "OPENAI_FT_MODEL",
        "ft:gpt-4.1-mini-2025-04-14:university-of-seoul:sumo-traffic-v2:DL1ENw1g"
    )
    base_model = "gpt-4.1-mini"

    print(f"{'='*60}")
    print(f"  LLM Benchmark: Fine-tuned vs Base Model")
    print(f"  FT model:   {ft_model}")
    print(f"  Base model: {base_model}")
    print(f"  Prompts:    {len(TEST_PROMPTS)}")
    print(f"  Runs/prompt: {runs_per_prompt}")
    print(f"{'='*60}\n")

    all_results = {"ft": [], "base": []}

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[{i+1}/{len(TEST_PROMPTS)}] {prompt}")

        for run in range(runs_per_prompt):
            for model_type, model_id in [("ft", ft_model), ("base", base_model)]:
                try:
                    raw, latency = call_model(prompt, model_id, FT_SYSTEM_PROMPT, api_key)
                    result = evaluate_response(prompt, model_type, raw, latency)
                except Exception as e:
                    result = SingleResult(prompt=prompt, model_type=model_type, error=str(e))

                all_results[model_type].append(result)

            status_ft = "ok" if all_results["ft"][-1].json_valid else "FAIL"
            status_base = "ok" if all_results["base"][-1].json_valid else "FAIL"
            print(f"  run {run+1}: FT={status_ft} ({all_results['ft'][-1].latency_ms}ms)  "
                  f"Base={status_base} ({all_results['base'][-1].latency_ms}ms)")

    # ── Aggregate metrics ──
    report = build_report(all_results, runs_per_prompt)
    print_report(report)

    if output_dir:
        save_report(report, all_results, output_dir)

    return report


def build_report(all_results: dict, runs_per_prompt: int) -> dict:
    """Aggregate individual results into summary metrics."""
    report = {}

    for model_type in ["ft", "base"]:
        results = all_results[model_type]
        total = len(results)
        json_ok = sum(1 for r in results if r.json_valid)
        fields_complete = sum(1 for r in results if r.json_valid and len(r.fields_missing) == 0)
        range_ok = sum(1 for r in results if r.json_valid and len(r.range_violations) == 0)
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]

        # Consistency: for each prompt, compute CV of numeric fields across runs
        consistency = compute_consistency(results, runs_per_prompt)

        report[model_type] = {
            "total_calls": total,
            "json_compliance_pct": round(json_ok / total * 100, 1) if total else 0,
            "field_completeness_pct": round(fields_complete / total * 100, 1) if total else 0,
            "range_validity_pct": round(range_ok / total * 100, 1) if total else 0,
            "avg_latency_ms": round(statistics.mean(latencies)) if latencies else 0,
            "consistency_avg_cv": consistency,
            "common_violations": get_common_violations(results),
            "common_missing": get_common_missing(results),
        }

    return report


def compute_consistency(results: list, runs_per_prompt: int) -> float:
    """
    Compute average Coefficient of Variation across prompts.
    Lower CV = more consistent output.
    """
    numeric_fields = ["speed_kmh", "volume_vph", "speed_limit_kmh", "sigma", "tau"]
    all_cvs = []

    # Group by prompt
    prompts = list(dict.fromkeys(r.prompt for r in results))
    for prompt in prompts:
        prompt_results = [r for r in results if r.prompt == prompt and r.json_valid]
        if len(prompt_results) < 2:
            continue

        for f in numeric_fields:
            values = []
            for r in prompt_results:
                if f in r.parsed:
                    try:
                        values.append(float(r.parsed[f]))
                    except (ValueError, TypeError):
                        pass
            if len(values) >= 2 and statistics.mean(values) != 0:
                cv = statistics.stdev(values) / abs(statistics.mean(values))
                all_cvs.append(cv)

    return round(statistics.mean(all_cvs) * 100, 2) if all_cvs else 0.0


def get_common_violations(results: list) -> list:
    """Get most common range violations."""
    from collections import Counter
    violations = Counter()
    for r in results:
        for v in r.range_violations:
            field_name = v.split("=")[0]
            violations[field_name] += 1
    return violations.most_common(5)


def get_common_missing(results: list) -> list:
    """Get most commonly missing fields."""
    from collections import Counter
    missing = Counter()
    for r in results:
        for f in r.fields_missing:
            missing[f] += 1
    return missing.most_common(5)


def print_report(report: dict):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Fine-tuned':>12} {'Base':>12}")
    print(f"{'-'*54}")

    metrics = [
        ("JSON compliance (%)", "json_compliance_pct"),
        ("Field completeness (%)", "field_completeness_pct"),
        ("Range validity (%)", "range_validity_pct"),
        ("Avg latency (ms)", "avg_latency_ms"),
        ("Consistency CV (%)", "consistency_avg_cv"),
    ]

    for label, key in metrics:
        ft_val = report["ft"][key]
        base_val = report["base"][key]

        # Determine winner (lower is better for CV and latency, higher for others)
        if key in ("consistency_avg_cv", "avg_latency_ms"):
            winner = "ft" if ft_val <= base_val else "base"
        else:
            winner = "ft" if ft_val >= base_val else "base"

        ft_str = f"{ft_val}"
        base_str = f"{base_val}"
        if winner == "ft":
            ft_str = f"*{ft_val}*"
        else:
            base_str = f"*{base_val}*"

        print(f"{label:<30} {ft_str:>12} {base_str:>12}")

    print(f"\n  * = better\n")

    # Violations summary
    for model_type in ["ft", "base"]:
        label = "Fine-tuned" if model_type == "ft" else "Base"
        violations = report[model_type]["common_violations"]
        missing = report[model_type]["common_missing"]
        if violations:
            print(f"  {label} range violations: {violations}")
        if missing:
            print(f"  {label} missing fields: {missing}")

    print()


def save_report(report: dict, all_results: dict, output_dir: str):
    """Save benchmark results to files."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary JSON
    summary_path = os.path.join(output_dir, f"benchmark_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "report": report,
            "prompts": TEST_PROMPTS,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {summary_path}")

    # Detailed results
    detail_path = os.path.join(output_dir, f"benchmark_detail_{ts}.jsonl")
    with open(detail_path, "w") as f:
        for model_type in ["ft", "base"]:
            for r in all_results[model_type]:
                f.write(json.dumps({
                    "model_type": r.model_type,
                    "prompt": r.prompt,
                    "json_valid": r.json_valid,
                    "parsed": r.parsed,
                    "fields_missing": r.fields_missing,
                    "range_violations": r.range_violations,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }, ensure_ascii=False) + "\n")
    print(f"  Details: {detail_path}")

    # Markdown table for README
    md_path = os.path.join(output_dir, f"benchmark_{ts}.md")
    with open(md_path, "w") as f:
        f.write("## LLM Benchmark: Fine-tuned vs Base Model\n\n")
        f.write(f"| Metric | Fine-tuned | Base (gpt-4.1-mini) |\n")
        f.write(f"|--------|-----------|--------------------|\n")
        for label, key in [
            ("JSON compliance", "json_compliance_pct"),
            ("Field completeness", "field_completeness_pct"),
            ("Range validity", "range_validity_pct"),
            ("Avg latency (ms)", "avg_latency_ms"),
            ("Consistency CV (%)", "consistency_avg_cv"),
        ]:
            ft_val = report["ft"][key]
            base_val = report["base"][key]
            f.write(f"| {label} | {ft_val} | {base_val} |\n")
        f.write(f"\n*Lower CV = more consistent output*\n")
    print(f"  Markdown: {md_path}")


def load_validation_set(path: str = None, n_samples: int = 30) -> list:
    """Load ground truth samples from validation JSONL."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "training", "data", "val_real_openai.jsonl")
    if not os.path.exists(path):
        print(f"Error: validation file not found: {path}")
        return []

    samples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            msgs = entry["messages"]
            prompt = msgs[1]["content"]
            gt_text = msgs[2]["content"]
            gt_match = re.search(r'\{[\s\S]*\}', gt_text)
            if gt_match:
                try:
                    gt = json.loads(gt_match.group())
                    samples.append({"prompt": prompt, "ground_truth": gt,
                                    "system_prompt": msgs[0]["content"]})
                except json.JSONDecodeError:
                    pass

    import random
    random.seed(42)
    if len(samples) > n_samples:
        samples = random.sample(samples, n_samples)
    return samples


ACCURACY_FIELDS = ["speed_kmh", "volume_vph", "lanes", "speed_limit_kmh", "sigma", "tau", "avg_block_m"]


def run_accuracy_benchmark(n_samples: int = 30, output_dir: str = None):
    """Run ground truth comparison benchmark."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    ft_model = os.environ.get(
        "OPENAI_FT_MODEL",
        "ft:gpt-4.1-mini-2025-04-14:university-of-seoul:sumo-traffic-v2:DL1ENw1g"
    )
    base_model = "gpt-4.1-mini"

    samples = load_validation_set(n_samples=n_samples)
    if not samples:
        print("No validation samples found.")
        return None

    print(f"\n{'='*60}")
    print(f"  ACCURACY BENCHMARK: FT vs Base (Ground Truth)")
    print(f"  FT model:   {ft_model}")
    print(f"  Base model: {base_model}")
    print(f"  Samples:    {len(samples)}")
    print(f"{'='*60}\n")

    # Collect per-field errors: {model_type: {field: [absolute_pct_errors]}}
    errors = {
        "ft": {f: [] for f in ACCURACY_FIELDS},
        "base": {f: [] for f in ACCURACY_FIELDS},
    }
    details = []

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        gt = sample["ground_truth"]
        sys_prompt = sample["system_prompt"]
        print(f"[{i+1}/{len(samples)}] {prompt[:50]}...")

        for model_type, model_id in [("ft", ft_model), ("base", base_model)]:
            try:
                raw, latency = call_model(prompt, model_id, sys_prompt, api_key)
                match = re.search(r'\{[\s\S]*\}', raw)
                if not match:
                    print(f"  {model_type}: JSON parse failed")
                    continue
                pred = json.loads(match.group())
            except Exception as e:
                print(f"  {model_type}: error - {e}")
                continue

            detail = {"model": model_type, "prompt": prompt, "latency_ms": latency}
            for f in ACCURACY_FIELDS:
                gt_val = gt.get(f)
                pred_val = pred.get(f)
                if gt_val is not None and pred_val is not None:
                    try:
                        gt_f = float(gt_val)
                        pred_f = float(pred_val)
                        if gt_f != 0:
                            pct_err = abs(pred_f - gt_f) / abs(gt_f) * 100
                            errors[model_type][f].append(pct_err)
                            detail[f"gt_{f}"] = gt_f
                            detail[f"pred_{f}"] = pred_f
                            detail[f"err_{f}"] = round(pct_err, 1)
                    except (ValueError, TypeError):
                        pass
            details.append(detail)

            status = "ok"
            key_err = errors[model_type]["speed_kmh"][-1] if errors[model_type]["speed_kmh"] else None
            if key_err is not None:
                status = f"speed err {key_err:.1f}%"
            print(f"  {model_type}: {status} ({latency}ms)")

    # Build accuracy report
    report = {}
    for model_type in ["ft", "base"]:
        field_mape = {}
        for f in ACCURACY_FIELDS:
            vals = errors[model_type][f]
            if vals:
                field_mape[f] = {
                    "mape": round(statistics.mean(vals), 1),
                    "median_ape": round(statistics.median(vals), 1),
                    "max_ape": round(max(vals), 1),
                    "n": len(vals),
                }
        overall_errors = []
        for f in ACCURACY_FIELDS:
            overall_errors.extend(errors[model_type][f])
        report[model_type] = {
            "overall_mape": round(statistics.mean(overall_errors), 1) if overall_errors else None,
            "fields": field_mape,
        }

    print_accuracy_report(report)

    if output_dir:
        save_accuracy_report(report, details, output_dir)

    return report


def print_accuracy_report(report: dict):
    """Print accuracy comparison table."""
    print(f"\n{'='*60}")
    print(f"  ACCURACY RESULTS (MAPE %)")
    print(f"{'='*60}")
    print(f"{'Field':<20} {'Fine-tuned':>12} {'Base':>12}")
    print(f"{'-'*44}")

    all_fields = set()
    for mt in ["ft", "base"]:
        all_fields.update(report[mt]["fields"].keys())

    for f in ACCURACY_FIELDS:
        if f not in all_fields:
            continue
        ft_mape = report["ft"]["fields"].get(f, {}).get("mape", "-")
        base_mape = report["base"]["fields"].get(f, {}).get("mape", "-")

        ft_str = f"{ft_mape}%" if ft_mape != "-" else "-"
        base_str = f"{base_mape}%" if base_mape != "-" else "-"

        if ft_mape != "-" and base_mape != "-":
            if ft_mape <= base_mape:
                ft_str = f"*{ft_mape}%*"
            else:
                base_str = f"*{base_mape}%*"

        print(f"{f:<20} {ft_str:>12} {base_str:>12}")

    ft_overall = report["ft"]["overall_mape"]
    base_overall = report["base"]["overall_mape"]
    print(f"{'-'*44}")

    ft_o = f"{ft_overall}%" if ft_overall else "-"
    base_o = f"{base_overall}%" if base_overall else "-"
    if ft_overall and base_overall:
        if ft_overall <= base_overall:
            ft_o = f"*{ft_overall}%*"
        else:
            base_o = f"*{base_overall}%*"
    print(f"{'OVERALL':<20} {ft_o:>12} {base_o:>12}")
    print(f"\n  * = better (lower MAPE)\n")


def save_accuracy_report(report: dict, details: list, output_dir: str):
    """Save accuracy benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    path = os.path.join(output_dir, f"accuracy_{ts}.json")
    with open(path, "w") as f:
        json.dump({"timestamp": ts, "report": report}, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {path}")

    # Detail JSONL
    detail_path = os.path.join(output_dir, f"accuracy_detail_{ts}.jsonl")
    with open(detail_path, "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Details: {detail_path}")

    # Markdown
    md_path = os.path.join(output_dir, f"accuracy_{ts}.md")
    with open(md_path, "w") as f:
        f.write("## Accuracy Benchmark: Fine-tuned vs Base (MAPE %)\n\n")
        f.write("| Field | Fine-tuned | Base (gpt-4.1-mini) |\n")
        f.write("|-------|-----------|--------------------|\n")
        for field in ACCURACY_FIELDS:
            ft_m = report["ft"]["fields"].get(field, {}).get("mape", "-")
            base_m = report["base"]["fields"].get(field, {}).get("mape", "-")
            f.write(f"| {field} | {ft_m}% | {base_m}% |\n")
        f.write(f"| **Overall** | **{report['ft']['overall_mape']}%** | **{report['base']['overall_mape']}%** |\n")
        f.write(f"\n*Lower MAPE = closer to ground truth*\n")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark: FT vs Base model")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt for consistency (default: 3)")
    parser.add_argument("--samples", type=int, default=30, help="Number of validation samples for accuracy (default: 30)")
    parser.add_argument("--output", type=str, default="evaluation/results", help="Output directory")
    parser.add_argument("--accuracy", action="store_true", help="Run ground truth accuracy benchmark")
    parser.add_argument("--all", action="store_true", help="Run both quality and accuracy benchmarks")
    args = parser.parse_args()

    if args.all:
        run_benchmark(runs_per_prompt=args.runs, output_dir=args.output)
        run_accuracy_benchmark(n_samples=args.samples, output_dir=args.output)
    elif args.accuracy:
        run_accuracy_benchmark(n_samples=args.samples, output_dir=args.output)
    else:
        run_benchmark(runs_per_prompt=args.runs, output_dir=args.output)
