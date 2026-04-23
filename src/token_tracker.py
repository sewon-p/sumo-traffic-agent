"""
LLM Token Usage Tracker

Accumulates token counts, latency, and estimated cost per API call.
Prints a summary at the end of each session.

Usage:
    from src.token_tracker import tracker
    tracker.record("openai", "gpt-4.1-mini", input_tokens=120, output_tokens=45, latency_ms=830)
    tracker.summary()  # prints accumulated stats
"""

import time
import threading
from dataclasses import dataclass, field


# Approximate pricing per 1M tokens (USD) — update as needed
_COST_TABLE: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5.4": (2.00, 8.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-3.1-flash-lite-preview": (0.075, 0.30),
}


@dataclass
class CallRecord:
    provider: str
    model: str
    caller: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: float


@dataclass
class TokenTracker:
    records: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        key = model
        if key not in _COST_TABLE:
            for k in _COST_TABLE:
                if k in model or model in k:
                    key = k
                    break
            else:
                return 0.0
        inp_rate, out_rate = _COST_TABLE[key]
        return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        caller: str = "",
    ) -> None:
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        entry = CallRecord(
            provider=provider,
            model=model,
            caller=caller,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            timestamp=time.time(),
        )
        with self._lock:
            self.records.append(entry)
        print(
            f"[TOKEN] {provider}/{model} | "
            f"in={input_tokens} out={output_tokens} | "
            f"{latency_ms:.0f}ms | ${cost:.6f}"
            f"{f' | {caller}' if caller else ''}"
        )

    def summary(self) -> dict:
        with self._lock:
            if not self.records:
                print("[TOKEN] No API calls recorded.")
                return {}

            total_in = sum(r.input_tokens for r in self.records)
            total_out = sum(r.output_tokens for r in self.records)
            total_cost = sum(r.cost_usd for r in self.records)
            total_latency = sum(r.latency_ms for r in self.records)
            count = len(self.records)

            # Per-model breakdown
            by_model: dict[str, dict] = {}
            for r in self.records:
                key = f"{r.provider}/{r.model}"
                if key not in by_model:
                    by_model[key] = {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "latency": 0.0}
                by_model[key]["calls"] += 1
                by_model[key]["in"] += r.input_tokens
                by_model[key]["out"] += r.output_tokens
                by_model[key]["cost"] += r.cost_usd
                by_model[key]["latency"] += r.latency_ms

            print("\n" + "=" * 60)
            print("  TOKEN USAGE SUMMARY")
            print("=" * 60)
            print(f"  Total calls:    {count}")
            print(f"  Input tokens:   {total_in:,}")
            print(f"  Output tokens:  {total_out:,}")
            print(f"  Total tokens:   {total_in + total_out:,}")
            print(f"  Total latency:  {total_latency:,.0f}ms ({total_latency/1000:.1f}s)")
            print(f"  Estimated cost: ${total_cost:.6f}")
            print("-" * 60)
            for model_key, stats in by_model.items():
                avg_lat = stats["latency"] / stats["calls"]
                print(
                    f"  {model_key}: {stats['calls']} calls | "
                    f"in={stats['in']:,} out={stats['out']:,} | "
                    f"avg {avg_lat:.0f}ms | ${stats['cost']:.6f}"
                )
            print("=" * 60 + "\n")

            return {
                "total_calls": count,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_cost_usd": total_cost,
                "total_latency_ms": total_latency,
                "by_model": by_model,
            }

    def reset(self) -> None:
        with self._lock:
            self.records.clear()

    def to_list(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "caller": r.caller,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "timestamp": r.timestamp,
                }
                for r in self.records
            ]


# Global singleton
tracker = TokenTracker()
