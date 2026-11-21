"""Rebuild the livekit/* entries of the price map from a LiveKit Inference pricing snapshot.

Snapshot: the model list embedded in https://livekit.com/pricing/inference
(model_id, provider, rates per million tokens), saved as livekit_inference_models.json.

Keys:
  livekit/<model_id>             routing left to LiveKit, priced as its default provider
  livekit/<provider>/<model_id>  provider pinned (X-LiveKit-Inference-Provider)
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).parent
PRICE_MAP = HERE.parent / "model_prices_and_context_window_backup.json"
SNAPSHOT = HERE / "livekit_inference_models.json"
SOURCE = "https://livekit.com/pricing/inference"
PRIORITY_SUFFIX = "-priority"
RATE_FIELDS = {
    "input_tokens": "input_cost_per_token",
    "output_tokens": "output_cost_per_token",
    "cached_input_tokens": "cache_read_input_token_cost",
    "cache_write_tokens": "cache_creation_input_token_cost",
}
PRIORITY_FIELDS = {
    "input_tokens": "input_cost_per_token_priority",
    "output_tokens": "output_cost_per_token_priority",
    "cached_input_tokens": "cache_read_input_token_cost_priority",
}


def per_token(rate: dict) -> float:
    assert rate["measure"] == "tokens", rate
    return float(Decimal(rate["build"]) / Decimal(rate["per"]))


def entry(row: dict, priority: dict | None) -> dict:
    out: dict = {field: per_token(r) for r in row["rates"] if (field := RATE_FIELDS.get(r["metric"]))}
    if priority is not None:
        out |= {field: per_token(r) for r in priority["rates"] if (field := PRIORITY_FIELDS.get(r["metric"]))}
    out |= {"litellm_provider": "livekit", "mode": "chat", "source": SOURCE}
    if "decision" not in row["model_id"]:
        out |= {"supports_function_calling": True, "supports_system_messages": True, "supports_tool_choice": True}
    if row["is_reasoning"]:
        out["supports_reasoning"] = True
    out["metadata"] = {"inference_provider": row["provider"], "pricing_as_of": row["as_of"]}
    return dict(sorted(out.items()))


def build() -> dict:
    rows = [r for r in json.loads(SNAPSHOT.read_text()) if r["use_case"] == "llm"]
    priority = {
        (r["model_id"].removesuffix(PRIORITY_SUFFIX), r["provider"]): r
        for r in rows
        if r["model_id"].endswith(PRIORITY_SUFFIX)
    }
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if not r["model_id"].endswith(PRIORITY_SUFFIX):
            by_model.setdefault(r["model_id"], []).append(r)
    out: dict = {}
    for model_id, variants in by_model.items():
        default = next((v for v in variants if v["provider"] == v["creator"]), variants[0])
        out[f"livekit/{model_id}"] = entry(default, priority.get((model_id, default["provider"])))
        for v in variants:
            out[f"livekit/{v['provider']}/{model_id}"] = entry(v, priority.get((model_id, v["provider"])))
    return out


def main() -> None:
    prices = json.loads(PRICE_MAP.read_text())
    kept = {k: v for k, v in prices.items() if not k.startswith("livekit/")}
    generated = build()
    PRICE_MAP.write_text(json.dumps(kept | generated, indent=4) + "\n")
    print(f"livekit entries: {len(generated)}")


if __name__ == "__main__":
    main()
