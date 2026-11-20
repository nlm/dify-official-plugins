#!/usr/bin/env python3
"""Generate the Scaleway model YAML files from the public Scaleway product catalog.

The catalog (no authentication needed) supplies which serverless models are active,
their EUR prices, whether they accept images, and their supported reasoning efforts.
It does not expose context windows or output limits, so those come from
`scripts/model_profiles.json`, maintained by hand from the Scaleway documentation.

Usage, from the plugin directory:

    python scripts/sync_catalog.py                    # fetch the catalog and rewrite the YAML files
    python scripts/sync_catalog.py --check            # exit 1 if the files differ from the catalog
    python scripts/sync_catalog.py --input snap.json  # use a saved catalog snapshot instead
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "scripts" / "model_profiles.json"
CATALOG_URL = "https://api.scaleway.com/product-catalog/v2alpha1/public-catalog/products"
REGION = "fr-par"
ACTIVE_STATUSES = {"preview", "general_availability"}
MODEL_TYPES = ("llm", "text_embedding", "speech2text")
# Model types that have a _position.yaml (speech2text has a single model and none).
POSITIONED_TYPES = ("llm", "text_embedding")
DEFAULT_MAX_TOKENS = 4096
PAGE_SIZE = 100


class CatalogError(Exception):
    pass


# --------------------------------------------------------------------------- catalog


def fetch_catalog() -> list[dict[str, Any]]:
    """Fetch every Generative APIs product from the public catalog, following pagination."""
    products: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {"product_types": "generative_apis", "page_size": PAGE_SIZE, "page": page}
        )
        with urllib.request.urlopen(f"{CATALOG_URL}?{query}", timeout=30) as response:
            payload = json.load(response)
        batch = payload.get("products") or []
        products.extend(batch)
        if not batch or len(products) >= int(payload.get("total_count", 0)):
            return products
        page += 1


def load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    products = payload.get("products") if isinstance(payload, dict) else payload
    if not isinstance(products, list):
        raise CatalogError(f"{path}: expected a 'products' array")
    return products


def _properties(product: dict[str, Any]) -> dict[str, Any]:
    return (product.get("properties") or {}).get("generative_apis") or {}


def _model_type(tasks: set[str]) -> str | None:
    if "embeddings" in tasks:
        return "text_embedding"
    if "chat" in tasks:
        return "llm"
    if "audio_transcription" in tasks:
        return "speech2text"
    return None  # e.g. OCR-only models, which Dify has no model type for


def _price_per_million(product: dict[str, Any]) -> Decimal:
    unit = product.get("unit_of_measure") or {}
    if unit.get("unit") != "token":
        raise CatalogError(f"{product['sku']}: expected a per-token price, got {unit}")
    money = product["price"]["retail_price"]
    if money.get("currency_code") != "EUR":
        raise CatalogError(f"{product['sku']}: expected EUR, got {money.get('currency_code')}")
    per_unit = Decimal(money.get("units", 0)) + Decimal(money.get("nanos", 0)) / Decimal(10**9)
    return per_unit * Decimal(1_000_000) / Decimal(unit["size"])


def collect_models(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group realtime catalog entries for the region into one record per active model."""
    models: dict[str, dict[str, Any]] = {}
    for product in products:
        props = _properties(product)
        if props.get("consumption_mode") != "realtime":
            continue  # batch prices are a fixed discount of the realtime ones
        if (product.get("locality") or {}).get("region") != REGION:
            continue
        if product.get("status") not in ACTIVE_STATUSES:
            continue
        name = product["product"]
        tasks = set(props.get("tasks") or [])
        model = models.setdefault(
            name,
            {
                "id": name,
                "type": _model_type(tasks),
                "vision": "vision" in tasks,
                "reasoning_values": list(props.get("supported_reasoning_values") or []),
                "prices": {},
            },
        )
        token_type = props.get("token_type")
        if token_type in ("input_token", "output_token") and product.get("price"):
            model["prices"][token_type] = _price_per_million(product)
    return models


# --------------------------------------------------------------------------- rendering


def _format_price(value: Decimal) -> str:
    """Render 1.8 as '1.80' and 0.003 as '0.003', so prices read like Scaleway's price list."""
    value = value.normalize()
    return f"{value:.2f}" if value.as_tuple().exponent >= -2 else f"{value:f}"


def _render_pricing(prices: dict[str, Decimal]) -> list[str]:
    if "input_token" not in prices:
        return []
    lines = ["pricing:", f"  input: '{_format_price(prices['input_token'])}'"]
    if "output_token" in prices:
        lines.append(f"  output: '{_format_price(prices['output_token'])}'")
    return lines + ["  unit: '0.000001'", "  currency: EUR"]


def render_llm(model: dict[str, Any], profile: dict[str, Any]) -> str:
    max_tokens = int(profile["max_tokens"])
    features = ["agent-thought", "tool-call", "multi-tool-call", "stream-tool-call"]
    if model["vision"]:
        features.append("vision")
    lines = [f"model: {model['id']}", "label:", f"  en_US: {profile['label']}"]
    lines += ["model_type: llm", "features:", *(f"  - {f}" for f in features)]
    lines += ["model_properties:", "  mode: chat", f"  context_size: {int(profile['context_size'])}"]
    lines += [
        "parameter_rules:",
        "  - name: temperature",
        "    use_template: temperature",
        "  - name: top_p",
        "    use_template: top_p",
        "  - name: presence_penalty",
        "    use_template: presence_penalty",
        "  - name: frequency_penalty",
        "    use_template: frequency_penalty",
        "  - name: max_tokens",
        "    use_template: max_tokens",
        f"    default: {min(DEFAULT_MAX_TOKENS, max_tokens)}",
        "    min: 1",
        f"    max: {max_tokens}",
        "  - name: response_format",
        "    label:",
        "      en_US: Response Format",
        "    type: string",
        "    help:",
        "      en_US: specifying the format that the model must output",
        "    required: false",
        "    options:",
        "      - text",
        "      - json_object",
        "      - json_schema",
        "  - name: json_schema",
        "    use_template: json_schema",
        "    required: false",
    ]
    if model["reasoning_values"]:
        lines += [
            "  - name: reasoning_effort",
            "    label:",
            "      en_US: Reasoning Effort",
            "    type: string",
            "    help:",
            "      en_US: Constrains how much the model reasons before answering."
            " Leave unset to use the model default.",
            "    required: false",
            "    options:",
            *(f"      - {value}" for value in model["reasoning_values"]),
        ]
    return "\n".join(lines + _render_pricing(model["prices"])) + "\n"


def render_text_embedding(model: dict[str, Any], profile: dict[str, Any]) -> str:
    lines = [
        f"model: {model['id']}",
        "model_type: text-embedding",
        "model_properties:",
        f"  context_size: {int(profile['context_size'])}",
        f"  max_chunks: {int(profile['max_chunks'])}",
    ]
    return "\n".join(lines + _render_pricing(model["prices"])) + "\n"


def render_speech2text(model: dict[str, Any], profile: dict[str, Any]) -> str:
    # Whisper is billed per audio minute, which Dify pricing cannot express.
    lines = [
        f"model: {model['id']}",
        "model_type: speech2text",
        "model_properties:",
        f"  file_upload_limit: {int(profile['file_upload_limit'])}",
        f"  supported_file_extensions: {profile['supported_file_extensions']}",
    ]
    return "\n".join(lines) + "\n"


RENDERERS = {
    "llm": render_llm,
    "text_embedding": render_text_embedding,
    "speech2text": render_speech2text,
}


def render_catalog(
    products: list[dict[str, Any]], profiles: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Return {model_type: {file name: content}} for every active catalog model."""
    models = collect_models(products)
    errors: list[str] = []
    outputs: dict[str, dict[str, str]] = {kind: {} for kind in MODEL_TYPES}

    for model in models.values():
        kind = model["type"]
        if kind is None:
            print(f"skipping {model['id']}: no matching Dify model type", file=sys.stderr)
            continue
        if model["id"] not in profiles.get(kind, {}):
            errors.append(
                f"new {kind} model {model['id']!r} in the catalog: add it to "
                f"scripts/model_profiles.json under {kind!r}"
            )

    for kind in MODEL_TYPES:
        rendered_ids = []
        for model_id, profile in profiles.get(kind, {}).items():
            model = models.get(model_id)
            if model is None or model["type"] != kind:
                errors.append(
                    f"{kind} model {model_id!r} is no longer active in the catalog: "
                    "remove it from scripts/model_profiles.json"
                )
                continue
            outputs[kind][f"{model_id}.yaml"] = RENDERERS[kind](model, profile)
            rendered_ids.append(model_id)
        if kind in POSITIONED_TYPES:
            outputs[kind]["_position.yaml"] = "".join(f"- {i}\n" for i in rendered_ids)

    if errors:
        raise CatalogError("\n  ".join(["catalog and model_profiles.json disagree:", *errors]))
    return outputs


# --------------------------------------------------------------------------- files


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sync_files(files: dict[str, str], *, check: bool, output_dir: Path) -> bool:
    current = {path.name: path.read_text(encoding="utf-8") for path in output_dir.glob("*.yaml")}
    stale = set(current) - files.keys()
    changed = {name for name, content in files.items() if current.get(name) != content}

    if check:
        for name in sorted(changed | stale):
            print(f"catalog drift: {_display(output_dir / name)}", file=sys.stderr)
        return not (changed or stale)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in stale:
        (output_dir / name).unlink()
    for name in changed:
        (output_dir / name).write_text(files[name], encoding="utf-8")
    if changed or stale:
        print(f"{_display(output_dir)}: {len(changed)} written, {len(stale)} removed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Scaleway Generative APIs models into Dify YAML")
    parser.add_argument("--input", type=Path, help="saved catalog JSON (default: fetch it)")
    parser.add_argument("--check", action="store_true", help="fail if generated files differ")
    args = parser.parse_args()

    try:
        products = load_catalog(args.input) if args.input else fetch_catalog()
        profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        outputs = render_catalog(products, profiles)
        results = [
            sync_files(files, check=args.check, output_dir=ROOT / "models" / kind)
            for kind, files in outputs.items()
        ]
    except (CatalogError, OSError, KeyError, ValueError) as ex:
        print(f"catalog sync failed: {ex}", file=sys.stderr)
        return 1
    if args.check and all(results):
        print("model files match the catalog")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
