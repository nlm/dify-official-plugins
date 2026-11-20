import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_catalog import (  # noqa: E402
    PROFILES_PATH,
    CatalogError,
    _format_price,
    render_catalog,
    sync_files,
)


def _product(name, tasks, token_type, nanos, *, status="general_availability", mode="realtime",
             reasoning=(), unit=("token", 1000)):
    return {
        "sku": f"/ai/generative_apis/consumption/{name}_{token_type}_{mode}/fr-par",
        "product": name,
        "status": status,
        "locality": {"region": "fr-par"},
        "price": {"retail_price": {"currency_code": "EUR", "units": 0, "nanos": nanos}},
        "unit_of_measure": {"unit": unit[0], "size": unit[1]},
        "properties": {
            "generative_apis": {
                "consumption_mode": mode,
                "tasks": list(tasks),
                "token_type": token_type,
                "supported_reasoning_values": list(reasoning),
            }
        },
    }


def _catalog():
    return [
        _product("chat-a", ["chat", "vision"], "input_token", 600_000, reasoning=["none", "high"]),
        _product("chat-a", ["chat", "vision"], "input_cached_token", 120_000),
        _product("chat-a", ["chat", "vision"], "output_token", 1_800_000, reasoning=["none", "high"]),
        _product("chat-a", ["chat", "vision"], "input_token", 300_000, mode="batch"),
        _product("chat-b", ["chat"], "input_token", 150_000),
        _product("chat-b", ["chat"], "output_token", 350_000),
        _product("retired-chat", ["chat"], "input_token", 1, status="end_of_life"),
        _product("embed", ["embeddings"], "input_token", 100_000),
        _product("whisper", ["audio_transcription"], "input_audio", 3_000_000, unit=("second", 60)),
        _product("ocr", ["ocr"], "input_token", 2_200_000),
    ]


def _profiles():
    return {
        "llm": {
            "chat-b": {"label": "Chat B", "context_size": 128000, "max_tokens": 2000},
            "chat-a": {"label": "Chat A", "context_size": 256000, "max_tokens": 32000},
        },
        "text_embedding": {"embed": {"context_size": 8192, "max_chunks": 16}},
        "speech2text": {
            "whisper": {"file_upload_limit": 25, "supported_file_extensions": "mp3,wav"}
        },
    }


def test_render_catalog_builds_llm_yaml_from_catalog_and_profile():
    outputs = render_catalog(_catalog(), _profiles())
    chat_a = yaml.safe_load(outputs["llm"]["chat-a.yaml"])

    assert chat_a["model_properties"]["context_size"] == 256000
    assert "vision" in chat_a["features"]
    assert chat_a["pricing"] == {"input": "0.60", "output": "1.80", "unit": "0.000001", "currency": "EUR"}
    rules = {rule["name"]: rule for rule in chat_a["parameter_rules"]}
    assert rules["max_tokens"]["max"] == 32000
    assert rules["max_tokens"]["default"] == 4096
    assert rules["reasoning_effort"]["options"] == ["none", "high"]

    chat_b = yaml.safe_load(outputs["llm"]["chat-b.yaml"])
    assert "vision" not in chat_b["features"]
    rules = {rule["name"]: rule for rule in chat_b["parameter_rules"]}
    assert "reasoning_effort" not in rules
    assert rules["max_tokens"]["default"] == 2000


def test_render_catalog_follows_profile_order_and_skips_inactive_and_unsupported():
    outputs = render_catalog(_catalog(), _profiles())

    assert yaml.safe_load(outputs["llm"]["_position.yaml"]) == ["chat-b", "chat-a"]
    assert yaml.safe_load(outputs["text_embedding"]["_position.yaml"]) == ["embed"]
    assert set(outputs["llm"]) == {"chat-a.yaml", "chat-b.yaml", "_position.yaml"}
    assert set(outputs["speech2text"]) == {"whisper.yaml"}


def test_render_catalog_embedding_and_speech2text():
    outputs = render_catalog(_catalog(), _profiles())

    embed = yaml.safe_load(outputs["text_embedding"]["embed.yaml"])
    assert embed["model_properties"] == {"context_size": 8192, "max_chunks": 16}
    assert embed["pricing"] == {"input": "0.10", "unit": "0.000001", "currency": "EUR"}

    whisper = yaml.safe_load(outputs["speech2text"]["whisper.yaml"])
    assert whisper["model_properties"]["file_upload_limit"] == 25
    assert "pricing" not in whisper


def test_new_catalog_model_without_profile_fails():
    catalog = _catalog() + [_product("chat-new", ["chat"], "input_token", 1)]
    with pytest.raises(CatalogError, match="chat-new"):
        render_catalog(catalog, _profiles())


def test_profile_for_model_no_longer_active_fails():
    profiles = _profiles()
    profiles["llm"]["retired-chat"] = {"label": "Old", "context_size": 1, "max_tokens": 1}
    with pytest.raises(CatalogError, match="retired-chat"):
        render_catalog(_catalog(), profiles)


def test_non_token_price_for_chat_model_fails():
    catalog = [_product("chat-b", ["chat"], "input_token", 1, unit=("second", 60))]
    with pytest.raises(CatalogError, match="per-token"):
        render_catalog(catalog, {"llm": {"chat-b": _profiles()["llm"]["chat-b"]}})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1.8", "1.80"), ("0.1", "0.10"), ("0.9", "0.90"), ("3.6", "3.60"), ("0.003", "0.003"), ("5", "5.00")],
)
def test_format_price(value, expected):
    assert _format_price(Decimal(value)) == expected


def test_sync_files_writes_removes_and_checks(tmp_path):
    (tmp_path / "stale.yaml").write_text("old\n")
    (tmp_path / "keep.yaml").write_text("outdated\n")
    files = {"keep.yaml": "new\n", "_position.yaml": "- keep\n"}

    assert sync_files(files, check=True, output_dir=tmp_path) is False
    assert sync_files(files, check=False, output_dir=tmp_path) is True
    assert sorted(p.name for p in tmp_path.iterdir()) == ["_position.yaml", "keep.yaml"]
    assert (tmp_path / "keep.yaml").read_text() == "new\n"
    assert sync_files(files, check=True, output_dir=tmp_path) is True


def test_committed_profiles_match_committed_model_files():
    """Every profile has a YAML file and every YAML file has a profile, in the same order."""
    profiles = json.loads(PROFILES_PATH.read_text())
    for kind in ("llm", "text_embedding", "speech2text"):
        directory = ROOT / "models" / kind
        on_disk = {p.stem for p in directory.glob("*.yaml") if not p.name.startswith("_")}
        assert set(profiles[kind]) == on_disk, kind
        position = directory / "_position.yaml"
        if position.exists():
            assert yaml.safe_load(position.read_text()) == list(profiles[kind]), kind


def _cached_first_catalog():
    """chat-a with its cached-input entry first: no reasoning values, no vision task."""
    catalog = _catalog()
    cached = catalog.pop(1)
    cached["properties"]["generative_apis"]["tasks"] = ["chat"]
    return [cached, *catalog]


def test_model_metadata_does_not_depend_on_catalog_order():
    expected = render_catalog(_catalog(), _profiles())
    assert render_catalog(_cached_first_catalog(), _profiles()) == expected
    assert render_catalog(list(reversed(_catalog())), _profiles()) == expected


def test_conflicting_reasoning_values_fail():
    catalog = _catalog()
    catalog[2]["properties"]["generative_apis"]["supported_reasoning_values"] = ["low"]
    with pytest.raises(CatalogError, match="disagree on reasoning values"):
        render_catalog(catalog, _profiles())


def test_strings_are_quoted_so_yaml_keeps_them_as_strings():
    catalog = _catalog()
    for product in catalog[:3]:
        product["properties"]["generative_apis"]["supported_reasoning_values"] = ["off", "on"]
    profiles = _profiles()
    profiles["llm"]["chat-a"]["label"] = "Chat: A # preview"
    chat_a = yaml.safe_load(render_catalog(catalog, profiles)["llm"]["chat-a.yaml"])

    assert chat_a["label"]["en_US"] == "Chat: A # preview"
    rules = {rule["name"]: rule for rule in chat_a["parameter_rules"]}
    assert rules["reasoning_effort"]["options"] == ["off", "on"]
