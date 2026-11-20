import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dify_plugin.entities.model import AIModelEntity  # noqa: E402
from dify_plugin.errors.model import CredentialsValidateFailedError  # noqa: E402

from models.common import prepare_credentials, resolve_endpoint_url  # noqa: E402
from models.llm.llm import ScalewayLargeLanguageModel  # noqa: E402

PROJECT_ID = "11111111-2222-3333-4444-555555555555"


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({"api_key": "k"}, "https://api.scaleway.ai/v1"),
        ({"api_key": "k", "project_id": ""}, "https://api.scaleway.ai/v1"),
        ({"api_key": "k", "project_id": f" {PROJECT_ID} "}, f"https://api.scaleway.ai/{PROJECT_ID}/v1"),
    ],
)
def test_resolve_endpoint_url(credentials, expected):
    assert resolve_endpoint_url(credentials) == expected


@pytest.mark.parametrize("project_id", ["not-a-uuid", "../evil", f"{PROJECT_ID}/../x"])
def test_resolve_endpoint_url_rejects_invalid_project_id(project_id):
    with pytest.raises(CredentialsValidateFailedError):
        resolve_endpoint_url({"project_id": project_id})


@pytest.mark.parametrize(
    "endpoint_url", ["https://abc.ifr.fr-par.scaleway.com/v1", "http://attacker.example/v1"]
)
def test_prepare_credentials_ignores_any_endpoint_url(endpoint_url):
    """Only the serverless endpoint is supported: a stray endpoint_url never receives the key."""
    prepared = prepare_credentials({"api_key": "k", "endpoint_url": endpoint_url})
    assert prepared["endpoint_url"] == "https://api.scaleway.ai/v1"


def test_prepare_credentials_does_not_mutate_input():
    credentials = {"api_key": "k", "project_id": PROJECT_ID}
    prepared = prepare_credentials(credentials)
    assert "endpoint_url" not in credentials
    assert prepared["endpoint_url"] == f"https://api.scaleway.ai/{PROJECT_ID}/v1"


def test_llm_prepare_enables_tool_calls_by_default():
    prepared = ScalewayLargeLanguageModel._prepare({"api_key": "k"})
    assert prepared["mode"] == "chat"
    assert prepared["function_calling_type"] == "tool_call"
    assert prepared["stream_function_calling"] == "supported"


def test_llm_prepare_respects_custom_model_setting():
    prepared = ScalewayLargeLanguageModel._prepare({"api_key": "k", "function_calling_type": "no_call"})
    assert prepared["function_calling_type"] == "no_call"


MODEL_YAMLS = sorted(p for p in (ROOT / "models").rglob("*.yaml") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", MODEL_YAMLS, ids=lambda p: p.stem)
def test_model_yaml_is_valid(path):
    data = yaml.safe_load(path.read_text())
    data.setdefault("label", {"en_US": data["model"]})
    data["fetch_from"] = "predefined-model"
    AIModelEntity(**data)


@pytest.mark.parametrize("model_type", ["llm", "text_embedding"])
def test_position_files_match_models(model_type):
    directory = ROOT / "models" / model_type
    positions = yaml.safe_load((directory / "_position.yaml").read_text())
    models = {yaml.safe_load(p.read_text())["model"] for p in directory.glob("*.yaml") if not p.name.startswith("_")}
    assert sorted(positions) == sorted(models)


def test_llm_request_targets_scaleway_and_forwards_tools(monkeypatch):
    from dify_plugin.entities.model.message import PromptMessageTool, UserPromptMessage

    captured = {}

    class FakeResponse:
        status_code = 200
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def json(self):
            return {
                "id": "x",
                "model": "llama-3.3-70b-instruct",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(url, headers=None, data=None, **kwargs):
        import json

        captured.update(url=url, headers=headers, body=json.loads(data))
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    model = ScalewayLargeLanguageModel(model_schemas=[])
    tool = PromptMessageTool(name="get_weather", description="d", parameters={"type": "object", "properties": {}})
    model._invoke(
        model="llama-3.3-70b-instruct",
        credentials={"api_key": "secret", "project_id": PROJECT_ID},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        tools=[tool],
        stream=False,
    )
    assert captured["url"] == f"https://api.scaleway.ai/{PROJECT_ID}/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "llama-3.3-70b-instruct"
    assert captured["body"]["tools"][0]["function"]["name"] == "get_weather"


def test_get_num_tokens_counts_tool_messages():
    from dify_plugin.entities.model.message import (
        AssistantPromptMessage,
        ToolPromptMessage,
        UserPromptMessage,
    )

    tool_output = "lorem ipsum dolor sit amet " * 400
    messages = [
        UserPromptMessage(content="search"),
        AssistantPromptMessage(
            content="",
            tool_calls=[
                AssistantPromptMessage.ToolCall(
                    id="call_1",
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name="search", arguments="{}"
                    ),
                )
            ],
        ),
        ToolPromptMessage(content=tool_output, tool_call_id="call_1", name="search"),
    ]
    model = ScalewayLargeLanguageModel(model_schemas=[])
    count = model.get_num_tokens("llama-3.3-70b-instruct", {"api_key": "k"}, messages)
    assert count >= 2000  # the tool output alone is ~2000 tokens
