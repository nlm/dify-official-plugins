from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool

from models.common import prepare_credentials


class ScalewayLargeLanguageModel(OAICompatLargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        return super()._invoke(
            model,
            self._prepare(credentials),
            prompt_messages,
            model_parameters,
            tools,
            stop,
            stream,
            user,
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        super().validate_credentials(model, self._prepare(credentials))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        return super().get_customizable_model_schema(model, self._prepare(credentials))

    @staticmethod
    def _prepare(credentials: dict) -> dict:
        prepared = prepare_credentials(credentials)
        prepared["mode"] = "chat"
        # The OAICompat base class only forwards `tools` when this is set. All Scaleway
        # serverless chat models support OpenAI-style tool calling; Dify only passes
        # tools for models whose YAML declares the `tool-call` feature.
        prepared.setdefault("function_calling_type", "tool_call")
        prepared.setdefault("stream_function_calling", "supported")
        return prepared
