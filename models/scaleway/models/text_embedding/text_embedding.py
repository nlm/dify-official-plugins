from typing import Optional

from dify_plugin import OAICompatEmbeddingModel
from dify_plugin.entities.model import AIModelEntity, EmbeddingInputType, ModelPropertyKey
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult

from models.common import prepare_credentials


class ScalewayTextEmbeddingModel(OAICompatEmbeddingModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        return super()._invoke(model, prepare_credentials(credentials), texts, user, input_type)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        super().validate_credentials(model, prepare_credentials(credentials))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        schema = super().get_customizable_model_schema(model, prepare_credentials(credentials))
        # The base class always uses 1; honor the "Max chunks per batch" form field.
        schema.model_properties[ModelPropertyKey.MAX_CHUNKS] = max(
            1, int(credentials.get("max_chunks") or 1)
        )
        return schema
