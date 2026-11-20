from typing import IO, Optional

from dify_plugin import OAICompatSpeech2TextModel
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import AIModelEntity, FetchFrom, ModelPropertyKey, ModelType

from models.common import prepare_credentials

# Limits of Scaleway's /v1/audio/transcriptions endpoint.
FILE_UPLOAD_LIMIT_MB = 25
SUPPORTED_FILE_EXTENSIONS = "flac,m4a,mpeg,mp2,mp3,mp4,ogg,wav,webm"


class ScalewaySpeech2TextModel(OAICompatSpeech2TextModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        file: IO[bytes],
        user: Optional[str] = None,
    ) -> str:
        return super()._invoke(model, prepare_credentials(credentials), file, user)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        super().validate_credentials(model, prepare_credentials(credentials))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        # The SDK base class returns None, which would make custom speech-to-text
        # models impossible to select once added.
        return AIModelEntity(
            model=model,
            label=I18nObject(en_us=model),
            model_type=ModelType.SPEECH2TEXT,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.FILE_UPLOAD_LIMIT: FILE_UPLOAD_LIMIT_MB,
                ModelPropertyKey.SUPPORTED_FILE_EXTENSIONS: SUPPORTED_FILE_EXTENSIONS,
            },
            parameter_rules=[],
        )
