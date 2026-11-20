from typing import IO, Optional

from dify_plugin import OAICompatSpeech2TextModel

from models.common import prepare_credentials


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
