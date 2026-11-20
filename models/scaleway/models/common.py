"""Shared helpers for the Scaleway model provider."""

import re
from collections.abc import Mapping

from dify_plugin.errors.model import CredentialsValidateFailedError

SERVERLESS_BASE_URL = "https://api.scaleway.ai"

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def resolve_endpoint_url(credentials: Mapping) -> str:
    """
    Return the serverless OpenAI-compatible base URL for the given credentials.

    With a `project_id`, usage is billed to that Project:
    https://api.scaleway.ai/<project_id>/v1. Otherwise: https://api.scaleway.ai/v1.
    """
    project_id = (credentials.get("project_id") or "").strip()
    if project_id:
        if not _UUID_RE.match(project_id):
            raise CredentialsValidateFailedError(
                f"Invalid Scaleway Project ID: {project_id!r}. Expected a UUID."
            )
        return f"{SERVERLESS_BASE_URL}/{project_id}/v1"
    return f"{SERVERLESS_BASE_URL}/v1"


def prepare_credentials(credentials: Mapping) -> dict:
    """
    Return a copy of `credentials` with `endpoint_url` set for the OAICompat base classes.

    Any `endpoint_url` already present is overwritten: only the Scaleway serverless
    endpoint is supported, so the API key is never sent anywhere else.
    """
    prepared = dict(credentials)
    prepared["endpoint_url"] = resolve_endpoint_url(credentials)
    return prepared
