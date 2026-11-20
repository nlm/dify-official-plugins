"""Shared helpers for the Scaleway Generative APIs model provider."""

import re
from collections.abc import Mapping
from urllib.parse import urlparse

from dify_plugin.errors.model import CredentialsValidateFailedError

SERVERLESS_BASE_URL = "https://api.scaleway.ai"

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def resolve_endpoint_url(credentials: Mapping) -> str:
    """
    Return the OpenAI-compatible base URL (ending in /v1) for the given credentials.

    - `endpoint_url` set (customizable models): a Dedicated Deployment URL such as
      https://<deployment-uuid>.ifr.fr-par.scaleway.com/v1. Must use https; `/v1` is appended
      if missing.
    - `project_id` set: https://api.scaleway.ai/<project_id>/v1, so usage is billed to
      that project instead of the default one.
    - otherwise: https://api.scaleway.ai/v1.
    """
    custom = (credentials.get("endpoint_url") or "").strip()
    if custom:
        parsed = urlparse(custom)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            # The API secret key is sent with every request, so never send it in clear text.
            raise CredentialsValidateFailedError(
                f"Invalid endpoint URL: {custom!r}. An https:// URL is required, e.g. "
                "https://<deployment-uuid>.ifr.fr-par.scaleway.com/v1"
            )
        custom = custom.rstrip("/")
        return custom if custom.endswith("/v1") else f"{custom}/v1"

    project_id = (credentials.get("project_id") or "").strip()
    if project_id:
        if not _UUID_RE.match(project_id):
            raise CredentialsValidateFailedError(
                f"Invalid Scaleway Project ID: {project_id!r}. Expected a UUID."
            )
        return f"{SERVERLESS_BASE_URL}/{project_id}/v1"

    return f"{SERVERLESS_BASE_URL}/v1"


def prepare_credentials(credentials: Mapping) -> dict:
    """Return a copy of `credentials` with `endpoint_url` resolved for the OAICompat base classes."""
    prepared = dict(credentials)
    prepared["endpoint_url"] = resolve_endpoint_url(credentials)
    return prepared
