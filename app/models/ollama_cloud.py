"""Adapter for the direct Ollama Cloud API (``https://ollama.com/api``).

Authentication uses ``Authorization: Bearer $OLLAMA_API_KEY``. The key is read
from the environment variable named in the registry and is never logged.
"""

from __future__ import annotations

from app.core.exceptions import LLMError
from app.models.base_adapter import AvailabilityResult
from app.models.ollama_local import OllamaLocalAdapter


class OllamaCloudAdapter(OllamaLocalAdapter):
    """Same wire protocol as local Ollama, different host and auth."""

    def _base_url(self) -> str:
        return (self.config.endpoint or self.settings.ollama_cloud_host).rstrip("/")

    def _headers(self) -> dict[str, str]:
        secret = self.settings.secret_for(self.config.auth_env)
        return {"Authorization": f"Bearer {secret.get_secret_value()}"} if secret else {}

    def check_availability(self) -> AvailabilityResult:
        """Check credentials and that the cloud model is listed."""
        if not self.settings.secret_for(self.config.auth_env):
            return AvailabilityResult(False, f"Environment variable {self.config.auth_env} is not set")
        try:
            models = self.list_models()
        except LLMError as exc:
            return AvailabilityResult(False, f"Ollama Cloud not reachable: {exc}")
        names = {m.get("name") for m in models} | {m.get("model") for m in models}
        if self.config.model_id in names:
            return AvailabilityResult(True, "listed by Ollama Cloud")
        return AvailabilityResult(False, f"Model {self.config.model_id!r} is not offered by Ollama Cloud")
