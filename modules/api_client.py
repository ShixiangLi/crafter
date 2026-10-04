"""OpenAI-compatible transport; reuse BALROG generation, retries and accounting."""
import os
from functools import partial

from openai import OpenAI
from balrog.client import OpenAIWrapper, create_llm_client as native_client


def api_key(config):
    name = config.get("api_key_env")
    if name is None:
        return "EMPTY"
    if not isinstance(name, str) or not name.strip():
        raise ValueError("api.api_key_env must be an environment variable name or null")
    value = os.environ.get(name)
    if not value or not value.strip():
        raise ValueError(f"Required API key environment variable is unset or empty: {name}")
    return value


class CompatibleClient(OpenAIWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def _initialize_client(self):
        if not self._initialized:
            self.client = OpenAI(base_url=self.base_url, api_key=api_key(self.config),
                                 timeout=self.timeout, max_retries=0)
            self._initialized = True


def create_llm_client(config):
    if config.client_name in {"openai_compatible", "vllm"}:
        return partial(CompatibleClient, config)
    return native_client(config)
