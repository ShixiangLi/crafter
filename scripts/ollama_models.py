"""Model-directory selection and preflight checks for GPU-local Ollama services."""
import json
import os
import shlex
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.config import load_config


def model_directory(model):
    explicit = os.environ.get("OLLAMA_MODELS")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
            raise ValueError(f"OLLAMA_MODELS is not a readable directory: {path}")
        return path
    local = Path.home() / ".ollama/models"
    name, _, tag = model.rpartition(":") if ":" in model.rsplit("/", 1)[-1] else (model, "", "latest")
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", *parts]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai", *parts]
    for path in (local, Path("/usr/share/ollama/.ollama/models"), Path("/var/lib/ollama/models")):
        manifest = path.joinpath("manifests", *parts, tag)
        if os.access(manifest, os.R_OK):
            return path.resolve()
    return local


def check_model(config):
    if config.client.client_name not in {"openai_compatible", "vllm"}:
        raise ValueError("--gpu requires an OpenAI-compatible Ollama client")
    base = config.client.base_url.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(
        base + "/api/show", data=json.dumps({"model": config.client.model_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        # The GPU launcher only targets loopback services.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10):
            pass
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        command = f"OLLAMA_HOST={shlex.quote(base)} ollama pull {shlex.quote(config.client.model_id)}"
        raise ValueError(f"Model {config.client.model_id!r} is missing at {base}. Run: {command}") from error


def main():
    mode, *args = sys.argv[1:]
    _, _, config = load_config(args)
    if mode == "--directory":
        print(model_directory(config.client.model_id))
    elif mode == "--check":
        check_model(config)
        print(f"Model ready: {config.client.model_id} at {config.client.base_url}", file=sys.stderr)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Ollama preflight: {error}", file=sys.stderr)
        raise SystemExit(1)
