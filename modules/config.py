"""Shared model/evaluation settings plus one agent's configuration."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"


def load_config(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Crafter agents with BALROG")
    parser.add_argument("--config", type=Path,
                        help="Agent config file; default: configs/<agent>.yaml (naive: balrog_baseline.yaml)")
    parser.add_argument("--agent", choices=["naive", "react", "planner", "ours", "spring"])
    parser.add_argument("--model")
    parser.add_argument("--ollama-gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, help="Fix both Crafter map and evaluator seed (same map every episode)")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without running episodes")
    args, overrides = parser.parse_known_args(argv)
    if any(item.startswith("--") for item in overrides):
        parser.error("Unknown option; use --help or BALROG Hydra overrides such as agent.max_text_history=8")
    if args.config is None:
        name = args.agent or "naive"
        args.config = ROOT / "configs" / ("balrog_baseline.yaml" if name == "naive" else f"{name}.yaml")
    try:
        settings = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        parser.error(f"Cannot load configuration: {error}")
    if not isinstance(settings, dict) or not isinstance(experiment, dict):
        parser.error("Common and agent configuration files must be YAML mappings")
    unsupported = experiment.keys() - {"agent", "history_length", "replan_interval", "max_thoughts"}
    if unsupported:
        parser.error(f"Agent config only accepts agent/history_length/replan_interval/max_thoughts; "
                     f"move shared settings to configs/default.yaml: {', '.join(sorted(unsupported))}")
    if experiment.get("agent") not in {"naive", "react", "planner", "ours", "spring"}:
        parser.error("Agent config must select a supported agent")
    if "max_thoughts" in experiment and experiment["agent"] != "react":
        parser.error("max_thoughts is a ReAct algorithm parameter")
    if args.agent is not None and args.agent != experiment["agent"]:
        parser.error("--agent conflicts with the selected --config; choose one matching agent")
    settings.update(experiment)
    api = settings.get("api")
    if not isinstance(api, dict) or not all(key in api for key in ("base_url", "model")):
        parser.error("configs/default.yaml must define api.base_url and api.model")
    api.setdefault("api_key_env", None)
    for key in ("agent", "max_steps"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.model is not None:
        api["model"] = args.model
    if args.ollama_gpu and (urlparse(api["base_url"]).hostname not in
                           {"127.0.0.1", "localhost", "::1"} or api["api_key_env"]):
        parser.error("--gpu is only for local Ollama; omit it for a remote API")
    if args.episodes is not None:
        settings["num_episodes"] = args.episodes
    source = (ROOT / settings["balrog_path"]).resolve()
    if not (source / "balrog/config/config.yaml").is_file():
        parser.error(f"BALROG checkout not found: {source}")
    values = {
        "envs.names": "crafter",
        "agent.type": settings["agent"],
        "agent.max_text_history": settings["history_length"],
        "agent.max_image_history": 0,
        "++agent.replan_interval": settings["replan_interval"],
        "eval.num_workers": settings["num_workers"],
        "eval.num_episodes.crafter": settings["num_episodes"],
        "eval.max_steps_per_episode": settings["max_steps"],
        "client.client_name": "openai_compatible",
        "client.model_id": api["model"],
        "client.base_url": api["base_url"],
        "++client.api_key_env": api["api_key_env"],
        "client.generate_kwargs.temperature": settings["temperature"],
        "client.generate_kwargs.max_tokens": settings["num_predict"],
        "eval.output_dir": str((ROOT / settings["output_dir"]).resolve()),
    }
    if settings["agent"] == "react":
        values["++agent.max_thoughts"] = settings.get("max_thoughts", 3)
    if args.seed is not None:
        values.update({"envs.env_kwargs.seed": args.seed, "envs.crafter_kwargs.seed": args.seed})
    with initialize_config_dir(config_dir=str(source / "balrog/config"), version_base="1.1"):
        config = compose(config_name="config", overrides=[
            *(f"{key}={json.dumps(value)}" for key, value in values.items()), *overrides,
        ])
    for name, value in {
        "episodes": config.eval.num_episodes.crafter,
        "workers": config.eval.num_workers,
        "max_steps": config.eval.max_steps_per_episode,
        "replan_interval": config.agent.replan_interval,
        "history_length": config.agent.max_text_history,
    }.items():
        if type(value) is not int or value <= 0:
            parser.error(f"{name} must be a positive integer")
    if config.agent.type == "react" and (type(config.agent.max_thoughts) is not int or config.agent.max_thoughts < 0):
        parser.error("max_thoughts must be a nonnegative integer")
    for seed in (config.envs.env_kwargs.seed, config.envs.crafter_kwargs.seed):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
            parser.error("seed must be an integer in [0, 2**31)")
    if config.agent.type != settings["agent"]:
        parser.error("Select the algorithm with --agent or --config, not agent.type overrides")
    if config.envs.names != "crafter":
        parser.error("This project evaluates Crafter only")
    if config.eval.resume_from is not None:
        parser.error("Resume is not supported by this launcher; use a new run")
    endpoint = urlparse(config.client.base_url or "")
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        parser.error("api.base_url must be an HTTP(S) URL without credentials, query or fragment")
    if not isinstance(config.client.model_id, str) or not config.client.model_id.strip():
        parser.error("api.model must be a nonempty string")
    if args.ollama_gpu and (config.client.client_name not in {"openai_compatible", "vllm"} or config.client.api_key_env):
        parser.error("--gpu requires the local Ollama compatibility API without a key")
    config.eval.output_dir = str((ROOT / config.eval.output_dir).resolve())
    return args, source, config

