"""One evaluation path: BALROG manager, environments, clients and metrics."""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Crafter agents with BALROG")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/balrog_baseline.yaml")
    parser.add_argument("--agent", choices=["naive", "react", "planner", "ours"])
    parser.add_argument("--model")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, help="Fix both Crafter map and evaluator seed (same map every episode)")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without running episodes")
    args, overrides = parser.parse_known_args(argv)
    if any(item.startswith("--") for item in overrides):
        parser.error("Unknown option; use --help or BALROG Hydra overrides such as agent.max_text_history=8")
    experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent / experiment.pop("base")
    settings = yaml.safe_load(base.read_text(encoding="utf-8"))
    settings.update(experiment)
    for key in ("agent", "model", "max_steps"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
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
        "client.client_name": "vllm",
        "client.model_id": settings["model"],
        "client.base_url": settings["ollama_url"].rstrip("/") + "/v1",
        "client.generate_kwargs.temperature": settings["temperature"],
        "client.generate_kwargs.max_tokens": settings["num_predict"],
        "eval.output_dir": str((ROOT / settings["output_dir"]).resolve()),
    }
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
    for seed in (config.envs.env_kwargs.seed, config.envs.crafter_kwargs.seed):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
            parser.error("seed must be an integer in [0, 2**31)")
    if config.envs.names != "crafter":
        parser.error("This project evaluates Crafter only")
    if config.eval.resume_from is not None:
        parser.error("Resume is not supported by this launcher; use a new run")
    config.eval.output_dir = str((ROOT / config.eval.output_dir).resolve())
    return args, source, config


def main(argv=None):
    args, source, config = load_config(argv)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return 0
    sys.path.insert(0, str(source))
    from agents.factory import AgentFactory
    from balrog.evaluator import EvaluatorManager
    from balrog.utils import collect_and_summarize_results, print_summary_table, setup_environment

    setup_environment(original_cwd=str(source))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id += f"_{config.agent.type}_{config.client.model_id.replace('/', '_')}"
    output = Path(config.eval.output_dir) / run_id
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(output / "eval.log")],
        force=True,
    )
    print(f"Results: {output}", flush=True)
    manager = EvaluatorManager(config, original_cwd=str(source), output_dir=str(output))
    expected = len(manager.tasks)
    error = None
    try:
        manager.run(AgentFactory(config))
    except Exception as exc:
        logging.exception("Evaluation failed")
        error = f"{type(exc).__name__}: {exc}"
    summary = collect_and_summarize_results(str(output))
    print_summary_table(summary)
    completed = sum(item["episodes_played"] for item in summary["environments"].values())
    status = {"expected_episodes": expected, "completed_episodes": completed,
              "missing_episodes": expected - completed, "error": error}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if error or completed != expected:
        print(f"Incomplete evaluation: {status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
