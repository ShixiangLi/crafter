"""Run all agents through the native BALROG evaluator and result summaries."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.common.config import load_config


def main(argv=None):
    args, source, config = load_config(argv)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return 0
    sys.path.insert(0, str(source))
    from agents.factory import AgentFactory
    from balrog.evaluator import EvaluatorManager
    from balrog.utils import collect_and_summarize_results, print_summary_table, setup_environment

    if config.client.client_name in {"openai_compatible", "vllm"}:
        from modules.common.api_client import api_key
        api_key(config.client)  # Fail before creating a run if a required key is missing.
    else:
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
