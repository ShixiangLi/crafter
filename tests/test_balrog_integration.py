import contextlib
import csv
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BALROG"))

from scripts.evaluate import load_config, main
from agents.factory import AgentFactory
from balrog.agents.naive import NaiveAgent
from balrog.client import LLMResponse
from balrog.environments import make_env


class FakeClient:
    def __init__(self, replies=None):
        self.replies = iter(replies) if replies is not None else None
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        if self.replies is not None:
            completion = next(self.replies)
        elif '"plan" array' in messages[-1].content:
            completion = '{"plan": ["Collect wood"]}'
        elif '"action"' in messages[-1].content:
            completion = '{"action": "Noop", "rationale": "Observe"}'
        else:
            completion = 'Noop'
        return LLMResponse("fake", completion, "stop", 7, 3, None)


def config_for(agent="planner", *overrides):
    return load_config(["--agent", agent, "--episodes", "1", "--max-steps", "3",
                        "--seed", "0", *overrides])[2]


def agent_with_client(agent, client, *overrides):
    with patch("agents.factory.create_llm_client", return_value=lambda: client), \
         patch("balrog.agents.create_llm_client", return_value=lambda: client):
        return AgentFactory(config_for(agent, *overrides)).create_agent()


def observation(text="Visible tree"):
    return {"text": {"long_term_context": text, "short_term_context": "Health: 9"}, "image": None}


class AgentContractTests(unittest.TestCase):
    def test_native_naive_is_upstream_class(self):
        agent = agent_with_client("naive", FakeClient())
        self.assertIs(type(agent), NaiveAgent)
        self.assertEqual(agent.act(observation()).completion, "Noop")

    def test_planning_cost_interval_and_reset(self):
        client = FakeClient()
        agent = agent_with_client("planner", client, "agent.replan_interval=2")
        agent.prompt_builder.update_instruction_prompt("Game rules")
        first = agent.act(observation())
        self.assertEqual((first.input_tokens, first.output_tokens), (14, 6))
        self.assertEqual(json.loads(first.reasoning)["plan"], ["Collect wood"])
        second = agent.act(observation("Visible water"), "Noop")
        self.assertEqual(second.input_tokens, 7)
        third = agent.act(observation(), "Noop")
        self.assertEqual(third.input_tokens, 14)
        agent.reset()
        self.assertEqual(agent.step, 0)
        self.assertEqual(agent.planner.plan, [])
        self.assertFalse(agent.prompt_builder._events)
        self.assertEqual(agent.act(observation()).input_tokens, 14)

    def test_invalid_output_reaches_balrog_validation(self):
        client = FakeClient(['not json'])
        result = agent_with_client("react", client).act(observation())
        self.assertEqual(result.completion, "")
        self.assertIn("parse_error", json.loads(result.reasoning))
        env = make_env("crafter", "default", config_for("react"))
        try:
            self.assertEqual(env.check_action_validity(result.completion), "Noop")
            self.assertEqual(env.failed_candidates, [""])
        finally:
            env.close()

    def test_failed_replan_keeps_previous_plan(self):
        client = FakeClient(['{"plan":["Collect wood"]}', '{"action":"Noop"}',
                             '{"plan":[]}', '{"action":"Noop"}'])
        agent = agent_with_client("planner", client, "agent.replan_interval=1")
        agent.act(observation())
        result = agent.act(observation(), "Noop")
        self.assertEqual(json.loads(result.reasoning)["plan"], ["Collect wood"])
        self.assertIsNotNone(json.loads(result.reasoning)["plan_error"])
        self.assertEqual(result.input_tokens, 14)

    def test_history_uses_native_observation_without_prompt_pollution(self):
        agent = agent_with_client("planner", FakeClient())
        agent.act(observation("First observation"))
        agent.act(observation("Second observation"), "Noop")
        events = agent.prompt_builder._events
        self.assertEqual([event["type"] for event in events], ["observation", "action", "observation"])
        self.assertEqual(events[0]["text"], "First observation")
        self.assertEqual(events[-1]["text"], "Second observation")


class LauncherTests(unittest.TestCase):
    def test_config_precedence(self):
        cfg = config_for("ours", "agent.type=react", "eval.num_episodes.crafter=2")
        self.assertEqual(cfg.agent.type, "react")
        self.assertEqual(cfg.eval.num_episodes.crafter, 2)
        self.assertEqual(cfg.envs.env_kwargs.seed, 0)
        self.assertEqual(cfg.envs.crafter_kwargs.seed, 0)

    def test_removed_options_and_invalid_budget_fail(self):
        for args in (["--seeds", "0,1"], ["--max-steps", "0"], ["--seed", "-1"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    load_config(args)
                self.assertEqual(error.exception.code, 2)

    def test_all_experiment_configs(self):
        for path in (ROOT / "experiments").glob("*.yaml"):
            with self.subTest(path=path):
                cfg = load_config(["--config", str(path)])[2]
                self.assertIn(cfg.agent.type, ["naive", "react", "planner", "ours"])
                self.assertEqual(cfg.client.generate_kwargs.max_tokens, 8192)

    def test_real_environment_evaluation_for_all_agents(self):
        for agent in ("naive", "react", "planner", "ours"):
            with self.subTest(agent=agent), tempfile.TemporaryDirectory() as tmp:
                client = FakeClient()
                with patch("agents.factory.create_llm_client", return_value=lambda: client), \
                     patch("balrog.agents.create_llm_client", return_value=lambda: client), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = main(["--agent", agent, "--episodes", "1", "--max-steps", "3",
                                 "--seed", "0", f"eval.output_dir={tmp}"])
                self.assertEqual(code, 0)
                run = next(Path(tmp).iterdir())
                episode = json.loads((run / "crafter/default/default_run_00.json").read_text())
                self.assertEqual(episode["num_steps"], 3)
                self.assertEqual(episode["failed_candidates"], [])
                self.assertEqual(episode["input_tokens"], 28 if agent in ("planner", "ours") else 21)
                self.assertEqual(episode["agent"]["type"], agent)
                self.assertEqual(episode["seed"], 0)
                status = json.loads((run / "status.json").read_text())
                self.assertEqual(status["missing_episodes"], 0)
                with (run / "crafter/default/default_run_00.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3)
                if agent in ("planner", "ours"):
                    self.assertEqual(json.loads(rows[0]["Reasoning"])["plan"], ["Collect wood"])
                logging.shutdown()

    def test_failed_run_is_nonzero_and_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient([])
            with patch("agents.factory.create_llm_client", return_value=lambda: client), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--agent", "react", "--episodes", "1", "--max-steps", "1",
                             "--seed", "0", f"eval.output_dir={tmp}"])
            self.assertEqual(code, 1)
            run = next(Path(tmp).iterdir())
            status = json.loads((run / "status.json").read_text())
            self.assertEqual(status["missing_episodes"], 1)
            self.assertIn("StopIteration", status["error"])
            logging.shutdown()

    def test_parallel_workers_use_external_factory(self):
        from balrog.evaluator import EvaluatorManager
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_for("ours", "eval.num_workers=2", "eval.num_episodes.crafter=2")
            with patch("agents.factory.create_llm_client", return_value=lambda: FakeClient()), \
                 contextlib.redirect_stderr(io.StringIO()):
                results = EvaluatorManager(cfg, original_cwd=str(ROOT / "BALROG"), output_dir=tmp).run(AgentFactory(cfg))
            self.assertEqual(len(results["crafter"]), 2)
            self.assertTrue(all(result["input_tokens"] == 28 for result in results["crafter"]))


if __name__ == "__main__":
    unittest.main()
