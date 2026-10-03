import contextlib
import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from test_balrog_integration import (
    ROOT, FakeClient, agent_with_client, config_for, main, observation,
)
from agents.factory import AgentFactory
from agents.spring_agent import ASSETS
from balrog.environments import make_env
from balrog.evaluator import EvaluatorManager
from modules.spring import SpringGraph, match_action


class SpringTests(unittest.TestCase):
    def setUp(self):
        # Earlier launcher tests configure handlers under temporary directories.
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)

    def tearDown(self):
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)

    def test_pinned_assets_and_official_graph(self):
        sources = json.loads((ASSETS / "sources.json").read_text())
        for name, digest in sources["files"].items():
            self.assertEqual(hashlib.sha256((ASSETS / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(sources["spring"]["commit"], "c0369b9127ab9ec63797797a3952be9e119334e1")
        graph = SpringGraph.from_file(ASSETS / "questions.json")
        self.assertEqual(len(graph.order), 9)
        self.assertEqual(sum(not parents for parents in graph.dependencies.values()), 2)
        self.assertEqual(graph.order[-1], "Choose the best executable action from above.")
        self.assertEqual(len(graph.dependencies[graph.action_question]), 4)

    def test_direct_parent_context_only_and_full_cost(self):
        client = FakeClient([f"answer-{i}" for i in range(8)] + ["I choose Move West."])
        agent = agent_with_client("spring", client)
        result = agent.act(observation())
        self.assertEqual(result.completion, "Move West")
        self.assertEqual((result.input_tokens, result.output_tokens), (63, 27))
        self.assertEqual(len(client.calls), 9)
        answers = {}
        for question, messages in zip(agent.graph.order, client.calls):
            parents = agent.graph.dependencies[question]
            self.assertEqual([message.role for message in messages[:3]], ["system"] * 3)
            self.assertEqual(messages[1].content, agent.manual)
            self.assertIn("Visible tree", messages[2].content)
            self.assertIn("Health: 9", messages[2].content)
            self.assertEqual(messages[-1].content, question)
            self.assertEqual(len(messages), 4 + 2 * len(parents))
            self.assertEqual([message.content for message in messages[3:-1:2]], parents)
            self.assertEqual([message.content for message in messages[4:-1:2]], [answers[q] for q in parents])
            answers[question] = f"answer-{len(answers)}"
        trace = json.loads(result.reasoning)
        self.assertEqual(trace["llm_calls"], 9)
        self.assertEqual(sum(node["input_tokens"] for node in trace["nodes"]), result.input_tokens)
        self.assertEqual(trace["raw_action"], "I choose Move West.")

    def test_two_observations_and_reset(self):
        client = FakeClient()
        agent = agent_with_client("spring", client)
        agent.act(observation("First scene"))
        agent.act(observation("Second scene"), "Move West")
        agent.act(observation("Third scene"), "Do")
        self.assertEqual(len(client.calls), 27)
        context = client.calls[-1][2].content
        self.assertNotIn("First scene", context)
        self.assertIn("Second scene", context)
        self.assertIn("Third scene", context)
        self.assertIn("Previous submitted action: Do", context)
        agent.reset()
        self.assertEqual(agent.step, 0)
        self.assertFalse(agent.observations)
        self.assertFalse(agent.last_trace)
        agent.act(observation("New episode"))
        self.assertNotIn("Third scene", client.calls[-1][2].content)
        self.assertIn("No previous agent action", client.calls[-1][2].content)

    def test_invalid_action_uses_balrog_fallback(self):
        client = FakeClient(["context"] * 8 + ["Unknown command"])
        agent = agent_with_client("spring", client)
        result = agent.act(observation())
        self.assertEqual(result.completion, "Unknown command")
        self.assertIsNotNone(json.loads(result.reasoning)["parse_error"])
        env = make_env("crafter", "default", config_for("spring"))
        try:
            self.assertEqual(env.check_action_validity(result.completion), "Noop")
            self.assertEqual(env.failed_candidates, ["Unknown command"])
        finally:
            env.close()
        self.assertEqual(match_action("move east", agent.actions), "Move East")
        self.assertEqual(match_action("Do, or Move West", agent.actions), "Move West")

    def test_invalid_graphs_fail_before_querying(self):
        graphs = [({}, "a"), ({"a": ["b"]}, "a"),
                  ({"a": ["b"], "b": ["a"]}, "a"),
                  ({"a": [], "b": []}, "a"),
                  ({"a": ["b", "b"], "b": []}, "a")]
        for dependencies, action in graphs:
            with self.subTest(dependencies=dependencies), self.assertRaises(ValueError):
                SpringGraph(dependencies, action)

    def test_partial_failure_is_not_converted_to_success(self):
        client = FakeClient(["First answer"])
        agent = agent_with_client("spring", client)
        with self.assertRaises(StopIteration):
            agent.act(observation())
        self.assertEqual(len(agent.last_trace), 1)
        self.assertEqual(agent.step, 0)

    def test_spring_parallel_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = config_for("spring", "eval.num_workers=2", "eval.num_episodes.crafter=2")
            with patch("agents.factory.create_llm_client", return_value=lambda: FakeClient()), \
                 contextlib.redirect_stderr(io.StringIO()):
                results = EvaluatorManager(config, original_cwd=str(ROOT / "BALROG"), output_dir=tmp).run(AgentFactory(config))
            self.assertEqual(len(results["crafter"]), 2)
            self.assertTrue(all(item["input_tokens"] == 189 for item in results["crafter"]))

    def test_native_client_http_and_real_environment(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, request))
                final = request["messages"][-1]["content"][0]["text"]
                answer = "Do" if final == "Choose the best executable action from above." else "Observed context."
                data = json.dumps({
                    "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                    "model": "mock-model", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": answer}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = main(["--agent", "spring", "--episodes", "1", "--max-steps", "1",
                                 "--seed", "0", "--model", "mock-model", f"eval.output_dir={tmp}",
                                 "client.max_retries=1", "client.delay=0",
                                 f"client.base_url=http://127.0.0.1:{server.server_port}/v1"])
                self.assertEqual(code, 0)
                self.assertEqual(len(requests), 9)
                self.assertTrue(all(path == "/v1/chat/completions" for path, _ in requests))
                run = next(Path(tmp).iterdir())
                episode = json.loads((run / "crafter/default/default_run_00.json").read_text())
                self.assertEqual((episode["input_tokens"], episode["output_tokens"]), (63, 27))
                self.assertEqual(episode["action_frequency"], {"Do": 1})
                self.assertEqual(episode["failed_candidates"], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            logging.shutdown()


if __name__ == "__main__":
    unittest.main()
