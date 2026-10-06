"""Observe native BALROG evaluation without changing the agent or environment."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import balrog.evaluator as native

from .statistics import parse_inventory


class EpisodeTrace:
    def __init__(self, stream):
        self.stream = stream
        self.step = 0
        self.calls = 0

    def write(self, event, **data):
        self.stream.write(json.dumps({"event": event, **data}, ensure_ascii=False) + "\n")
        self.stream.flush()

    def state(self, observation, info, **data):
        # Store reported state only; never send this diagnostic data to the agent.
        inventory = info.get("inventory") or parse_inventory(observation["text"]["short_term_context"])
        self.write("state", step=self.step,
                   inventory={key: int(value) for key, value in inventory.items()},
                   achievements={key: int(value) for key, value in info.get("achievements", {}).items()},
                   **data)


class RecordedClient:
    def __init__(self, client, trace):
        self.client, self.trace = client, trace

    def __getattr__(self, name):
        return getattr(self.client, name)

    def generate(self, messages):
        started = time.monotonic()
        self.trace.calls += 1
        data = {"call": self.trace.calls, "environment_step": self.trace.step + 1}
        try:
            response = self.client.generate(messages)
        except BaseException as error:
            self.trace.write("model_call", **data, elapsed_seconds=time.monotonic() - started,
                             status="failed", error_type=type(error).__name__,
                             input_tokens=None, output_tokens=None)
            raise
        self.trace.write("model_call", **data, elapsed_seconds=time.monotonic() - started,
                         status="returned", input_tokens=response.input_tokens,
                         output_tokens=response.output_tokens, stop_reason=response.stop_reason,
                         requested_model=response.model_id)
        return response


class RecordedEnvironment:
    def __init__(self, env, trace):
        self.env, self.trace = env, trace
        self.requested_action = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.trace.state(observation, info, seed=kwargs.get("seed"))
        return observation, info

    def check_action_validity(self, candidate):
        self.requested_action = candidate
        return self.env.check_action_validity(candidate)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.trace.step += 1
        self.trace.state(observation, info, requested_action=self.requested_action,
                         executed_action=action, reward=float(reward),
                         terminated=bool(terminated), truncated=bool(truncated))
        return observation, reward, terminated, truncated, info


class RecordedEvaluator(native.Evaluator):
    def run_episode(self, task, agent, process_num=None, position=0, episode_idx=0):
        path = Path(self.output_dir) / "traces" / self.env_name / task / f"{task}_run_{episode_idx:02d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        environments = []
        make_env = native.make_env

        with path.open("w", encoding="utf-8") as stream:
            trace = EpisodeTrace(stream)
            trace.write("episode_start", agent=self.config.agent.type,
                        requested_model=self.config.client.model_id)

            def make_recorded_env(*args, **kwargs):
                env = RecordedEnvironment(make_env(*args, **kwargs), trace)
                environments.append(env)
                return env

            # Native workers are separate processes; restore both hooks after each episode.
            with patch.object(native, "make_env", make_recorded_env), \
                 patch.object(agent, "client", RecordedClient(agent.client, trace)):
                try:
                    result = super().run_episode(task, agent, process_num, position, episode_idx)
                except BaseException as error:
                    trace.write("episode_end", completed=False, stop="error",
                                error_type=type(error).__name__, elapsed_seconds=time.monotonic() - started)
                    raise
                else:
                    trace.write("episode_end", completed=True, elapsed_seconds=time.monotonic() - started)
                    return result
                finally:
                    for env in environments:
                        env.close()


class EvaluatorManager(native.EvaluatorManager):
    """Keep BALROG scheduling and metrics; add an observer to each evaluator."""

    def __init__(self, config, original_cwd="", output_dir="."):
        super().__init__(config, original_cwd, output_dir)
        self.env_evaluators = {
            name: RecordedEvaluator(name, config, original_cwd, output_dir)
            for name in self.env_names
        }
