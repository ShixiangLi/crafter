"""SPRING strategy on BALROG observations; no separate environment or client.

Adapted from Holmeswww/SPRING, Copyright (c) 2023 Yue Wu (MIT).
See prompts/spring/README.md for pinned sources and protocol differences.
"""
import hashlib
import json
import logging
import time
from collections import deque

from balrog.agents.base import BaseAgent
from balrog.prompt_builder.history import Message

from modules.common import ROOT
from .graph import SpringGraph, match_action

ASSETS = ROOT / "prompts" / "spring"
logger = logging.getLogger(__name__)


class SPRINGAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        from balrog.environments.crafter import ACTIONS

        self.actions = tuple(ACTIONS)
        self.graph = SpringGraph.from_file(ASSETS / "questions.json")
        self.manual = (ASSETS / "manual.txt").read_text(encoding="utf-8")
        self.asset_hashes = {
            name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
            for name in ("questions.json", "manual.txt")
        }
        self.reset()

    def reset(self):
        super().reset()
        self.prompt_builder.previous_reasoning = None
        # SPRING retains two full observations, including the vitals of each.
        self.observations = deque(maxlen=2)
        self.step = 0
        self.last_trace = []

    def act(self, observation, prev_action=None):
        text = observation["text"]
        previous = "No previous agent action." if prev_action is None else f"Previous submitted action: {prev_action}"
        self.observations.append(
            f"Player Observation Step {self.step}:\n{previous}\n"
            f"{text.get('long_term_context', '')}\n\n{text.get('short_term_context', '')}"
        )
        text_obs = "\n\n".join(self.observations)
        self.last_trace = []
        responses = []

        def ask(question, context):
            messages = [
                Message("system", "You’re a player trying to play the game of crafter."),
                Message("system", self.manual),
                Message("system", "Most recent two steps of the player's in-game observation:\n" + text_obs),
            ]
            for parent, answer in context:
                messages.extend([Message("user", parent), Message("assistant", answer)])
            messages.append(Message("user", question))
            started = time.monotonic()
            response = self.client.generate(messages)
            responses.append(response)
            answer = response.completion.strip()
            node = {
                "question": question,
                "dependencies": list(self.graph.dependencies[question]),
                "answer": answer,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "stop_reason": response.stop_reason,
                "elapsed_seconds": time.monotonic() - started,
            }
            self.last_trace.append(node)
            logger.info("SPRING step=%s node=%s", self.step, json.dumps(node, ensure_ascii=False))
            return answer

        answers = self.graph.run(ask)
        raw_action = answers[self.graph.action_question]
        matched_action = match_action(raw_action, self.actions)
        action = matched_action or "Do"  # Official SPRING fallback (paper section 2.2).
        details = {
            "strategy": "spring",
            "step": self.step,
            "asset_sha256": self.asset_hashes,
            "nodes": self.last_trace,
            "llm_calls": len(responses),
            "raw_action": raw_action,
            "action": action,
            "parse_error": None if matched_action else "No allowed action found in final answer",
            "fallback_action": None if matched_action else "Do",
        }
        self.step += 1
        return responses[-1]._replace(
            completion=action,
            reasoning=json.dumps(details, ensure_ascii=False),
            input_tokens=sum(response.input_tokens for response in responses),
            output_tokens=sum(response.output_tokens for response in responses),
        )
