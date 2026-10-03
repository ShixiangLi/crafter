import json
from collections import deque

from balrog.prompt_builder.history import Message

from .base_agent import BaseAgent


class ReActAgent(BaseAgent):
    """ReAct with bounded observation/thought/action history and real feedback."""

    def reset(self):
        super().reset()
        # Count the current observation in the configured history window.
        self.history = deque(maxlen=self.config.agent.max_text_history - 1)

    def act(self, observation, prev_action=None):
        self.responses = []
        text = observation["text"]
        current = "\n\n".join(filter(None, (
            text.get("long_term_context", ""), text.get("short_term_context", ""),
        )))
        if prev_action is not None:
            current = f"Previous submitted action: {prev_action}\n{current}"
        messages = []
        if self.prompt_builder.system_prompt:
            messages.append(Message("user", self.prompt_builder.system_prompt))
        for previous_observation, decision in self.history:
            messages.extend([
                Message("user", "Observation:\n" + previous_observation),
                Message("assistant", decision),
            ])
        messages.append(Message("user", "Current Observation:\n" + current))
        decision = self.ask("react", messages=messages)
        response = self.finish_step(decision)
        # Keep the actual submitted candidate, including invalid-output evidence.
        self.history.append((current, json.dumps(
            {**decision, "action": response.completion}, ensure_ascii=False,
        )))
        return response
