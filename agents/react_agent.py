import json
from collections import deque

from balrog.prompt_builder.history import Message

from .base_agent import BaseAgent


class ReActAgent(BaseAgent):
    """Sparse ReAct: thinking updates context; only an action advances Crafter."""

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
        instructions = (self.prompt_builder.system_prompt or "") + "\n\n" + self.load_prompt("react/react")
        messages = [Message("system", instructions)]
        for previous_observation, decision in self.history:
            messages.extend([
                Message("user", "Observation:\n" + previous_observation),
                Message("assistant", decision),
            ])
        messages.append(Message("user", "Current Observation:\n" + current))
        thoughts = []
        limit = self.config.agent.max_thoughts
        for attempt in range(limit + 1):
            if attempt == limit:
                messages[-1].content += "\nThinking limit reached: return an action now."
            decision = self.ask(messages=messages)
            thought = decision.get("rationale")
            if (decision.get("action") not in (None, "") or decision.get("parse_error")
                    or not isinstance(thought, str) or not thought.strip()):
                break
            if attempt == limit:
                decision["parse_error"] = "ReAct thinking limit reached without an action"
                break
            thoughts.append(thought)
            messages.extend([
                Message("assistant", json.dumps(decision, ensure_ascii=False)),
                Message("user", "Thought recorded. The game has not advanced; continue reasoning or choose an action."),
            ])
        response = self.finish_step(decision, strategy="react", protocol="sparse-thought-v1",
                                    thoughts=thoughts, llm_calls=len(self.responses))
        # Keep every thought and the actual submitted candidate, including errors.
        self.history.append((current, json.dumps(
            {**decision, "thoughts": thoughts, "action": response.completion}, ensure_ascii=False,
        )))
        return response
