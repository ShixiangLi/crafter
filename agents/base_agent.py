import copy
import json
from pathlib import Path

from balrog.agents.base import BaseAgent as BALROGAgent

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class BaseAgent(BALROGAgent):
    """Use BALROG history and client; adapt JSON decisions to LLMResponse."""

    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.config = config
        self.reset()

    def reset(self):
        super().reset()
        self.prompt_builder.previous_reasoning = None
        self.step = 0
        self.responses = []

    def begin_step(self, observation, prev_action):
        self.responses = []
        if prev_action is not None:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(observation)

    def ask(self, prompt_name, extra=""):
        messages = copy.deepcopy(self.prompt_builder.get_prompt())
        messages[-1].content += "\n\n" + (PROMPTS / f"{prompt_name}.txt").read_text(encoding="utf-8") + extra
        response = self.client.generate(messages)
        self.responses.append(response)
        try:
            answer = json.loads(response.completion)
            if not isinstance(answer, dict):
                raise ValueError("Expected a JSON object")
            return answer
        except (ValueError, TypeError) as error:
            return {"parse_error": str(error), "raw_output": response.completion}

    def finish_step(self, decision, **details):
        action = decision.get("action")
        # Leave invalid output invalid so BALROG records it and applies its fallback.
        completion = action if isinstance(action, str) else ""
        reasoning = json.dumps({**decision, **details}, ensure_ascii=False)
        self.step += 1
        return self.responses[-1]._replace(
            completion=completion,
            reasoning=reasoning,
            input_tokens=sum(response.input_tokens for response in self.responses),
            output_tokens=sum(response.output_tokens for response in self.responses),
        )
