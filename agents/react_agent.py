from .base_agent import BaseAgent


class ReActAgent(BaseAgent):
    """A brief rationale followed by one action, using native observations."""

    def act(self, observation, prev_action=None):
        self.begin_step(observation, prev_action)
        return self.finish_step(self.ask("react"))
