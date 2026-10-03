from balrog.agents import AgentFactory as BALROGAgentFactory
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder

from . import AGENTS


class AgentFactory(BALROGAgentFactory):
    """Register project algorithms without changing upstream BALROG."""

    def create_agent(self):
        agent_class = AGENTS.get(self.config.agent.type)
        if agent_class is None:
            return super().create_agent()
        return agent_class(
            create_llm_client(self.config.client),
            create_prompt_builder(self.config.agent),
            self.config,
        )
