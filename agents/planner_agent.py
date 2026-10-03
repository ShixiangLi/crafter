from modules.executor import Executor
from modules.planner import Planner

from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder, config):
        self.planner = Planner(config.agent.replan_interval)
        self.executor = Executor()
        super().__init__(client_factory, prompt_builder, config)

    def reset(self):
        super().reset()
        self.planner.reset()

    def act(self, observation, prev_action=None):
        self.begin_step(observation, prev_action)
        plan_error = self.planner.update(self)
        decision = self.executor.select_action(self, self.planner.plan)
        return self.finish_step(decision, plan=list(self.planner.plan), plan_error=plan_error)
