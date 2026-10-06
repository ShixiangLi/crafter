"""Action selection from the planner baseline's current plan."""
import json


class Executor:
    def select_action(self, agent, plan):
        return agent.ask("planner/executor", f"\nCurrent plan: {json.dumps(plan)}")
