import json


class Executor:
    def select_action(self, agent, plan):
        return agent.ask("planner/executor", f"\nCurrent plan: {json.dumps(plan)}")
