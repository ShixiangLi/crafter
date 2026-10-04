import json


class Planner:
    def __init__(self, interval=10):
        self.interval = interval
        self.reset()

    def reset(self):
        self.plan = []

    def update(self, agent):
        if self.plan and agent.step % self.interval != 0:
            return None
        answer = agent.ask("planner/planner", f"\nPrevious plan: {json.dumps(self.plan)}")
        plan = answer.get("plan")
        if (
            isinstance(plan, list)
            and 1 <= len(plan) <= 5
            and all(isinstance(item, str) and item.strip() for item in plan)
        ):
            self.plan = plan
            return None
        return answer.get("parse_error", "Expected 1 to 5 nonempty plan strings")
