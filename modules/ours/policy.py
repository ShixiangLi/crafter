from modules.planner import PlannerAgent


class OurAgent(PlannerAgent):
    # Algorithm extension point. Currently identical to the planner baseline.
    # BALROG calls reset() per episode and act(observation, prev_action) per step.
    pass
