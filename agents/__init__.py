"""BALROG entries and registration; algorithm implementations live in modules."""
from .our_agent import OurAgent
from .planner_agent import PlannerAgent
from .react_agent import ReActAgent
from .spring_agent import SPRINGAgent

AGENTS = {"react": ReActAgent, "planner": PlannerAgent, "ours": OurAgent, "spring": SPRINGAgent}
