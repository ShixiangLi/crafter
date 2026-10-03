"""SPRING question DAG, adapted from Holmeswww/SPRING (MIT, Yue Wu).

Original questions and attribution are in prompts/spring/.
"""
import json


class SpringGraph:
    def __init__(self, dependencies, action_question):
        self.dependencies = dependencies
        self.action_question = action_question
        if not isinstance(dependencies, dict) or not dependencies:
            raise ValueError("SPRING requires a nonempty question graph")
        if action_question not in dependencies:
            raise ValueError("SPRING action question is missing")
        for question, parents in dependencies.items():
            if not isinstance(question, str) or not question.strip():
                raise ValueError("SPRING questions must be nonempty strings")
            if not isinstance(parents, list) or any(
                not isinstance(parent, str) or parent not in dependencies for parent in parents
            ):
                raise ValueError(f"Unknown or invalid dependencies for {question}")
            if len(parents) != len(set(parents)):
                raise ValueError(f"Duplicate dependencies for {question}")
        self.order = []
        visiting, visited = set(), set()

        def visit(question):
            if question in visiting:
                raise ValueError("Cycle in SPRING question graph")
            if question in visited:
                return
            visiting.add(question)
            for parent in dependencies[question]:
                visit(parent)
            visiting.remove(question)
            visited.add(question)
            self.order.append(question)

        visit(action_question)
        if visited != set(dependencies):
            raise ValueError("Every SPRING question must contribute to the action node")

    @classmethod
    def from_file(cls, path):
        spec = json.loads(path.read_text(encoding="utf-8"))
        return cls(spec["dependencies"], spec["action_question"])

    def run(self, ask):
        """Evaluate each node once; supply only its direct parent Q/A pairs."""
        answers = {}
        for question in self.order:
            context = [(parent, answers[parent]) for parent in self.dependencies[question]]
            answers[question] = ask(question, context)
        return answers


def match_action(answer, actions):
    """Keep the official ordered, case-insensitive substring action matching.

    Unmatched text is left invalid for BALROG's action validator (Noop fallback).
    """
    return next((action for action in actions if action.lower() in answer.lower()), None)
