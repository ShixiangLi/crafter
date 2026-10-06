"""Read BALROG results and compute reusable Crafter statistics."""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling",
    "collect_stone", "collect_wood", "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone",
    "place_table", "wake_up",
)
VITALS = ("health", "food", "drink", "energy")
RESOURCES = ("wood", "stone", "coal", "iron", "diamond")


def parse_inventory(text):
    """Parse BALROG's visible status/inventory text, including the reset observation."""
    return {name: int(value) for name, value in re.findall(r"^- ([a-z_]+): (\d+)(?:/\d+)?\s*$", text, re.M)}


def achievement_rates(episodes):
    """Percent of completed episodes unlocking each of the 22 achievements."""
    return {name: 100 * sum(episode.get("achievements", {}).get(name, 0) > 0 for episode in episodes)
            / len(episodes) if episodes else None for name in ACHIEVEMENTS}


def crafter_score(rates):
    """Official offset geometric mean on percentages, not fractions.

    Reference: https://github.com/danijar/crafter/blob/main/analysis/common.py
    """
    values = [rates[name] for name in ACHIEVEMENTS]
    if any(value is None for value in values):
        return None
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise ValueError("Achievement rates must be percentages in [0, 100]")
    return math.expm1(sum(math.log1p(value) for value in values) / len(values))


def load_episode(run_dir, relative):
    run_dir, relative = Path(run_dir), Path(relative)
    result_path = run_dir / relative.with_suffix(".json")
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    csv_path = run_dir / relative.with_suffix(".csv")
    rows = []
    if csv_path.exists():
        # SPRING reasoning can exceed the CSV reader's default 128 KiB field limit.
        csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = [{key: row.get(key, "") for key in ("Step", "Action", "Reward", "Done")}
                    for row in csv.DictReader(stream, escapechar="˘")]
    trace_path = run_dir / "traces" / relative.with_suffix(".jsonl")
    events = []
    if trace_path.exists():
        lines = trace_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:  # An interrupted write may leave an unfinished final line.
                    raise
    states = [event for event in events if event["event"] == "state"]
    calls = [event for event in events if event["event"] == "model_call"]
    end = next((event for event in reversed(events) if event["event"] == "episode_end"), {})
    completed = result is not None
    stop = "error" if end.get("stop") == "error" else "incomplete"
    if completed:
        last = states[-1] if states else {}
        stop = ("terminated" if last.get("terminated") else "truncated" if last.get("truncated")
                else "step_limit" if states else "done" if result.get("done") else "step_limit")
    achievements = (result or {}).get("achievements") or (states[-1].get("achievements", {}) if states else {})
    return {"id": relative.with_suffix("").as_posix(), "result": result, "rows": rows,
            "states": states, "calls": calls, "completed": completed, "stop": stop,
            "elapsed_seconds": end.get("elapsed_seconds"), "achievements": achievements,
            "steps": result["num_steps"] if completed else max(len(rows), states[-1]["step"] if states else 0),
            "seed": result.get("seed") if completed else states[0].get("seed") if states else None,
            "requested_actions": dict(Counter(state["requested_action"] for state in states if "requested_action" in state))
                                 if any("requested_action" in state for state in states)
                                 else dict(Counter(row["Action"] for row in rows)),
            "executed_actions": dict(Counter(state["executed_action"] for state in states if "executed_action" in state))}


def load_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    paths = {path.relative_to(run_dir).with_suffix("") for suffix in ("csv", "json")
             for path in run_dir.glob(f"crafter/**/*_run_*.{suffix}")}
    paths.update(path.relative_to(run_dir / "traces").with_suffix("")
                 for path in (run_dir / "traces").glob("crafter/**/*_run_*.jsonl"))
    return [load_episode(run_dir, path) for path in sorted(paths)]


def summarize(episodes, expected=None):
    completed = [episode["result"] for episode in episodes if episode["completed"]]
    rates = achievement_rates(completed)
    return {
        "expected_episodes": expected, "completed_episodes": len(completed),
        "partial_episodes": sum(not episode["completed"] for episode in episodes),
        "missing_episodes": max(0, expected - len(completed)) if expected is not None else None,
        "distinct_seeds": len({result["seed"] for result in completed}),
        "crafter_score": crafter_score(rates), "achievement_success_rates": rates,
        "diamond_success_rate": rates["collect_diamond"],
        "average_unlocked_achievements": sum(sum(count > 0 for count in result["achievements"].values())
                                              for result in completed) / len(completed) if completed else None,
        "average_steps": sum(result["num_steps"] for result in completed) / len(completed) if completed else None,
        "completed_input_tokens": sum(result["input_tokens"] for result in completed),
        "completed_output_tokens": sum(result["output_tokens"] for result in completed),
        "recorded_model_calls": sum(len(episode["calls"]) for episode in episodes),
        "failed_model_calls": sum(call["status"] == "failed" for episode in episodes for call in episode["calls"]),
    }
