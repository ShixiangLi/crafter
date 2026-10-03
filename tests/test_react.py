import copy
import json
import unittest

from test_balrog_integration import FakeClient, agent_with_client, config_for, observation
from balrog.environments import make_env


class ReActTests(unittest.TestCase):
    def test_retains_thought_action_and_full_observation(self):
        client = FakeClient([
            '{"rationale":"Need a pickaxe before mining.","action":"Do"}',
            '{"rationale":"No stone gained; gather wood first.","action":"Move East"}',
            '{"action":"Do"}',
        ])
        agent = agent_with_client("react", client)
        first = observation("Stone ahead")
        first["text"]["short_term_context"] = "Health: 8; wood: 0; stone: 0"
        original = copy.deepcopy(first)
        agent.act(first)
        second = observation("Stone unchanged; tree east")
        second["text"]["short_term_context"] = "Health: 7; wood: 0; stone: 0"
        agent.act(second, "Do")
        agent.act(observation("Tree ahead"), "Move East")
        messages = client.calls[-1]
        decisions = [json.loads(m.content) for m in messages if m.role == "assistant"]
        self.assertEqual([d["action"] for d in decisions], ["Do", "Move East"])
        self.assertEqual(decisions[0]["rationale"], "Need a pickaxe before mining.")
        self.assertEqual(decisions[1]["rationale"], "No stone gained; gather wood first.")
        self.assertIn("Health: 8; wood: 0; stone: 0", messages[0].content)
        self.assertIn("Health: 7; wood: 0; stone: 0", messages[2].content)
        self.assertNotIn("Example 1", messages[0].content)
        self.assertEqual(first, original)

    def test_optional_thought_and_one_call_per_action(self):
        client = FakeClient(['{"action":"Move East"}', '{"rationale":"Tree ahead.","action":"Do"}'])
        agent = agent_with_client("react", client)
        for prev in (None, "Move East"):
            result = agent.act(observation(), prev)
            self.assertEqual((result.input_tokens, result.output_tokens), (7, 3))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(agent.step, 2)

    def test_window_drops_whole_pairs_and_reset_clears_state(self):
        client = FakeClient()
        agent = agent_with_client("react", client, "agent.max_text_history=2")
        agent.act(observation("OLD_SCENE"))
        agent.act(observation("RECENT_SCENE"), "Noop")
        agent.act(observation("CURRENT_SCENE"), "Noop")
        messages = client.calls[-1]
        self.assertEqual([m.role for m in messages], ["user", "assistant", "user"])
        self.assertNotIn("OLD_SCENE", "".join(m.content for m in messages))
        self.assertIn("RECENT_SCENE", messages[0].content)
        self.assertIn("CURRENT_SCENE", messages[-1].content)
        agent.reset()
        self.assertEqual(agent.step, 0)
        self.assertFalse(agent.history)
        agent.act(observation("NEW_EPISODE"))
        self.assertEqual(len(client.calls[-1]), 1)
        self.assertNotIn("RECENT_SCENE", client.calls[-1][0].content)

    def test_single_observation_window_and_rules(self):
        client = FakeClient()
        agent = agent_with_client("react", client, "agent.max_text_history=1")
        agent.prompt_builder.update_instruction_prompt("GAME_RULES")
        agent.act(observation("First"))
        agent.act(observation("Second"), "Noop")
        self.assertEqual([m.role for m in client.calls[-1]], ["user", "user"])
        self.assertEqual(client.calls[-1][0].content, "GAME_RULES")
        self.assertFalse(agent.history)

    def test_parse_failure_preserves_evidence_and_actual_feedback(self):
        client = FakeClient(['broken output', '{"action":"Do"}'])
        agent = agent_with_client("react", client)
        result = agent.act(observation())
        self.assertEqual(result.completion, "")
        agent.act(observation("Invalid output. Defaulted to action: Noop. Tree ahead."), "")
        history = json.loads(client.calls[-1][1].content)
        self.assertEqual(history["action"], "")
        self.assertEqual(history["raw_output"], "broken output")
        self.assertIn("parse_error", history)
        self.assertIn("Defaulted to action: Noop", client.calls[-1][-1].content)

    def test_prompt_examples_against_real_crafter(self):
        # Construct deterministic demonstration scenes only in the test fixture.
        # The live agent only receives BALROG observations.
        for scenario in ("craft", "recover"):
            with self.subTest(scenario=scenario):
                env = make_env("crafter", "default", config_for("react"))
                try:
                    env.reset(seed=0)
                    raw = env.env.gym_env.env
                    player, world = raw._player, raw._world
                    for obj in list(world.objects):
                        if obj is not player:
                            world.remove(obj)
                    x, y = map(int, player.pos)
                    for dx in range(-4, 5):
                        for dy in range(-4, 5):
                            world[x + dx, y + dy] = "grass"
                    player.facing = (0, -1)
                    player.inventory.update(wood=0, stone=0, wood_pickaxe=0)
                    if scenario == "craft":
                        player.inventory["wood"] = 1
                        world[x - 1, y] = "table"
                        world[x, y - 1] = "tree"
                        _, _, _, _, info = env.step("Make Wood Pickaxe")
                        self.assertEqual(info["inventory"]["wood"], 0)
                        self.assertEqual(info["inventory"]["wood_pickaxe"], 1)
                        _, _, _, _, info = env.step("Do")
                        self.assertEqual(info["inventory"]["wood"], 1)
                        self.assertEqual(world[x, y - 1][0], "grass")
                    else:
                        world[x, y - 1] = "stone"
                        world[x + 1, y] = "tree"
                        _, _, _, _, info = env.step("Do")
                        self.assertEqual(info["inventory"]["stone"], 0)
                        self.assertEqual(world[x, y - 1][0], "stone")
                        env.step("Move East")
                        self.assertEqual(tuple(player.pos), (x, y))
                        self.assertEqual(tuple(player.facing), (1, 0))
                        _, _, _, _, info = env.step("Do")
                        self.assertEqual(info["inventory"]["wood"], 1)
                        self.assertEqual(world[x + 1, y][0], "grass")
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
