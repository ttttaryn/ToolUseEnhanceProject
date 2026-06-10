import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_grpo_tooluse.rewards import (
    argument_reward,
    format_reward,
    hallucination_penalty,
    tool_selection_reward,
)
from trl_grpo_tooluse.tool_parser import parse_tool_calls


WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]


class RewardTests(unittest.TestCase):
    def test_parse_openai_style_tool_call(self):
        text = json.dumps({"tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Beijing"}}}]})
        calls = parse_tool_calls(text)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments["city"], "Beijing")

    def test_correct_call_gets_positive_rewards(self):
        completion = json.dumps({"name": "get_weather", "arguments": {"city": "Beijing", "date": "today"}})
        gold = json.dumps({"name": "get_weather", "arguments": {"city": "Beijing", "date": "today"}})
        self.assertEqual(format_reward([completion], [gold]), [1.0])
        self.assertEqual(tool_selection_reward([completion], [gold]), [1.0])
        self.assertEqual(argument_reward([completion], [gold]), [1.0])

    def test_wrong_tool_is_penalized(self):
        completion = json.dumps({"name": "book_flight", "arguments": {"city": "Beijing"}})
        gold = json.dumps({"name": "get_weather", "arguments": {"city": "Beijing"}})
        self.assertEqual(tool_selection_reward([completion], [gold]), [-1.0])

    def test_missing_argument_lowers_argument_reward(self):
        completion = json.dumps({"name": "get_weather", "arguments": {"city": "Beijing"}})
        gold = json.dumps({"name": "get_weather", "arguments": {"city": "Beijing", "date": "today"}})
        self.assertLess(argument_reward([completion], [gold])[0], 1.0)

    def test_no_tool_sample_rewards_direct_answer(self):
        completion = "You can answer this without using any external tool."
        self.assertEqual(format_reward([completion], ["none"], task_type=["irrelevance"]), [1.0])
        self.assertEqual(tool_selection_reward([completion], ["none"], task_type=["irrelevance"]), [1.0])

    def test_no_tool_sample_penalizes_tool_call(self):
        completion = json.dumps({"name": "get_weather", "arguments": {"city": "Beijing"}})
        self.assertEqual(tool_selection_reward([completion], ["none"], task_type=["irrelevance"]), [-1.0])

    def test_hallucinated_tool_and_argument_are_penalized(self):
        completion = json.dumps({"name": "get_weather", "arguments": {"city": "Beijing", "unit": "kelvin"}})
        penalty = hallucination_penalty([completion], ["{}"], tools=[WEATHER_TOOL])[0]
        self.assertLess(penalty, 0.0)

        invented = json.dumps({"name": "invented_tool", "arguments": {}})
        self.assertEqual(hallucination_penalty([invented], ["{}"], tools=[WEATHER_TOOL]), [-1.0])


if __name__ == "__main__":
    unittest.main()
