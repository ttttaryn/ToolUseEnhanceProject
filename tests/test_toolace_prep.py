import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_grpo_tooluse.tool_parser import parse_tool_calls
from trl_grpo_tooluse.toolace_prep import convert_toolace_record, extract_json_tools


class ToolAcePrepTests(unittest.TestCase):
    def test_parse_toolace_function_call_with_spaces(self):
        calls = parse_tool_calls('[Market Trends API(trend_type="MARKET_INDEXES", country="us")]')
        self.assertEqual(calls[0].name, "Market Trends API")
        self.assertEqual(calls[0].arguments["trend_type"], "MARKET_INDEXES")

    def test_convert_toolace_record_with_inferred_schema(self):
        record = {
            "system": "You can use tools.",
            "conversations": [
                {"from": "human", "value": "Show market indexes in the US."},
                {"from": "gpt", "value": '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]'},
            ],
        }
        converted = convert_toolace_record(record)
        self.assertIsNotNone(converted)
        self.assertIn("Market Trends API", converted["ground_truth"])
        self.assertIn("Market Trends API", converted["tools"])

    def test_extract_json_tools_from_system(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        parsed = extract_json_tools("Available tools:\n" + json.dumps(tools))
        self.assertEqual(parsed[0]["function"]["name"], "get_weather")


if __name__ == "__main__":
    unittest.main()
