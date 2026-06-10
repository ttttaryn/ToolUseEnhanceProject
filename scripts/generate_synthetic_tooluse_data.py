from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


TOOLS: dict[str, dict[str, Any]] = {
    "get_weather": {
        "description": "Get weather for a city and date.",
        "properties": {"city": "string", "date": "string"},
        "required": ["city"],
    },
    "search_flights": {
        "description": "Search available flights.",
        "properties": {"from_city": "string", "to_city": "string", "date": "string"},
        "required": ["from_city", "to_city", "date"],
    },
    "book_flight": {
        "description": "Book a flight using a flight id.",
        "properties": {"flight_id": "string", "passenger_name": "string"},
        "required": ["flight_id", "passenger_name"],
    },
    "cancel_flight": {
        "description": "Cancel an existing flight booking.",
        "properties": {"booking_id": "string"},
        "required": ["booking_id"],
    },
    "create_calendar_event": {
        "description": "Create a calendar event.",
        "properties": {"title": "string", "date": "string", "time": "string"},
        "required": ["title", "date"],
    },
    "delete_calendar_event": {
        "description": "Delete a calendar event.",
        "properties": {"event_id": "string"},
        "required": ["event_id"],
    },
    "send_email": {
        "description": "Send an email.",
        "properties": {"recipient": "string", "subject": "string", "body": "string"},
        "required": ["recipient", "subject", "body"],
    },
    "draft_email": {
        "description": "Draft an email without sending it.",
        "properties": {"recipient": "string", "subject": "string", "body": "string"},
        "required": ["recipient", "subject"],
    },
    "track_package": {
        "description": "Track a package by tracking number.",
        "properties": {"tracking_number": "string", "carrier": "string"},
        "required": ["tracking_number"],
    },
    "get_stock_price": {
        "description": "Get the latest stock price.",
        "properties": {"ticker": "string"},
        "required": ["ticker"],
    },
    "convert_currency": {
        "description": "Convert money from one currency to another.",
        "properties": {"amount": "number", "from_currency": "string", "to_currency": "string"},
        "required": ["amount", "from_currency", "to_currency"],
    },
    "calculate_tip": {
        "description": "Calculate a restaurant tip.",
        "properties": {"bill_amount": "number", "tip_percent": "number"},
        "required": ["bill_amount", "tip_percent"],
    },
    "find_restaurant": {
        "description": "Find restaurants by city and cuisine.",
        "properties": {"city": "string", "cuisine": "string"},
        "required": ["city"],
    },
    "reserve_restaurant": {
        "description": "Reserve a restaurant table.",
        "properties": {"restaurant_name": "string", "date": "string", "time": "string", "party_size": "integer"},
        "required": ["restaurant_name", "date", "time", "party_size"],
    },
    "get_directions": {
        "description": "Get directions between two places.",
        "properties": {"origin": "string", "destination": "string", "mode": "string"},
        "required": ["origin", "destination"],
    },
}


NORMAL_TEMPLATES = [
    ("What is the weather in {city} on {date}?", "get_weather", {"city": "{city}", "date": "{date}"}, ["get_weather", "get_directions", "find_restaurant"]),
    ("Find flights from {from_city} to {to_city} on {date}.", "search_flights", {"from_city": "{from_city}", "to_city": "{to_city}", "date": "{date}"}, ["search_flights", "book_flight", "cancel_flight"]),
    ("Book flight {flight_id} for {name}.", "book_flight", {"flight_id": "{flight_id}", "passenger_name": "{name}"}, ["book_flight", "search_flights", "cancel_flight"]),
    ("Cancel my flight booking {booking_id}.", "cancel_flight", {"booking_id": "{booking_id}"}, ["cancel_flight", "book_flight", "search_flights"]),
    ("Add {event_title} to my calendar on {date} at {time}.", "create_calendar_event", {"title": "{event_title}", "date": "{date}", "time": "{time}"}, ["create_calendar_event", "delete_calendar_event", "send_email"]),
    ("Send {recipient} an email with subject {subject} saying {body}.", "send_email", {"recipient": "{recipient}", "subject": "{subject}", "body": "{body}"}, ["send_email", "draft_email", "create_calendar_event"]),
    ("Draft an email to {recipient} about {subject}.", "draft_email", {"recipient": "{recipient}", "subject": "{subject}", "body": ""}, ["draft_email", "send_email", "create_calendar_event"]),
    ("Track package {tracking_number} shipped by {carrier}.", "track_package", {"tracking_number": "{tracking_number}", "carrier": "{carrier}"}, ["track_package", "get_directions", "send_email"]),
    ("What is the latest price of {ticker} stock?", "get_stock_price", {"ticker": "{ticker}"}, ["get_stock_price", "convert_currency", "calculate_tip"]),
    ("Convert {amount} {from_currency} to {to_currency}.", "convert_currency", {"amount": "{amount}", "from_currency": "{from_currency}", "to_currency": "{to_currency}"}, ["convert_currency", "get_stock_price", "calculate_tip"]),
    ("Calculate a {tip_percent}% tip on a ${bill_amount} bill.", "calculate_tip", {"bill_amount": "{bill_amount}", "tip_percent": "{tip_percent}"}, ["calculate_tip", "convert_currency", "find_restaurant"]),
    ("Find {cuisine} restaurants in {city}.", "find_restaurant", {"city": "{city}", "cuisine": "{cuisine}"}, ["find_restaurant", "reserve_restaurant", "get_directions"]),
    ("Reserve a table at {restaurant} for {party_size} people on {date} at {time}.", "reserve_restaurant", {"restaurant_name": "{restaurant}", "date": "{date}", "time": "{time}", "party_size": "{party_size}"}, ["reserve_restaurant", "find_restaurant", "create_calendar_event"]),
    ("Give me directions from {origin} to {destination} by {mode}.", "get_directions", {"origin": "{origin}", "destination": "{destination}", "mode": "{mode}"}, ["get_directions", "find_restaurant", "get_weather"]),
]

HARD_TEMPLATES = [
    ("I found flight {flight_id}; reserve it for {name}.", "book_flight", {"flight_id": "{flight_id}", "passenger_name": "{name}"}, ["search_flights", "book_flight", "cancel_flight"]),
    ("I need options, not a booking, from {from_city} to {to_city} on {date}.", "search_flights", {"from_city": "{from_city}", "to_city": "{to_city}", "date": "{date}"}, ["book_flight", "search_flights", "cancel_flight"]),
    ("Write an email draft to {recipient} about {subject}, but do not send it.", "draft_email", {"recipient": "{recipient}", "subject": "{subject}", "body": ""}, ["send_email", "draft_email", "create_calendar_event"]),
    ("Actually send this email to {recipient}: subject {subject}, body {body}.", "send_email", {"recipient": "{recipient}", "subject": "{subject}", "body": "{body}"}, ["draft_email", "send_email", "create_calendar_event"]),
    ("Remove calendar item {event_id}; do not create a new event.", "delete_calendar_event", {"event_id": "{event_id}"}, ["create_calendar_event", "delete_calendar_event", "send_email"]),
    ("I only want restaurant suggestions for {cuisine} food in {city}, no reservation yet.", "find_restaurant", {"city": "{city}", "cuisine": "{cuisine}"}, ["reserve_restaurant", "find_restaurant", "get_directions"]),
    ("Reserve {restaurant} for {party_size} people on {date} at {time}; I already chose the place.", "reserve_restaurant", {"restaurant_name": "{restaurant}", "date": "{date}", "time": "{time}", "party_size": "{party_size}"}, ["find_restaurant", "reserve_restaurant", "create_calendar_event"]),
]

IRRELEVANCE_QUESTIONS = [
    "Explain why the sky is blue in one sentence.",
    "Write a short thank-you note for my teacher.",
    "What are three benefits of regular sleep?",
    "Summarize the idea of reinforcement learning without using tools.",
    "Give me a polite way to decline an invitation.",
    "What is the difference between precision and recall?",
    "Tell me a fun fact about natural language processing.",
    "Translate 'good morning' into French.",
    "How should I prepare for a ten-minute presentation?",
    "List two common causes of overfitting.",
]

MISSING_PARAM_QUESTIONS = [
    ("Book a flight for me tomorrow.", ["search_flights", "book_flight", "cancel_flight"]),
    ("Reserve a restaurant table tonight.", ["reserve_restaurant", "find_restaurant", "create_calendar_event"]),
    ("Send an email to my teammate.", ["send_email", "draft_email", "create_calendar_event"]),
    ("Track my package.", ["track_package", "send_email", "get_directions"]),
    ("Convert my money to euros.", ["convert_currency", "get_stock_price", "calculate_tip"]),
    ("Add a meeting to my calendar.", ["create_calendar_event", "delete_calendar_event", "send_email"]),
]

VALUES = {
    "city": ["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Chengdu", "Seattle", "Boston", "Paris"],
    "from_city": ["Beijing", "Shanghai", "Guangzhou", "Boston", "Seattle"],
    "to_city": ["Shanghai", "Beijing", "Chengdu", "New York", "Paris"],
    "date": ["today", "tomorrow", "Friday", "2026-06-15", "next Monday"],
    "time": ["09:00", "14:30", "18:00", "20:15"],
    "flight_id": ["CA123", "MU568", "DL204", "UA881"],
    "booking_id": ["BK1002", "FL7781", "RES9021"],
    "name": ["Alice Chen", "Bob Li", "Ming Zhao", "Taryn Wang"],
    "event_title": ["project sync", "NLP presentation", "doctor appointment", "team review"],
    "recipient": ["alice@example.com", "bob@example.com", "team@example.com"],
    "subject": ["project update", "meeting notes", "schedule change", "thank you"],
    "body": ["I will send the slides tonight", "Please review the notes", "The meeting moved to Friday"],
    "tracking_number": ["YT123456789", "SF987654321", "1Z999AA10123456784"],
    "carrier": ["SF Express", "UPS", "FedEx", "DHL"],
    "ticker": ["AAPL", "MSFT", "NVDA", "BABA"],
    "amount": [25, 80, 120.5, 300],
    "from_currency": ["USD", "CNY", "EUR"],
    "to_currency": ["CNY", "USD", "JPY", "EUR"],
    "tip_percent": [15, 18, 20, 22],
    "bill_amount": [42, 68.5, 120, 256],
    "cuisine": ["Sichuan", "Italian", "Japanese", "vegetarian"],
    "restaurant": ["Lotus Garden", "Pasta House", "Sakura Bistro", "Green Table"],
    "party_size": [2, 3, 4, 6],
    "origin": ["Peking University", "Beijing South Railway Station", "Times Square", "the hotel"],
    "destination": ["Capital Airport", "Tsinghua University", "Central Park", "the conference center"],
    "mode": ["driving", "walking", "transit"],
    "event_id": ["EVT1001", "CAL778", "MEET42"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic BFCL-style synthetic tool-use data.")
    parser.add_argument("--output", type=Path, default=Path("data/raw_selfbuilt/synthetic_tooluse_1000.jsonl"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_dataset(count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {args.output}")


def generate_dataset(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    quotas = [
        ("single_turn", int(count * 0.40)),
        ("hard_tool_selection", int(count * 0.25)),
        ("irrelevance", int(count * 0.25)),
    ]
    used = sum(quota for _, quota in quotas)
    quotas.append(("missing_param", count - used))

    rows: list[dict[str, Any]] = []
    for task_type, quota in quotas:
        for _ in range(quota):
            idx = len(rows)
            if task_type == "single_turn":
                rows.append(build_tool_record(idx, rng, NORMAL_TEMPLATES, task_type))
            elif task_type == "hard_tool_selection":
                rows.append(build_tool_record(idx, rng, HARD_TEMPLATES, task_type))
            elif task_type == "irrelevance":
                rows.append(build_no_tool_record(idx, rng, task_type))
            else:
                rows.append(build_missing_param_record(idx, rng))
    rng.shuffle(rows)
    for idx, row in enumerate(rows):
        row["id"] = f"synthetic_{idx:04d}"
    return rows


def build_tool_record(index: int, rng: random.Random, templates: list[tuple[str, str, dict[str, str], list[str]]], task_type: str) -> dict[str, Any]:
    question_template, tool_name, arg_template, tool_names = rng.choice(templates)
    values = sample_values(rng)
    question = question_template.format(**values)
    arguments = {key: coerce_value(value.format(**values)) for key, value in arg_template.items()}
    return {
        "id": f"synthetic_{index:04d}",
        "question": question,
        "tools": [tool_schema(name) for name in tool_names],
        "ground_truth": {"name": tool_name, "arguments": arguments},
        "task_type": task_type,
        "source": "synthetic_rule_based",
    }


def build_no_tool_record(index: int, rng: random.Random, task_type: str) -> dict[str, Any]:
    tool_names = rng.sample(list(TOOLS), 3)
    return {
        "id": f"synthetic_{index:04d}",
        "question": rng.choice(IRRELEVANCE_QUESTIONS),
        "tools": [tool_schema(name) for name in tool_names],
        "ground_truth": "none",
        "task_type": task_type,
        "source": "synthetic_rule_based",
    }


def build_missing_param_record(index: int, rng: random.Random) -> dict[str, Any]:
    question, tool_names = rng.choice(MISSING_PARAM_QUESTIONS)
    return {
        "id": f"synthetic_{index:04d}",
        "question": question,
        "tools": [tool_schema(name) for name in tool_names],
        "ground_truth": "none",
        "task_type": "missing_param",
        "source": "synthetic_rule_based",
    }


def sample_values(rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in VALUES.items()}


def tool_schema(name: str) -> dict[str, Any]:
    spec = TOOLS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": value_type}
                    for key, value_type in spec["properties"].items()
                },
                "required": spec["required"],
            },
        },
    }


def coerce_value(value: Any) -> Any:
    if isinstance(value, str) and value.replace(".", "", 1).isdigit():
        return float(value) if "." in value else int(value)
    return value


if __name__ == "__main__":
    main()
