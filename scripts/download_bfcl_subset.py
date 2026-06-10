from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

DEFAULT_FILES = [
    "BFCL_v4_simple_python.json",
    "BFCL_v4_multiple.json",
    "BFCL_v4_irrelevance.json",
    "BFCL_v4_live_irrelevance.json",
    "BFCL_v4_live_relevance.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small BFCL v4 subset without cloning Gorilla.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL.rstrip("/"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in args.files:
        url = f"{args.base_url.rstrip('/')}/{filename}"
        output = args.output_dir / filename
        download_with_retries(url=url, output=output, retries=args.retries, timeout=args.timeout)
        validate_json(output)
        print(f"Downloaded {filename} -> {output}")


def download_with_retries(url: str, output: Path, retries: int, timeout: int) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ToolUseEnhanceProject/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            output.write_bytes(data)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = min(2**attempt, 20)
            print(f"Download failed ({attempt}/{retries}) for {url}: {exc}. Retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    raise SystemExit(
        f"Failed to download {url} after {retries} attempts: {last_error}\n"
        "If GitHub raw is blocked or slow on AutoDL, rerun with --base-url pointing to a reachable mirror, "
        "or upload the JSON files manually into data/raw."
    )


def validate_json(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except json.JSONDecodeError as exc:
        path.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded file is not valid JSON: {path}: {exc}") from exc


if __name__ == "__main__":
    main()
