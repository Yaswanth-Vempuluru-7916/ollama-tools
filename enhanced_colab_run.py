import os 
import time
import requests
from typing import Dict, Any, Callable
from urllib.parse import quote
from ollama import chat
from dotenv import load_dotenv
from thefuzz import process
from requests.exceptions import RequestException

# Load environment variables
load_dotenv()

# ==== Configs and Constants ====
BASE_URL = os.getenv("BASE_URL")
TOKEN = os.getenv("TOKEN")
API_TOKEN = f"Bearer {TOKEN}"

DEFAULT_CONTAINER = "/staging-cobi-v2"
DEFAULT_TIME_RANGE = 3600  # 1 hour
DEFAULT_LIMIT = 100
MAX_LOOKBACK = 30 * 24 * 3600  # 30 days
API_TIMEOUT = 10  # seconds
FUZZY_MATCH_THRESHOLD = 80

CONTAINERS = [
    "/stage-bit-ponder", "/staging-cobi-v2", "/staging-evm-relay", "/staging-evm-watcher",
    "/staging-info-server", "/staging-quote", "/quote-staging", "/solana-relayer-staging",
    "/solana-watcher-staging", "/starkner-watcher-staging",
]

# ==== Tool Definition ====
fetch_logs_tool = {
    'type': 'function',
    'function': {
        'name': 'fetch_logs',
        'description': 'Retrieve logs from a container. Supports optional start_time, end_time, and limit.',
        'parameters': {
            'type': 'object',
            'required': ['container'],
            'properties': {
                'container': {'type': 'string', 'enum': CONTAINERS},
                'start_time': {'type': 'integer'},
                'end_time': {'type': 'integer'},
                'limit': {'type': 'integer', 'maximum': 5000}
            },
        },
    },
}

# ==== Log Fetching ====
def fetch_logs(container: str, start_time: int = None, end_time: int = None, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    if not API_TOKEN:
        raise ValueError("Missing API_TOKEN. Ensure .env is configured correctly.")

    now = int(time.time())
    start = start_time or now - DEFAULT_TIME_RANGE
    end = end_time or now

    if start > end:
        start, end = end, start
    if end - start > MAX_LOOKBACK:
        start = end - MAX_LOOKBACK

    query = quote(f'{{container="{container}"}}')
    url = f"{BASE_URL}?query={query}&start={start}&end={end}&limit={limit}"

    try:
        response = requests.get(url, headers={
            "Authorization": API_TOKEN,
            "Content-Type": "application/json"
        }, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        raise RuntimeError(f"Request failed for container '{container}': {e}") from e

available_functions: Dict[str, Callable] = {'fetch_logs': fetch_logs}

# ==== Prompt Processing ====
def process_prompt(prompt: str) -> tuple[str, str, str, float, float, float]:
    start_total = time.time()
    time_api = 0.0
    time_ollama2 = 0.0

    # Step 1: Enhance prompt with container list
    container_list = ", ".join(CONTAINERS)
    enhanced_prompt = f"Available containers: {container_list}\nUser query: {prompt}"

    # Step 2: Initial Ollama Call
    start_ollama1 = time.time()
    initial_response = chat(
        model='qwen2.5:14b',
        messages=[{'role': 'user', 'content': enhanced_prompt}],
        tools=[fetch_logs_tool]
    )
    time_ollama1 = time.time() - start_ollama1

    if 'tool_calls' not in initial_response['message']:
        return "", "", initial_response['message']['content'], time_ollama1, time_api, time_ollama2

    for tool in initial_response['message']['tool_calls']:
        try:
            fn = available_functions.get(tool['function']['name'])
            if not fn:
                raise ValueError(f"Unknown function: {tool['function']['name']}")

            args = tool['function']['arguments']
            query_lower = prompt.lower()
            has_start = any(word in query_lower for word in ["start", "from", "beginning"])
            has_end = any(word in query_lower for word in ["end", "to", "until"])
            has_limit = "limit" in query_lower

            # Extract container and fuzzy match it
            container = args.get('container', DEFAULT_CONTAINER).replace(" ", "-")
            if not container.startswith("/"):
                container = "/" + container
            best_match, score = process.extractOne(container, CONTAINERS)
            if score >= FUZZY_MATCH_THRESHOLD:
                container = best_match
            else:
                raise ValueError(f"Invalid container: {container}. Closest match: {best_match} (score: {score})")

            # Handle optional parameters
            start_time = args.get('start_time') if has_start else None
            end_time = args.get('end_time') if has_end else None
            limit = args.get('limit') if has_limit else DEFAULT_LIMIT

            if isinstance(start_time, str) and start_time.isdigit():
                start_time = int(start_time)
            if isinstance(end_time, str) and end_time.isdigit():
                end_time = int(end_time)
            if isinstance(limit, str) and limit.isdigit():
                limit = int(limit)

            now = int(time.time())
            actual_start = start_time if start_time is not None else now - DEFAULT_TIME_RANGE
            actual_end = end_time if end_time is not None else now

            arguments_str = f"Arguments: {{'container': '{container}', 'start_time': {actual_start}, 'end_time': {actual_end}, 'limit': {limit}}}"

            # Step 3: Fetch logs
            start_api = time.time()
            logs = fn(container, start_time, end_time, limit)
            time_api = time.time() - start_api

            log_entries = logs.get('data', {}).get('result', [])
            raw_logs = [f"Timestamp: {ts}, Message: {msg}" for entry in log_entries for ts, msg in entry.get('values', [])]
            raw_logs_str = "\n".join(raw_logs) if raw_logs else "No logs found."

            # Step 4: Follow-up Ollama call for interpretation
            start_ollama2 = time.time()
            analysis_prompt = (
                f"Here are the logs:\n\n{raw_logs_str}\n\n"
                f"The original query was: '{prompt}'. Provide a human-readable summary. "
                "Only reference what is in the logs. Do not hallucinate or invent responses."
            )

            query_specific = any(term in query_lower for term in ["find", "get", "extract", "show", "list"])
            if query_specific:
                for dtype in ["id", "code", "error", "status", "number"]:
                    if dtype in query_lower:
                        analysis_prompt += f"\n\nPlease extract any {dtype}s if available. Otherwise, report that they weren't found."
                        break

            analysis_response = chat(
                model='qwen2.5:14b',
                messages=[{'role': 'user', 'content': analysis_prompt}]
            )
            time_ollama2 = time.time() - start_ollama2

            return arguments_str, raw_logs_str, analysis_response['message']['content'], time_ollama1, time_api, time_ollama2
        except Exception as e:
            return "", "", f"Error: {str(e)}", time_ollama1, time_api, time_ollama2

# ==== Interactive CLI Test ====
def test_delay():
    print("Prompt examples: 'Fetch logs from /staging-cobi-v2', or 'quit' to exit.")
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == "quit":
            break
        start = time.time()
        arguments, raw_logs, result, t1, t2, t3 = process_prompt(prompt)
        total = time.time() - start

        print(f"\nArguments: {arguments}")
        print(f"Raw Logs:\n{raw_logs}")
        print(f"Result:\n{result}")
        print("\nTiming:")
        print(f"  Ollama (initial): {t1:.2f}s")
        print(f"  API fetch:        {t2:.2f}s")
        print(f"  Ollama (analysis):{t3:.2f}s")
        print(f"  Total time:       {total:.2f}s")

if __name__ == "__main__":
    test_delay()