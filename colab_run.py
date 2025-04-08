import os
import time
import requests
from typing import Dict, Any, Callable
from urllib.parse import quote
from ollama import chat
from dotenv import load_dotenv

BASE_URL = os.getenv("BASE_URL")
TOKEN = os.getenv("TOKEN")
API_TOKEN = f"Bearer {TOKEN}"

CONTAINERS = [
    "/stage-bit-ponder", "/staging-cobi-v2", "/staging-evm-relay", "/staging-evm-watcher",
    "/staging-info-server", "/staging-quote", "/quote-staging", "/solana-relayer-staging",
    "/solana-watcher-staging", "/starkner-watcher-staging",
]

def fetch_logs(container: str, start_time: int = None, end_time: int = None, limit: int = 100) -> Dict[str, Any]:
    current_time = int(time.time())
    start = start_time if start_time is not None else current_time - 3600
    end = end_time if end_time is not None else current_time
    if start > end:
        start, end = end, start
    if end - start > 30 * 24 * 3600:
        start = end - 30 * 24 * 3600
    query = quote(f'{{container="{container}"}}')
    url = f"{BASE_URL}?query={query}&start={start}&end={end}&limit={limit}"

    response = requests.get(url, headers={
        "Authorization": API_TOKEN,
        "Content-Type": "application/json"
    })

    if not response.ok:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    return response.json()

fetch_logs_tool = {
    'type': 'function',
    'function': {
        'name': 'fetch_logs',
        'description': (
            'Call fetch_logs on the server. Parameters like start_time, end_time, and limit '
            'are optional and will use defaults (last hour, 100 logs) if omitted.'
        ),
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

available_functions: Dict[str, Callable] = {'fetch_logs': fetch_logs}

def process_prompt(prompt: str) -> tuple[str, str, str, float, float, float]:
    start_total = time.time()
    time_api = 0.0
    time_ollama2 = 0.0

    # Step 1: Initial Ollama chat with tool
    start_ollama1 = time.time()
    initial_response = chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        tools=[fetch_logs_tool]
    )
    time_ollama1 = time.time() - start_ollama1

    # Step 2: Process tool calls (API fetch)
    if 'tool_calls' in initial_response['message']:
        for tool in initial_response['message']['tool_calls']:
            if function_to_call := available_functions.get(tool['function']['name']):
                try:
                    args = tool['function']['arguments']
                    # Filter args based on query content
                    query_lower = prompt.lower()
                    has_start = any(term in query_lower for term in ["start", "from", "beginning"])
                    has_end = any(term in query_lower for term in ["end", "to", "until"])
                    has_limit = "limit" in query_lower
                    
                    filtered_args = {"container": args.get('container', '/staging-cobi-v2')}
                    if has_start and 'start_time' in args:
                        filtered_args['start_time'] = args['start_time']
                    if has_end and 'end_time' in args:
                        filtered_args['end_time'] = args['end_time']
                    if has_limit and 'limit' in args:
                        filtered_args['limit'] = args['limit']
                    args = filtered_args

                    # Ensure numeric fields are integers
                    for key in ['start_time', 'end_time', 'limit']:
                        if key in args and isinstance(args[key], str) and args[key].isdigit():
                            args[key] = int(args[key])

                    # Normalize container name with leading slash and validate
                    container = args.get('container', '/staging-cobi-v2')
                    if not container.startswith('/'):
                        container = '/' + container
                    if container not in CONTAINERS:
                        raise ValueError(f"Invalid container: {container}. Must be one of {CONTAINERS}")
                    
                    start_time = args.get('start_time')
                    end_time = args.get('end_time')
                    limit = args.get('limit', 100)

                    # Compute actual values for display
                    current_time = int(time.time())
                    actual_start = start_time if start_time is not None else current_time - 3600
                    actual_end = end_time if end_time is not None else current_time
                    arguments_str = f"Arguments: {{'container': '{container}', 'start_time': {actual_start}, 'end_time': {actual_end}, 'limit': {limit}}}"

                    start_api = time.time()
                    logs = function_to_call(container, start_time, end_time, limit)
                    time_api = time.time() - start_api

                    log_entries = logs.get('data', {}).get('result', [])
                    raw_logs = []
                    for entry in log_entries:
                        for ts, msg in entry.get('values', []):
                            raw_logs.append(f"Timestamp: {ts}, Message: {msg}")
                    raw_logs_str = "\n".join(raw_logs) if raw_logs else "No logs found."

                    # Step 3: Second Ollama call for analysis with dynamic prompt
                    start_ollama2 = time.time()
                    analysis_prompt = (
                        f"Here are the results from the tool 'fetch_logs':\n\n{raw_logs_str}\n\n"
                        f"The user's original query was: '{prompt}'. Analyze these results and provide a clear, "
                        f"human-readable response that addresses the user's intent. "
                        f"Do NOT generate or hallucinate any data that is not explicitly present in the logs."
                    )

                    # Detect if user is asking for specific data
                    specific_data_terms = ["find", "get", "extract", "show", "list"]
                    data_types = ["id", "code", "error", "status", "number"]
                    requested_data = None
                    for term in specific_data_terms:
                        if term in query_lower:
                            for data_type in data_types:
                                if data_type in query_lower:
                                    requested_data = f"{data_type}s"  # Pluralize for generality (e.g., "IDs", "codes")
                                    break
                            if requested_data:
                                break
                    
                    if requested_data:
                        analysis_prompt += (
                            f" If the query asks for specific data like {requested_data}, extract them directly from the provided logs. "
                            f"If the requested data (e.g., {requested_data}) is not found, state that clearly instead of making up values."
                        )

                    analysis_response = chat(
                        model='llama3.1',
                        messages=[{'role': 'user', 'content': analysis_prompt}]
                    )
                    time_ollama2 = time.time() - start_ollama2

                    result = analysis_response['message']['content']
                    total_time = time.time() - start_total
                    return arguments_str, raw_logs_str, result, time_ollama1, time_api, time_ollama2
                except Exception as e:
                    total_time = time.time() - start_total
                    return "", "", f"Error: {str(e)}", time_ollama1, time_api, time_ollama2
    else:
        total_time = time.time() - start_total
        return "", "", initial_response['message']['content'], time_ollama1, time_api, time_ollama2

def test_delay():
    print("Type your prompt (e.g., 'Fetch logs from /staging-cobi-v2') or 'quit' to exit.")
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() == "quit":
            break
        start_time = time.time()
        arguments, raw_logs, result, ollama1_time, api_time, ollama2_time = process_prompt(prompt)
        total_time = time.time() - start_time
        print(f"\nArguments: {arguments}")
        print(f"Raw Logs:\n{raw_logs}")
        print(f"Result:\n{result}")
        print(f"\nTiming Breakdown:")
        print(f"  Initial Ollama call: {ollama1_time:.2f} seconds")
        print(f"  API fetch: {api_time:.2f} seconds")
        print(f"  Analysis Ollama call: {ollama2_time:.2f} seconds")
        print(f"  Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    test_delay()