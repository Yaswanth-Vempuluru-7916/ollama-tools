import os
import time
import requests
from typing import Dict, Any, Callable
from urllib.parse import quote
from ollama import chat
from dotenv import load_dotenv
from datetime import datetime

# Load .env variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
TOKEN = os.getenv("TOKEN")
API_TOKEN = f"Bearer {TOKEN}"

# Define container options
CONTAINERS = [
    "/stage-bit-ponder", "/staging-cobi-v2", "/staging-evm-relay", "/staging-evm-watcher",
    "/staging-info-server", "/staging-quote", "/quote-staging", "/solana-relayer-staging",
    "/solana-watcher-staging", "/starkner-watcher-staging",
]

# Function to fetch logs from the API
def fetch_logs(container: str, start_time: int = None, end_time: int = None, limit: int = 100) -> Dict[str, Any]:
    current_time = int(time.time())
    start = start_time if start_time is not None else current_time - 3600  # Default to last hour
    end = end_time if end_time is not None else current_time
    query = quote(f'{{container="{container}"}}')
    url = f"{BASE_URL}?query={query}&start={start}&end={end}&limit={limit}"

    response = requests.get(url, headers={
        "Authorization": API_TOKEN,
        "Content-Type": "application/json"
    })

    if not response.ok:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    return response.json()

# Tool definition for fetching logs
fetch_logs_tool = {
    'type': 'function',
    'function': {
        'name': 'fetch_logs',
        'description': (
            'Fetch logs from a specified container in the staging environment. '
            'If not specified, uses the last hour as the time range and 100 as the limit (max 5000). '
            'Times should be in Unix seconds.'
        ),
        'parameters': {
            'type': 'object',
            'required': ['container'],
            'properties': {
                'container': {'type': 'string', 'enum': CONTAINERS, 'description': 'The container to fetch logs from'},
                'start_time': {'type': 'integer', 'description': 'Start time in Unix seconds (optional, defaults to 1 hour ago)'},
                'end_time': {'type': 'integer', 'description': 'End time in Unix seconds (optional, defaults to now)'},
                'limit': {'type': 'integer', 'description': 'Maximum number of log entries (default 100, max 5000)'}
            },
        },
    },
}

# Main execution
available_functions: Dict[str, Callable] = {
    'fetch_logs': fetch_logs,
}

def process_prompt(prompt: str) -> str:
    # Step 1: Initial chat to interpret the prompt and call fetch_logs
    initial_response = chat(
        'llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        tools=[fetch_logs_tool],
    )

    if initial_response.message.tool_calls:
        for tool in initial_response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                try:
                    print(f"Calling function: {tool.function.name}")
                    print(f"Arguments from Ollama: {tool.function.arguments}")
                    
                    # Resolve arguments with defaults and compute actual values
                    args = {'container': tool.function.arguments['container']}
                    current_time = int(time.time())
                    resolved_start = tool.function.arguments.get('start_time', None) if 'start_time' in tool.function.arguments else current_time - 3600
                    resolved_end = tool.function.arguments.get('end_time', None) if 'end_time' in tool.function.arguments else current_time
                    resolved_limit = tool.function.arguments.get('limit', 100)
                    print(f"Actual Arguments Used: {{'container': '{args['container']}', 'start_time': {resolved_start}, 'end_time': {resolved_end}, 'limit': {resolved_limit}}}")
                    
                    logs = function_to_call(args['container'], resolved_start, resolved_end, resolved_limit)
                    
                    # Step 2: Extract logs
                    log_entries = logs.get('data', {}).get('result', [])
                    if not log_entries:
                        return "No logs found for the specified container and time range."

                    # Extract all logs and convert nanosecond timestamps to seconds
                    raw_logs = []
                    for entry in log_entries:
                        values = entry.get('values', [])
                        if values and len(values) > 0:
                            for ts, msg in values:
                                ts_seconds = int(int(ts) / 1_000_000_000)  # Convert nanoseconds to seconds
                                raw_logs.append(f"Timestamp: {ts_seconds}, Message: {msg}")
                    
                    raw_logs_str = "\n".join(raw_logs)  # Send all logs, no truncation
                    print(f"Raw Logs Extracted: {raw_logs_str}")
                    
                    # Step 3: Analyze and refine with all logs
                    analysis_prompt = (
                        f"Here are the raw logs fetched based on the user's request:\n\n{raw_logs_str}\n\n"
                        f"Total log entries retrieved: {len(raw_logs)}\n\n"
                        f"Now, based on the original prompt '{prompt}', analyze these logs and provide a clear, "
                        "human-readable response. Convert timestamps (in Unix seconds) to readable dates (UTC), "
                        "extract key details, and address the user's specific intent (e.g., listing logs, finding errors). "
                        "Summarize patterns if applicable."
                    )

                    try:
                        analysis_response = chat(
                            'llama3.1',
                            messages=[{'role': 'user', 'content': analysis_prompt}],
                        )
                        refined_result = analysis_response.message.content
                        print(f"Refined Response: {refined_result}")
                        return refined_result
                    except Exception as e:
                        return f"Error analyzing logs with Ollama: {str(e)}\nRaw Logs:\n{raw_logs_str}"

                except Exception as e:
                    error_msg = f"Error fetching logs: {str(e)}"
                    print(error_msg)
                    return error_msg
            else:
                print(f"Function {tool.function.name} not found")
                return "Tool not found."
    else:
        return initial_response.message.content

# Example usage
if __name__ == "__main__":
    prompts = [
        "Fetch the logs of /staging-cobi-v2 start=1743750222 and end = 1743753822",
        "Fetch the logs of /staging-cobi-v2 and analyse the logs or errors or whatever",
    ]

    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        result = process_prompt(prompt)
        print(f"Final Output:\n{result}")