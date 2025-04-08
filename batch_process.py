import os
import time
import requests
from typing import Dict, Any, Callable
from urllib.parse import quote
from ollama import chat
from dotenv import load_dotenv

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

    start_api = time.time()
    response = requests.get(url, headers={
        "Authorization": API_TOKEN,
        "Content-Type": "application/json"
    })
    api_duration = time.time() - start_api
    print(f"API call duration: {api_duration:.2f} seconds")

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
    # Use a smaller, quantized model for faster inference
    model = 'llama3.1'

    # Step 1: Initial chat to interpret the prompt and call fetch_logs
    start_initial = time.time()
    initial_response = chat(
        model,
        messages=[{'role': 'user', 'content': prompt}],
        tools=[fetch_logs_tool],
    )
    initial_duration = time.time() - start_initial
    print(f"Initial prompt interpretation duration: {initial_duration:.2f} seconds")

    if initial_response.message.tool_calls:
        for tool in initial_response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                try:
                    print(f"Calling function: {tool.function.name}")
                    print(f"Arguments from Ollama: {tool.function.arguments}")
                    
                    # Resolve arguments with defaults
                    args = {'container': tool.function.arguments['container']}
                    current_time = int(time.time())
                    resolved_start = tool.function.arguments.get('start_time', current_time - 3600)
                    resolved_end = tool.function.arguments.get('end_time', current_time)
                    resolved_limit = tool.function.arguments.get('limit', 100)
                    print(f"Actual Arguments Used: {{'container': '{args['container']}', 'start_time': {resolved_start}, 'end_time': {resolved_end}, 'limit': {resolved_limit}}}")
                    
                    logs = function_to_call(args['container'], resolved_start, resolved_end, resolved_limit)
                    
                    # Step 2: Extract logs
                    start_extract = time.time()
                    log_entries = logs.get('data', {}).get('result', [])
                    if not log_entries:
                        return "No logs found for the specified container and time range."

                    raw_logs = []
                    for entry in log_entries:
                        values = entry.get('values', [])
                        if values:
                            for ts, msg in values:
                                raw_logs.append(f"Timestamp: {ts}, Message: {msg}")

                    if not raw_logs:
                        return "No logs found matching the criteria."
                    
                    extract_duration = time.time() - start_extract
                    print(f"Log extraction duration: {extract_duration:.2f} seconds")

                    # Step 3: Batch and analyze logs
                    batch_size = 50  # Process in smaller chunks
                    refined_result = ""
                    for i in range(0, len(raw_logs), batch_size):
                        batch = raw_logs[i:i + batch_size]
                        raw_logs_str = "\n".join(batch)
                        
                        # Prompt aligned with original code
                        analysis_prompt = (
                            f"Here are the raw logs fetched based on the user's request:\n\n{raw_logs_str}\n\n"
                            f"The user's prompt is: '{prompt}'. Analyze these logs and provide a clear, human-readable response "
                            f"that directly addresses the user's intent as stated in the prompt. "
                            f"Timestamps are in their original format (likely nanoseconds). "
                            f"Extract and focus on the details relevant to what the user asked for."
                        )

                        start_analysis = time.time()
                        try:
                            analysis_response = chat(
                                model,
                                messages=[{'role': 'user', 'content': analysis_prompt}],
                            )
                            batch_result = analysis_response.message.content
                            refined_result += batch_result + "\n"
                        except Exception as e:
                            return f"Error analyzing logs: {str(e)}\nRaw Logs:\n{raw_logs_str}"
                        analysis_duration = time.time() - start_analysis
                        print(f"Analysis duration for batch {i//batch_size + 1}: {analysis_duration:.2f} seconds")

                    print(f"Refined Response: {refined_result}")
                    return refined_result.strip()

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
        "Fetch the logs of /staging-cobi-v2 start_time: 1744002429, end_time: 1744006029, limit: 100. check whether txid:f8c67a65e30bcbc68e29939afe110a2d5444a46504e5c3a84b087a64c8a24b71 exists or not",
    ]

    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        start_time = time.time()
        result = process_prompt(prompt)
        print(f"Final Output:\n{result}")
        total_duration = time.time() - start_time
        print(f"Total time taken: {total_duration:.2f} seconds")