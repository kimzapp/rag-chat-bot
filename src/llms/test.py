from ast import parse
import requests, json

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "gemma3_1b_base", "prompt": "Hello"}
)

for line in resp.text.splitlines():
    parsed_response = json.loads(line)
    response_message = parsed_response['response']
    print(response_message, end='', flush=True)
