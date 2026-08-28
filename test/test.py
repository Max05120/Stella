import ollama
response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'What is the weather in Toronto?'}],
    tools=[{'type': 'function', 'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {'type': 'object', 'properties': {
            'city': {'type': 'string'}}, 'required': ['city']},
    }}],
)
print(response['message'].get('tool_calls'))