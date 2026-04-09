import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Dummy tools
schema = {
    "type": "object",
    "properties": {
        "alpha": {"type": "number", "description": "Thermal diffusivity"}
    },
    "required": ["alpha"]
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "compare_to_target",
            "description": "Evaluate how close an alpha is to the target",
            "parameters": schema
        }
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Find alpha. Use the compare_to_target tool."}
]

try:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    print("Content:", msg.content)
    print("Tool calls:", msg.tool_calls)
    print("Refusal:", getattr(msg, 'refusal', None))
except Exception as e:
    print("Exception:", e)
