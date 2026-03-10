# mcp_client.py
# MCP client connects to mcp_server.py via stdio,
# then uses Gemini to iteratively find the alpha that produced a target profile

import asyncio
import json
import os
from contextlib import AsyncExitStack

import numpy as np
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
import os

from solver import solve  # generate the target profile

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

async def run():
    # generate a target profile with a known alpha
    TRUE_ALPHA = 0.006
    _, target_T = solve(alpha=TRUE_ALPHA)
    target_list = target_T.tolist()
    print(f"Target generated with alpha={TRUE_ALPHA}  (agent must find this)\n")

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # list tools so we can pass them to Gemini
        tools_response = await session.list_tools()
        tools = []
        for t in tools_response.tools:
            # copy the schema because need to modify
            schema = t.inputSchema.copy()
            if t.name == "compare_to_target":
                # Remove target_T so LLM doesn't send
                if "target_T" in schema.get("properties", {}):
                    del schema["properties"]["target_T"]
                if "target_T" in schema.get("required", []):
                    schema["required"].remove("target_T")
            
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema,
                }
            })
        print("Tools available:", [t["function"]["name"] for t in tools])

        load_dotenv()
        # load API key
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are solving an inverse heat diffusion problem. "
                    f"A temperature profile was measured after 5 seconds. "
                    f"Your job is to find the value of alpha (thermal diffusivity) "
                    f"that produced it. Alpha is somewhere between 0.001 and 0.02.\n\n"
                    f"Use the compare_to_target tool repeatedly, adjusting alpha "
                    f"each time to minimize the MSE. "
                    f"Stop when MSE < 1e-6 or after 10 attempts.\n\n"
                    f"Target profile: {json.dumps(target_list[:20])}... "
                    f"(first 20 of {len(target_list)} values)"
                ),
            }
        ]

        print("\n--- Agent starting ---\n")

        for step in range(15):  # max iterations
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                tools=tools,
                messages=messages,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            messages.append(msg)

            # check if agent is done (no tool calls)
            if not msg.tool_calls:
                print("\n--- Agent finished ---")
                print(msg.content)
                break

            # process tool calls
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                print(f"Step {step+1}: calling {tool_call.function.name} with alpha={args.get('alpha')}")

                # inject the full target profile if needed
                if tool_call.function.name == "compare_to_target":
                    args["target_T"] = target_list

                # call the tool on the MCP server
                result = await session.call_tool(tool_call.function.name, args)
                result_text = result.content[0].text
                result_data = json.loads(result_text)
                print(f"         MSE={result_data.get('mse', 'N/A'):.2e}  hint: {result_data.get('hint', '')}")

                # add tool result back to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        print(f"\nTrue alpha was: {TRUE_ALPHA}")


if __name__ == "__main__":
    asyncio.run(run())