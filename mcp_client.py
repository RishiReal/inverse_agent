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
    # load heat equation context
    with open("heat_equation_context.txt", "r") as f:
        physics_context = f.read()
    
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
                "role": "system",
                "content": (
                    f"You are an expert in heat diffusion and inverse problems. "
                    f"Use the following reference material to reason analytically "
                    f"about how alpha affects the temperature profile:\n\n"
                    f"{physics_context}"
                    f"IMPORTANT: You must ONLY use the compare_to_target tool to evaluate alpha. "
                    f"Do NOT compute MSE or gradients yourself. "
                    f"Do NOT return any results without calling the tool first."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Find the value of alpha that produced this target temperature "
                    f"profile after t=5 seconds of heat diffusion.\n\n"
                    f"IMPORTANT: First use the peak height formula from your reference "
                    f"sheet to compute a good initial guess for alpha. Then refine it "
                    f"using gradient descent with compare_to_target.\n\n"
                    f"Use adaptive learning rate: lr = 0.01 * alpha / abs(gradient)\n"
                    f"Stop when MSE < 1e-4 or after 10 steps.\n\n"
                    f"Target profile (first 20 of {len(target_list)} values): "
                    f"{json.dumps(target_list[:20])}\n"
                    f"Target peak value: {max(target_list):.6f}\n"
                    f"Target average value: {np.mean(target_list):.6f}"
                ),
            }
        ]

        print("\n--- Agent starting ---\n")

        for step in range(15):
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
 
            msg = response.choices[0].message
            messages.append(msg)
 
            if not msg.tool_calls:
                print("\n--- Agent finished ---")
                print(msg.content)
                break
 
            mse_achieved = False
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                alpha_tried = args.get('alpha')
                print(f"Step {step+1}: calling {tool_call.function.name} with alpha={alpha_tried}")
 
                if tool_call.function.name == "compare_to_target":
                    args["target_T"] = target_list
 
                result = await session.call_tool(tool_call.function.name, args)
                result_text = result.content[0].text
                result_data = json.loads(result_text)
                mse = result_data.get('mse', float('inf'))
                print(f"         MSE={mse:.2e}  gradient={result_data.get('gradient', 'N/A'):.4f}")
 
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
 
                if mse < 1e-4:
                    mse_achieved = True
                    print(f"\nStopping early: MSE {mse:.2e} is < 1e-4")
                    print(f"Agent found alpha: {alpha_tried}")
                    break

            if mse_achieved:
                break
 
        print(f"\nTrue alpha was: {TRUE_ALPHA}")
 
 
if __name__ == "__main__":
    asyncio.run(run())