# mcp_client.py
import asyncio
import json
import os
from contextlib import AsyncExitStack

from groq import Groq
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent():
    load_dotenv()
    
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # fetch tools from MCP Server
        tools_response = await session.list_tools()
        tools = []
        for t in tools_response.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                }
            })
            
        print("Tools available from MCP Server:", [t["function"]["name"] for t in tools])
        
        client = Groq()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI agent designed to find a specific thermal diffusivity (alpha). "
                    "You have access to a tool called 'evaluate_mse'. "
                    "Guess an alpha, evaluate the MSE, and continuously adjust your guesses "
                    "to minimize the MSE until you find the exact alpha where MSE is < 1e-6."
                )
            },
            {
                "role": "user",
                "content": "Please find the correct alpha using your tool. Start with a guess like 0.001."
            }
        ]

        print("\n--- Agent starting ---\n")

        for step in range(15):
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                tools=tools,
                tool_choice="required" if step == 0 else "auto",
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None
            )
            msg = response.choices[0].message
            
            # Format msg dump to append properly for multi-tool calling compatibility
            msg_dict = msg.model_dump(exclude_unset=True)
            messages.append(msg_dict)

            if not msg.tool_calls:
                print(f"Step {step+1}: Agent finished with response: {msg.content}")
                break

            mse_achieved = False
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "evaluate_mse":
                    args = json.loads(tool_call.function.arguments)
                    alpha_tried = args.get('alpha')
                    
                    try:
                        # call tool in mcp server
                        result = await session.call_tool("evaluate_mse", args)
                        result_text = result.content[0].text
                        result_data = json.loads(result_text)
                        
                        mse = result_data.get('mse', float('inf'))
                        print(f"Step {step+1}: evaluate_mse(alpha={alpha_tried}) -> MSE={mse:.2e}")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text
                        })
                        
                        if mse < 1e-6:
                            print(f"\nSuccess! Found correct alpha: {alpha_tried} with MSE {mse:.2e}")
                            mse_achieved = True
                            break
                        
                    except Exception as e:
                        print(f"Step {step+1}: Tool execution failed: {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": str(e)})
                        })

            if mse_achieved:
                break
                
        else:
            print("\n--- Agent finished without finding exact alpha within steps limit ---")

if __name__ == "__main__":
    asyncio.run(run_agent())