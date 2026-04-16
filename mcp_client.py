# mcp_client.py
import asyncio
import json
import os
from contextlib import AsyncExitStack

from groq import Groq
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import datetime

LOGS_DIR = "logs"

def _write_log(log, alpha_tried, success):
    os.makedirs(LOGS_DIR, exist_ok=True)  # ensure logs/ folder exists
    t_final = os.environ.get("HEAT_T_FINAL", "unknown")
    out = {
        "t_final": float(t_final) if t_final != "unknown" else t_final,
        "timestamp": datetime.datetime.now().isoformat(),
        "success": success,
        "final_alpha": alpha_tried,
        "n_steps": len(log),
        "steps": log,
    }
    fname = os.path.join(LOGS_DIR, f"log_t{t_final}_{datetime.datetime.now().strftime('%H%M%S')}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Log saved → {fname}")

async def run_agent():
    load_dotenv()
    
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env={**os.environ}
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
                    "You are solving an inverse problem for the 1D heat equation: dT/dt = alpha * d²T/dx². "
                    "A Gaussian pulse diffused with unknown alpha. Find alpha.\n\n"
                    "You can call evaluate_mse(alpha), which returns the error between your simulation "
                    "and the target. Use previous results to guide your guesses. Stop when MSE < 1e-6."
                )
            },
            {
                "role": "user",
                "content": "Find the correct alpha. Use the evaluate_mse tool to evaluate your guesses. Stop when MSE < 1e-6."
            }
        ]

        print("\n--- Agent starting ---\n")
        log = []
        alpha_tried = None  # track last tried alpha for final log

        for step in range(25):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                    parallel_tool_calls=False
                )
            except Exception as api_err:
                err_str = str(api_err)
                if "tool_use_failed" in err_str:
                    # LLM tried to give a text answer instead of calling the tool.
                    # Force it back on track by injecting a stern reminder.
                    print(f"Step {step+1}: Agent tried to stop early — pushing it back...")
                    messages.append({
                        "role": "user",
                        "content": "You have NOT found the answer yet. You MUST call evaluate_mse with your next guess. Do not stop."
                    })
                    continue
                raise

            msg = response.choices[0].message
            
            # Gemini rejects null content values — ensure it's always a string
            msg_dict = msg.model_dump(exclude_unset=True)
            if msg_dict.get("content") is None:
                msg_dict["content"] = ""
            messages.append(msg_dict)

            if not msg.tool_calls:
                print(f"Step {step+1}: Agent finished with response: {msg.content}")
                _write_log(log, alpha_tried, success=False)
                return step + 1

            mse_achieved = False
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "evaluate_mse":
                    args = json.loads(tool_call.function.arguments)
                    alpha_tried = args.get('alpha')
                    
                    try:
                        result = await session.call_tool("evaluate_mse", args)
                        result_text = result.content[0].text
                        result_data = json.loads(result_text)
                        
                        mse = result_data.get('mse', float('inf'))
                        print(f"Step {step+1}: evaluate_mse(alpha={alpha_tried}) -> MSE={mse:.2e}")

                        log.append({
                            "step": step + 1,
                            "alpha": alpha_tried,
                            "mse": mse,
                        })  
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text
                        })
                        
                        if mse < 1e-6:
                            print(f"\nSuccess! Found correct alpha: {alpha_tried} with MSE {mse:.2e}")
                            _write_log(log, alpha_tried, success=True)
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
                return step + 1

        print("\n--- Agent finished without finding exact alpha within steps limit ---")
        _write_log(log, alpha_tried=alpha_tried, success=False)
        return -1

if __name__ == "__main__":
    asyncio.run(run_agent())