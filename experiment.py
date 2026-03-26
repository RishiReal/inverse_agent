# experiment.py
# Runs the LLM agent N times for each t_final and tabulates results.
# Fixed alpha=0.005, t_final in [5, 10, 20], N_ATTEMPTS per t_final.

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
 
import numpy as np
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 
import solver
from solver import solve
 
load_dotenv()
 
# ── config ────────────────────────────────────────────────────────────────────
TRUE_ALPHA  = 0.005
T_FINALS    = [5.0, 10.0, 20.0]
N_ATTEMPTS  = 3
SUCCESS_MSE = 1e-4
 
# load physics context once
with open("heat_equation_context.txt", "r") as f:
    PHYSICS_CONTEXT = f.read()
 
 
async def run_once(attempt_id, true_alpha, t_final, target_list):
    """Run one agent attempt. Returns result dict."""
    import io, sys
 
    # suppress solver prints during target generation
    solver.call_count = 0
 
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
 
    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
 
        # build tools, hide target_T from LLM
        tools_response = await session.list_tools()
        tools = []
        for t in tools_response.tools:
            schema = t.inputSchema.copy()
            if t.name == "compare_to_target":
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
 
        # import here so we can swap LLM easily
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
 
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert in heat diffusion and inverse problems. "
                    f"Use the following reference material:\n\n{PHYSICS_CONTEXT}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Find the value of alpha that produced this target temperature "
                    f"profile after t={t_final} seconds of heat diffusion.\n\n"
                    f"First use the peak height formula to get an initial guess. "
                    f"Then refine with gradient descent: alpha_new = alpha - lr * gradient "
                    f"where lr = 0.1 * alpha / abs(gradient).\n"
                    f"Stop when MSE < 1e-5 or after 10 steps.\n\n"
                    f"Target peak value: {max(target_list):.6f}\n"
                    f"Target average value: {np.mean(target_list):.6f}\n"
                    f"Target profile (first 20 values): {json.dumps(target_list[:20])}"
                ),
            }
        ]
 
        final_alpha = None
        final_mse   = None
        n_steps     = 0
 
        for step in range(15):
            try:
                response = client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=messages,
                    tools=tools,
                    tool_choice={
                        "type": "function",
                        "function": {"name": "compare_to_target"}
                    },
                )
            except Exception as e:
                print(f"      [Step {step}] API Error: {getattr(e, 'message', str(e))}")
                break
 
            msg = response.choices[0].message
            messages.append(msg)
 
            if not msg.tool_calls:
                break
 
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name.split("<")[0].strip()
                args = json.loads(tool_call.function.arguments)
 
                if tool_name == "compare_to_target":
                    args["target_T"] = target_list
                    n_steps += 1
 
                result      = await session.call_tool(tool_name, args)
                result_text = result.content[0].text
                result_data = json.loads(result_text)
 
                if "mse" in result_data:
                    final_alpha = args.get("alpha")
                    final_mse   = result_data["mse"]
 
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
 
            if final_mse is not None and final_mse < 1e-5:
                break
 
        success   = final_mse is not None and final_mse < SUCCESS_MSE
        alpha_err = abs(final_alpha - true_alpha) / true_alpha * 100 if final_alpha else None
 
        return {
            "t_final":      t_final,
            "attempt":      attempt_id,
            "found_alpha":  final_alpha,
            "alpha_err_pct": alpha_err,
            "final_mse":    final_mse,
            "n_steps":      n_steps,
            "solver_calls": solver.call_count,
            "success":      success,
        }
 
 
async def main():
    print(f"Running {N_ATTEMPTS} attempts × {len(T_FINALS)} t_finals")
    print(f"True alpha = {TRUE_ALPHA}\n")
 
    all_results = {}
 
    for t_final in T_FINALS:
        print(f"\n{'='*60}")
        print(f"t_final = {t_final}s")
        print(f"{'='*60}")
 
        # generate target once for this t_final
        import io, sys
        sys.stdout = io.StringIO()
        _, target_T = solve(alpha=TRUE_ALPHA, t_final=t_final)
        sys.stdout = sys.__stdout__
        target_list = target_T.tolist()
 
        results = []
        for attempt in range(1, N_ATTEMPTS + 1):
            print(f"  Attempt {attempt}/{N_ATTEMPTS}...", end=" ", flush=True)
            r = await run_once(attempt, TRUE_ALPHA, t_final, target_list)
            results.append(r)
            print(f"alpha={r['found_alpha']:.5f}  MSE={r['final_mse']:.2e}  "
                  f"steps={r['n_steps']}  {'✓' if r['success'] else '✗'}")
            time.sleep(15)  # avoid rate limiting
 
        all_results[t_final] = results
 
    # ── print final table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print("RESULTS TABLE")
    print(f"{'='*75}")
    print(f"{'t_final':>8}  {'Attempt':>7}  {'Found α':>9}  {'Err%':>6}  "
          f"{'MSE':>10}  {'Steps':>6}  {'Solver Calls':>12}  {'Success':>7}")
    print("-" * 75)
 
    for t_final in T_FINALS:
        for r in all_results[t_final]:
            print(f"{r['t_final']:>8.1f}  {r['attempt']:>7}  "
                  f"{r['found_alpha']:>9.5f}  {r['alpha_err_pct']:>6.2f}%  "
                  f"{r['final_mse']:>10.2e}  {r['n_steps']:>6}  "
                  f"{r['solver_calls']:>12}  "
                  f"{'✓' if r['success'] else '✗':>7}")
        print("-" * 75)
 
    # ── summary by t_final ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY BY t_final")
    print(f"{'='*60}")
    print(f"{'t_final':>8}  {'Success':>8}  {'Avg MSE':>10}  {'Avg Err%':>9}  {'Avg Steps':>10}")
    print("-" * 55)
 
    for t_final in T_FINALS:
        results = all_results[t_final]
        successes = sum(r["success"] for r in results)
        avg_mse   = np.mean([r["final_mse"] for r in results])
        avg_err   = np.mean([r["alpha_err_pct"] for r in results])
        avg_steps = np.mean([r["n_steps"] for r in results])
        print(f"{t_final:>8.1f}  {successes}/{N_ATTEMPTS:>6}  "
              f"{avg_mse:>10.2e}  {avg_err:>9.2f}%  {avg_steps:>10.1f}")
 
 
if __name__ == "__main__":
    asyncio.run(main())