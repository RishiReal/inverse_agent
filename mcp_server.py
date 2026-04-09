# mcp_server.py
from mcp.server.fastmcp import FastMCP
import numpy as np
import json
import jax.numpy as jnp
from implicit_solver import solve

mcp = FastMCP("heat-diffusion")

# Global physics configurations that were previously in the client
N = 201
alpha_cfl = 0.01
l = 0.4  # CFL number
dx = 2 / (N - 1)
dt = l * dx**2 / alpha_cfl
t_final_c = 1.0

# Initial Conditions 
x = jnp.linspace(-1, 1, N)
T_0 = jnp.exp(-(x + 0.5)**2 / 0.001)[jnp.newaxis, :]

# Target logic
TRUE_ALPHA = 0.006
target_T = solve(T_0, TRUE_ALPHA, dt, t_final_c)[0]

@mcp.tool()
def evaluate_mse(alpha: float) -> str:
    """
    Evaluate the Mean Squared Error (MSE) of the simulated temperature profile
    using a given alpha compared to the target profile. Adjust your next alpha 
    guess based on the MSE trend to minimize it. The correct alpha is between 0.001 and 0.01.
    """
    pred_T = solve(T_0, alpha, dt, t_final_c)[0]
    mse = float(np.mean((pred_T - target_T)**2))
    
    return json.dumps({
        "alpha": alpha,
        "mse": mse
    })

if __name__ == "__main__":
    mcp.run()