# mcp_server.py
from mcp.server.fastmcp import FastMCP
import os
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
t_final_c = float(os.environ.get("HEAT_T_FINAL", "1.0"))

# Initial Conditions 
x = jnp.linspace(-1, 1, N)
T_0 = jnp.exp(-(x + 0.5)**2 / 0.001)[jnp.newaxis, :]

# Target logic
TRUE_ALPHA = float(os.environ.get("HEAT_TRUE_ALPHA", "0.006"))
target_T = solve(T_0, TRUE_ALPHA, dt, t_final_c)[0]

if os.environ.get("CORRUPT_TARGET") == "1":
    # Add a spatial sine wave corruption so exact matching is impossible
    target_T += 0.1 * jnp.sin(10 * jnp.pi * x)

@mcp.tool()
def evaluate_mse(alpha: float) -> str:
    """
    Evaluate the Mean Squared Error (MSE) of the simulated temperature profile
    using a given alpha compared to the target profile. Adjust your next alpha 
    guess based on the MSE trend to minimize it.
    """
    pred_T = solve(T_0, alpha, dt, t_final_c)[0]
    
    sensor_indices = jnp.array([N // 4, N // 2])
    
    # mse only at that point
    pred_sensors = pred_T[sensor_indices]
    target_sensors = target_T[sensor_indices]
    mse = float(np.mean((pred_sensors - target_sensors)**2))
    
    return json.dumps({
        "alpha": alpha,
        "mse": mse
    })

if __name__ == "__main__":
    mcp.run()