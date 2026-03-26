# mcp_server.py
from mcp.server.fastmcp import FastMCP
import numpy as np
import json

from solver import solve

mcp = FastMCP("heat-diffusion")

EPS = 1e-5 # finite difference step size

def _mse(alpha, target, sigma=0.1):
    """Helper: run solver and return MSE against target."""
    _, T = solve(alpha=alpha, sigma=sigma)
    return float(np.mean((T - np.array(target)) ** 2))

@mcp.tool()
def run_solver(alpha: float, sigma: float = 0.1) -> str:
    """
    Run the 1D heat diffusion solver and return the final temperature profile.

    alpha : thermal diffusivity to test
    sigma : width of the initial Gaussian (default 0.1)

    Returns
    JSON string with x grid and final temperature T
    """
    x, T = solve(alpha=alpha, sigma=sigma)
    return json.dumps({
        "alpha": alpha,
        "sigma": sigma,
        "x": x.tolist(),
        "T": T.tolist(),
    })



@mcp.tool()
def compare_to_target(alpha: float, target_T: list[float], sigma: float = 0.1) -> str:
    """
    Run the solver with a given alpha, compute MSE against the target profile,
    and return a finite-difference gradient dMSE/dalpha.
 
    The agent should use gradient descent to update alpha:
        alpha_new = alpha - learning_rate * gradient
 
    gradient > 0 means increasing alpha makes MSE worse so decrease alpha
    gradient < 0 means increasing alpha makes MSE better so increase alpha
 
    alpha    : thermal diffusivity to test
    target_T : observed/target temperature profile (list of floats)
    sigma    : width of initial Gaussian (default 0.1)
 
    Returns
    JSON with alpha, mse, and gradient dMSE/dalpha
    """
    mse      = _mse(alpha, target_T, sigma)
    mse_plus = _mse(alpha + EPS, target_T, sigma)
    mse_minus = _mse(alpha - EPS, target_T, sigma)
 
    # central difference gradient
    gradient = (mse_plus - mse_minus) / (2 * EPS)
 
    return json.dumps({
        "alpha":    alpha,
        "mse":      mse,
        "gradient": gradient,   # dMSE/dalpha
    })



if __name__ == "__main__":
    mcp.run()