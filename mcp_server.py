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


def _curve_width(x, T, threshold=0.1):
    """Estimate curve width as the x-range where T > threshold * max(T)."""
    max_T = T.max()
    if max_T < 1e-10:
        return 0.0
    above = x[T > threshold * max_T]
    if len(above) < 2:
        return 0.0
    return float(above[-1] - above[0])


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


def _hint_compare_to_target(alpha: float, target_T: list[float], sigma: float = 0.1) -> str:
    """
    Deprecated: use compare_to_target instead

    Run the solver with a given alpha and compare to the target temperature profile.
    Returns MSE plus rich shape hints so the agent can reason about which direction
    to adjust alpha

    alpha    : thermal diffusivity to test
    target_T : observed/target temperature profile (list of floats)
    sigma    : width of initial Gaussian (default 0.1)

    Returns
    JSON with alpha, mse, peak_error, width_error, and a hint string
    """
    x, T = solve(alpha=alpha, sigma=sigma)
    target = np.array(target_T)

    mse         = float(np.mean((T - target) ** 2))
    peak_T      = float(T.max())
    peak_target = float(target.max())
    peak_error  = float(peak_T - peak_target)   # + too peaked, - too flat

    width_T      = _curve_width(x, T)
    width_target = _curve_width(x, target)
    width_error  = float(width_T - width_target)  # + too wide, - too narrow

    # hint we are doing (this is pretty basic right now)
    if mse < 1e-6:
        hint = "Alpha is correct"
    elif peak_error > 0:
        hint = (f"Curve is too peaked by {peak_error:.4f} and too narrow by "
                f"{-width_error:.4f} — try a LARGER alpha (more diffusion).")
    else:
        hint = (f"Curve is too flat by {-peak_error:.4f} and too wide by "
                f"{width_error:.4f} — try a SMALLER alpha (less diffusion).")

    return json.dumps({
        "alpha":       alpha,
        "mse":         mse,
        "peak_error":  peak_error,   # + too peaked, - too flat
        "width_error": width_error,  # + too wide,   - too narrow
        "hint":        hint,
    })


if __name__ == "__main__":
    mcp.run()