# mcp_server.py
from mcp.server.fastmcp import FastMCP
import numpy as np
import json

from solver import solve

mcp = FastMCP("heat-diffusion")


@mcp.tool()
def run_solver(alpha: float, sigma: float = 0.1) -> str:
    """
    Run the 1D heat diffusion solver and return the final temperature profile.

    Parameters
    ----------
    alpha : thermal diffusivity to test
    sigma : width of the initial Gaussian (default 0.1)

    Returns
    -------
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
    Run the solver with a given alpha and compare to the target temperature profile.
    Returns MSE plus rich shape hints so the agent can reason about which direction
    to adjust alpha.

    Parameters
    ----------
    alpha    : thermal diffusivity to test
    target_T : observed/target temperature profile (list of floats)
    sigma    : width of initial Gaussian (default 0.1)

    Returns
    -------
    JSON with alpha, mse, peak_error, width_error, and a hint string
    """
    x, T = solve(alpha=alpha, sigma=sigma)
    target = np.array(target_T)

    mse         = float(np.mean((T - target) ** 2))
    peak_T      = float(T.max())
    peak_target = float(target.max())
    peak_error  = float(peak_T - peak_target)   # + = too peaked, - = too flat

    width_T      = _curve_width(x, T)
    width_target = _curve_width(x, target)
    width_error  = float(width_T - width_target)  # + = too wide, - = too narrow

    # hint
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