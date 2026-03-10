# the heat equation is "how fast temperature changes in time depends on how curved it is in space"
import numpy as np
import matplotlib.pyplot as plt


def solve(alpha=0.004, sigma=0.1, dx=0.01, dt=0.0025, t_final=5.0, plot=False):
    """
    Solves the 1D heat equation:  dT/dt = alpha * d²T/dx²

    Initial condition: Gaussian bump centered at x=0.5
    Boundary conditions: T=0 at x=0 and x=1

    Returns
    -------
    x : spatial grid
    T : final temperature profile
    """

    # stability check
    r = alpha * dt / dx**2
    if r > 0.5:
        raise ValueError(f"Unstable! r = {r:.4f} > 0.5. Reduce dt or alpha.")

    # grid
    x = np.arange(0, 1 + dx, dx)

    # initial condition — Gaussian centered at 0.5 so it's away from boundaries
    T = np.exp(-(x - 0.5)**2 / (2 * sigma**2))

    # time loop
    t = 0
    while t < t_final:
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        T_new[0]  = 0  # left BC
        T_new[-1] = 0  # right BC
        T = T_new
        t += dt

    if plot:
        plt.plot(x, T)
        plt.xlabel("x")
        plt.ylabel("T(x)")
        plt.title(f"Heat diffusion at t={t_final}s  (alpha={alpha})")
        plt.show()

    return x, T


if __name__ == "__main__":
    x, T = solve(alpha=0.004, sigma=0.1, plot=True, t_final=300)
    print(f"Max temperature at t=___: {T.max():.6f}")