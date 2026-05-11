import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# --- Grid setup ---
N = 201
x = np.linspace(-1, 1, N)
dx = 2 / (N - 1)
alpha = 0.005


# Initial condition: narrow Gaussian centered at x = -0.5
T_0 = np.exp(-(x + 0.5)**2 / 0.001)

# --- Solvers ---
def solve_explicit(T_init, lam, n_steps):
    T = T_init.copy()
    for _ in range(n_steps):
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + lam * (T[2:] - 2*T[1:-1] + T[:-2])
        T_new[0] = 0
        T_new[-1] = 0
        T = T_new
    return T

def build_tridiag(n, r):
    ab = np.zeros((3, n))
    ab[0, 1:]  = -r
    ab[1, :]   =  1 + 2*r
    ab[2, :-1] = -r
    return ab

def solve_implicit(T_init, lam, n_steps):
    T = T_init.copy()
    n_int = len(T) - 2
    ab = build_tridiag(n_int, lam)
    for _ in range(n_steps):
        b = T[1:-1].copy()
        b[0]  += lam * T[0]
        b[-1] += lam * T[-1]
        T[1:-1] = solve_banded((1, 1), ab, b)
        T[0] = 0
        T[-1] = 0
    return T

# --- 3 different lambdas: stable, borderline, unstable ---
lambdas = [0.4, 0.5, 0.75]
colors  = ['steelblue', 'orange', 'red']
n_steps_list = [30, 30, 30]

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# -- Explicit (top) --
ax_exp = axes[0]
ax_exp.plot(x, T_0, '--', color='gray', lw=2, label='Initial')
for lam, c, ns in zip(lambdas, colors, n_steps_list):
    T_exp = solve_explicit(T_0, lam, ns)
    ax_exp.plot(x, np.clip(T_exp, -0.5, 1.5), color=c, lw=2,
                label=f'λ = {lam}')
ax_exp.set_ylabel('T')
ax_exp.set_ylim(-0.5, 1.5)
ax_exp.set_title('Explicit (FTCS)')
ax_exp.legend()
ax_exp.axhline(0, color='black', linewidth=0.5)

# -- Implicit (bottom) --
ax_imp = axes[1]
ax_imp.plot(x, T_0, '--', color='gray', lw=2, label='Initial')
for lam, c, ns in zip(lambdas, colors, n_steps_list):
    T_imp = solve_implicit(T_0, lam, ns)
    ax_imp.plot(x, T_imp, color=c, lw=2, label=f'λ = {lam}')
ax_imp.set_xlabel('x')
ax_imp.set_ylabel('T')
ax_imp.set_ylim(-0.5, 1.5)
ax_imp.set_title('Implicit (Backward Euler)')
ax_imp.legend()
ax_imp.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('explicit_vs_implicit.png', dpi=150)
plt.show()
print("Saved explicit_vs_implicit.png")
