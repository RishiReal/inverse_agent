import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

N = 201
dx = 1 / 200
x = np.linspace(-1, 1, N)
T_0 = np.exp(-(x + 0.5)**2 / 0.001)

def solve_explicit(T_init, l, n_steps):
    T = T_init.copy()
    for _ in range(n_steps):
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + l * (T[2:] - 2*T[1:-1] + T[:-2])
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

def solve_implicit(T_init, l, n_steps):
    T = T_init.copy()
    n_int = len(T) - 2  # interior nodes
    ab = build_tridiag(n_int, l)
    for _ in range(n_steps):
        b = T[1:-1].copy()
        b[0]  += l * T[0]
        b[-1] += l * T[-1]
        T[1:-1] = solve_banded((1, 1), ab, b)
        T[0] = 0
        T[-1] = 0
    return T

n_steps = N

lambdas = [0.4, 0.49, 0.51, 0.6]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, label in zip(axes, ['Explicit (FTCS)', 'Implicit (Backward Euler)']):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    ax.set_title(label)
    ax.plot(x, T_0, '--', color='gray', alpha=0.4, label='Initial')
    ax.axhline(0, color='black', linewidth=0.5)

for l in lambdas:
    T_exp = solve_explicit(T_0, l, n_steps)
    T_exp_plot = np.clip(T_exp, -0.5, 1.1)
    T_imp = solve_implicit(T_0, l, n_steps)
    axes[0].plot(x, T_exp_plot, label=f'λ = {l:.2f}')
    axes[1].plot(x, T_imp, label=f'λ = {l:.2f}')

axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig('scratch.png')
print("Done")
