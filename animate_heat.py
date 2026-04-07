"""
animate_heat.py
---------------
Saves a GIF of the implicit heat solver evolving over time,
with ∂T/∂alpha overlaid to show sensitivity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from implicit_solver import solve   # your JAX solver

jax.config.update("jax_enable_x64", True)

# ── Setup (match your existing params) ───────────────────────────────
N     = 201
alpha = 0.01
l     = 0.4
dx    = 2 / (N - 1)
dt    = l * dx**2 / alpha
n_steps = N

x     = jnp.linspace(-1, 1, N)
T_0   = jnp.exp(-(x + 0.5)**2 / 0.001)[jnp.newaxis, :]  # (1, N)

# ── Collect snapshots at different t values ───────────────────────────
n_frames   = 40
t_values   = np.linspace(dt, n_steps * dt, n_frames)
snapshots  = []
grads      = []

grad_fn = jax.grad(lambda a, t: jnp.sum(solve(T_0, a, dt, t)**2), argnums=0)

print("Computing frames...")
for i, t in enumerate(t_values):
    T_f   = solve(T_0, alpha, dt, float(t))
    dTda  = jax.jacobian(lambda a: solve(T_0, a, dt, float(t))[0])(alpha)  # (N,)
    snapshots.append(np.array(T_f[0]))
    grads.append(np.array(dTda))
    if i % 8 == 0:
        print(f"  frame {i+1}/{n_frames}, t={t:.3f}")

x_np = np.array(x)

# ── Plot ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), facecolor="#0f1117")
fig.subplots_adjust(hspace=0.4)

for ax in (ax1, ax2):
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#8892a4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2330")

# Temperature plot
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.1, 1.1)
ax1.set_ylabel("T(x, t)", color="#8892a4", fontsize=11)
ax1.set_title("Implicit Heat Solver — Gaussian Pulse Diffusion", color="#e2e8f0", fontsize=12, pad=10)
ax1.axhline(0, color="#1f2330", lw=1)
ax1.grid(True, color="#1a1e2a", lw=0.5)

line_T,   = ax1.plot([], [], color="#00e5a0", lw=2, label="T(x,t)")
line_T0,  = ax1.plot(x_np, snapshots[0], color="#ffffff", lw=1,
                     alpha=0.2, linestyle="--", label="T₀")
time_text = ax1.text(0.02, 0.92, "", transform=ax1.transAxes,
                     color="#e2e8f0", fontsize=10, fontfamily="monospace")
ax1.legend(loc="upper right", facecolor="#13151a", edgecolor="#1f2330",
           labelcolor="#8892a4", fontsize=9)

# Sensitivity plot
ax2.set_xlim(-1, 1)
ax2.set_ylabel("∂T/∂α", color="#8892a4", fontsize=11)
ax2.set_xlabel("x", color="#8892a4", fontsize=11)
ax2.set_title("Sensitivity to α  (where does α matter most?)", color="#e2e8f0", fontsize=11, pad=8)
ax2.axhline(0, color="#1f2330", lw=1)
ax2.grid(True, color="#1a1e2a", lw=0.5)

all_grads = np.concatenate(grads)
glim = max(abs(all_grads.min()), abs(all_grads.max())) * 1.1
ax2.set_ylim(-glim, glim)

line_g, = ax2.plot([], [], color="#ff6b35", lw=2, label="∂T/∂α")
fill_pos = ax2.fill_between(x_np, 0, 0, alpha=0.15, color="#ff6b35")
ax2.legend(loc="upper right", facecolor="#13151a", edgecolor="#1f2330",
           labelcolor="#8892a4", fontsize=9)

def init():
    line_T.set_data([], [])
    line_g.set_data([], [])
    return line_T, line_g, time_text

def update(i):
    global fill_pos
    line_T.set_data(x_np, snapshots[i])
    line_g.set_data(x_np, grads[i])
    time_text.set_text(f"t = {t_values[i]:.3f}")

    # Update fill
    fill_pos.remove()
    fill_pos = ax2.fill_between(x_np, 0, grads[i], alpha=0.15, color="#ff6b35")

    return line_T, line_g, time_text

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=80, blit=False
)

out = "heat_diffusion.gif"
ani.save(out, writer="pillow", fps=12, dpi=120)
print(f"\nSaved → {out}")
plt.close()