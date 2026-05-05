import jax.numpy as jnp
import numpy as np
from implicit_solver import solve

N = 201
alpha_cfl = 0.01
l = 0.4
dx = 2 / (N - 1)
dt = l * dx**2 / alpha_cfl
t_final_c = 1.0

x = jnp.linspace(-1, 1, N)
T_0 = jnp.exp(-(x + 0.5)**2 / 0.001)[jnp.newaxis, :]

target_T = solve(T_0, 0.006, dt, t_final_c)[0]
pred_T = solve(T_0, 0.05, dt, t_final_c)[0]

sensor_indices = jnp.array([3 * N // 4])
print("MSE at 3*N//4:", np.mean((pred_T[sensor_indices] - target_T[sensor_indices])**2))
print("MSE entire domain:", np.mean((pred_T - target_T)**2))
