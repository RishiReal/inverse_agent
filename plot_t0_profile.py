import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Same parameters as mcp_server.py
N = 201
x = jnp.linspace(-1, 1, N)
T_0 = jnp.exp(-(x + 0.5)**2 / 0.001)

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot initial profile
plt.plot(x, T_0, label="Initial Temperature (t=0)", color="blue", linewidth=2)

# Mark the sensor location
sensor_idx = N // 4
sensor_x = x[sensor_idx]
sensor_T = T_0[sensor_idx]

plt.plot(sensor_x, sensor_T, 'ro', markersize=10, label=f"Sensor Location (x={sensor_x:.2f})")

plt.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5, label="Pulse Center (x=-0.5)")

# Adding some labels and grid
plt.title("Initial Temperature Profile (t=0) and Sensor Location", fontsize=14)
plt.xlabel("Spatial Coordinate (x)", fontsize=12)
plt.ylabel("Temperature (T)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=11)

# Save the plot
plt.tight_layout()
plt.savefig("/Users/rishi/research/inverse_agent/figures/initial_profile_t0_new.png", dpi=150)
print("Saved figure to /Users/rishi/research/inverse_agent/figures/initial_profile_t0_new.png")
