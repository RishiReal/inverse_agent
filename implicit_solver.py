import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

jax.config.update("jax_enable_x64", True)

def _build_tridiag(n, r):
    """Build the tridiagonal matrix for the implicit scheme."""
    main = (1 + 2 * r) * jnp.ones(n)
    off = -r * jnp.ones(n - 1)
    return jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)

@partial(jax.jit, static_argnames=("n",))
def _solve_inner(T_interior, tridag, n):
    """Solve the implicit scheme for one time step."""
    def _step(T_prev, _):
        return jnp.linalg.solve(tridag, T_prev), None
    
    T_final_interior, _ = lax.scan(_step, T_interior, None, length=n)
    return T_final_interior

def solve(T0, alpha, dt, t_final, domain_length=1.0):
    """Solve 1D heat equation using implicit scheme"""
    T0 = jnp.asarray(T0, dtype=jnp.float64)
    n = T0.shape[1]
    dx = domain_length / (n - 1)
    r = alpha * dt / dx**2
    num_time_steps  = int(round(t_final / dt))
    tridag = _build_tridiag(n - 2, r)

    T_interior = T0[0, 1:-1]
    T_final_interior = _solve_inner(T_interior, tridag, num_time_steps)
    T_final = jnp.concatenate([T0[0, :1], T_final_interior, T0[0, -1:]])
    return T_final[jnp.newaxis, :]

N, alpha, dt, t_final = 64, 0.01, 1e-2, 1.0
x = jnp.linspace(0, 1.0, N)
T0 = jnp.sin(jnp.pi * x)[jnp.newaxis, :]
 
T_f = solve(T0, alpha, dt, t_final)
T_exact = jnp.exp(-alpha * jnp.pi**2 * t_final) * jnp.sin(jnp.pi * x)
print(f"T0 shape : {T0.shape}")
print(f"T_f shape: {T_f.shape}")
print(f"L∞ error : {jnp.max(jnp.abs(T_f[0] - T_exact)):.2e}")
 
dL_dalpha = jax.grad(
    lambda a: jnp.sum(solve(T0, a, dt, t_final)**2)
)(alpha)
print(f"∂loss/∂alpha: {dL_dalpha:.4f}")
    
