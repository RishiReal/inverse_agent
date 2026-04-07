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

@partial(jax.jit, static_argnames=("num_time_steps",))
def _solve_inner(T_interior, tridag, r, T_left, T_right, num_time_steps):
    """Solve the implicit scheme for one time step."""
    def _step(T, _):
        b = T.at[0].add(r * T_left)
        b = b.at[-1].add(r * T_right)
        return jnp.linalg.solve(tridag, b), None
    
    T_final, _ = lax.scan(_step, T_interior, None, length=num_time_steps)
    return T_final

def solve(T0, alpha, dt, t_final, domain_length=2.0):
    """Solve 1D heat equation using implicit scheme"""
    T0 = jnp.asarray(T0, dtype=jnp.float64)
    n = T0.shape[1]
    dx = domain_length / (n - 1)
    r = alpha * dt / dx**2
    num_time_steps  = int(round(t_final / dt))
    tridag = _build_tridiag(n - 2, r)

    T_interior = T0[0, 1:-1]
    T_final_interior = _solve_inner(T_interior, tridag, r, T0[0, 0], T0[0, -1], num_time_steps)
    T_final = jnp.concatenate([T0[0, :1], T_final_interior, T0[0, -1:]])
    return T_final[jnp.newaxis, :]
    
