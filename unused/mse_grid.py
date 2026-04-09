# mse_grid.py
import io, sys
import numpy as np
from solver import solve

TRUE_ALPHAS = [0.003, 0.006, 0.012]
T_FINALS    = [5.0, 10.0, 20.0]
EPS         = 1e-5
MAX_STEPS   = 50
SUCCESS_MSE = 1e-6


def compute_mse(alpha, target, t_final):
    sys.stdout = io.StringIO()  # suppress solver prints
    _, T = solve(alpha=alpha, t_final=t_final)
    sys.stdout = sys.__stdout__
    return float(np.mean((T - np.array(target)) ** 2))


def compute_gradient(alpha, target, t_final):
    mse_plus  = compute_mse(alpha + EPS, target, t_final)
    mse_minus = compute_mse(alpha - EPS, target, t_final)
    return (mse_plus - mse_minus) / (2 * EPS)


def gradient_descent(true_alpha, t_final):
    sys.stdout = io.StringIO()
    _, target_T = solve(alpha=true_alpha, t_final=t_final)
    sys.stdout = sys.__stdout__
    target = target_T.tolist()

    # analytical initial guess from peak height
    peak  = max(target)
    sigma = 0.1
    alpha = float(np.clip(sigma**2 * (1/peak**2 - 1) / (2 * t_final), 0.001, 0.02))

    n_calls = 0
    final_mse = None

    for step in range(MAX_STEPS):
        mse  = compute_mse(alpha, target, t_final)
        grad = compute_gradient(alpha, target, t_final)
        n_calls += 3

        final_mse = mse
        if mse < SUCCESS_MSE:
            return alpha, mse, step + 1, n_calls

        if abs(grad) > 1e-10:
            lr = 0.1 * alpha / abs(grad)   # increased lr
        else:
            break

        alpha = float(np.clip(alpha - lr * grad, 0.001, 0.02))

    return alpha, final_mse, MAX_STEPS, n_calls


# ── run grid ──────────────────────────────────────────────────────────────────
results = {}
for true_alpha in TRUE_ALPHAS:
    for t_final in T_FINALS:
        found, mse, steps, calls = gradient_descent(true_alpha, t_final)
        results[(true_alpha, t_final)] = (found, mse, steps, calls)

# ── table 1: MSE ──────────────────────────────────────────────────────────────
print("=" * 60)
print("TABLE 1: FINAL MSE")
print("=" * 60)
print(f"{'':>10}  {'T=5s':>10}  {'T=10s':>10}  {'T=20s':>10}")
print("-" * 50)
for a in TRUE_ALPHAS:
    row = f"α={a:.3f}   "
    for t in T_FINALS:
        row += f"  {results[(a,t)][1]:>10.2e}"
    print(row)

# ── table 2: found alpha ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 2: FOUND ALPHA (error %)")
print("=" * 60)
print(f"{'':>10}  {'T=5s':>12}  {'T=10s':>12}  {'T=20s':>12}")
print("-" * 55)
for a in TRUE_ALPHAS:
    row = f"α={a:.3f}   "
    for t in T_FINALS:
        found = results[(a,t)][0]
        err   = abs(found - a) / a * 100
        row  += f"  {found:.4f}({err:.1f}%)"
    print(row)

# ── table 3: solver calls ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 3: SOLVER CALLS")
print("=" * 60)
print(f"{'':>10}  {'T=5s':>10}  {'T=10s':>10}  {'T=20s':>10}")
print("-" * 50)
for a in TRUE_ALPHAS:
    row = f"α={a:.3f}   "
    for t in T_FINALS:
        row += f"  {results[(a,t)][3]:>10}"
    print(row)

# ── summary ───────────────────────────────────────────────────────────────────
all_mses  = [v[1] for v in results.values()]
all_calls = [v[3] for v in results.values()]
successes = sum(1 for m in all_mses if m < SUCCESS_MSE)
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Success rate:      {successes}/{len(all_mses)}")
print(f"Avg MSE:           {np.mean(all_mses):.2e}")
print(f"Avg solver calls:  {np.mean(all_calls):.1f}")