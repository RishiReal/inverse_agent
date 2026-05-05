"""
Plot convergence figures and print analysis for the new 1-sensor logs
with sensor at X=-0.06 (N//5) for alpha 0.05 and 0.005,
and sensor at X=0.06 for alpha 0.006.
"""
import os
import json
import glob
import collections
import matplotlib.pyplot as plt

LOG_DIRS = {
    "logs_0.05_X=-0.06":  {"alpha": "0.05",  "sensor": "x ≈ -0.06 (N//5)"},
    "logs_0.005_X=-0.06": {"alpha": "0.005", "sensor": "x ≈ -0.06 (N//5)"},
    "logs_0.006_X=0.06":  {"alpha": "0.006", "sensor": "x ≈ 0.06 (N//5)"},
}

os.makedirs("figures", exist_ok=True)

for log_dir_name, meta in LOG_DIRS.items():
    target_alpha = meta["alpha"]
    sensor_label = meta["sensor"]

    if not os.path.isdir(log_dir_name):
        print(f"[skip] {log_dir_name} not found")
        continue

    # Load all logs
    logs = []
    for f in sorted(glob.glob(os.path.join(log_dir_name, "*.json"))):
        try:
            with open(f) as fh:
                data = json.load(fh)
                data["_file"] = os.path.basename(f)
                logs.append(data)
        except Exception as e:
            print(f"  [skip] {f}: {e}")

    # Group by t_final
    by_tfinal = collections.defaultdict(list)
    for l in logs:
        tf = str(l.get("t_final", "unknown"))
        by_tfinal[tf].append(l)

    print(f"\n{'='*65}")
    print(f"  {log_dir_name}  (target α={target_alpha}, sensor at {sensor_label})")
    print(f"{'='*65}")

    for tf in sorted(by_tfinal.keys(), key=float):
        group = by_tfinal[tf]
        total = len(group)
        successes = [l for l in group if l.get("success")]
        n_steps_list = [l.get("n_steps", 0) for l in group]

        final_mses = []
        min_mses = []
        for l in group:
            steps = l.get("steps", [])
            if steps:
                final_mses.append(steps[-1].get("mse", float('inf')))
                min_mses.append(min(s.get("mse", float('inf')) for s in steps))

        sr = len(successes) / total * 100 if total else 0
        avg_steps = sum(n_steps_list) / len(n_steps_list) if n_steps_list else 0
        avg_min_mse = sum(min_mses) / len(min_mses) if min_mses else 0
        avg_final_mse = sum(final_mses) / len(final_mses) if final_mses else 0

        print(f"\n  t_final={tf}")
        print(f"    Trials: {total}  |  Success: {len(successes)}/{total} ({sr:.0f}%)")
        print(f"    Avg Steps: {avg_steps:.1f} (range: {min(n_steps_list)}-{max(n_steps_list)})")
        print(f"    Avg Final MSE: {avg_final_mse:.2e}  |  Avg Min MSE: {avg_min_mse:.2e}")
        final_alphas = [f"{l.get('final_alpha', 0):.6g}" for l in group]
        print(f"    Final Alphas: {final_alphas}")

        # ---- Plot ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Agent Convergence — α={target_alpha}, t_final={tf}\n"
            f"(1-sensor at {sensor_label})",
            fontsize=14, fontweight='bold', y=1.05
        )

        cmap = plt.get_cmap('tab10')

        # Deduplicate identical trajectories
        unique_trajectories = collections.defaultdict(list)
        for idx, l in enumerate(group):
            steps = l.get("steps", [])
            if not steps:
                continue
            traj_key = tuple((s["step"], s["mse"], s["alpha"]) for s in steps)
            unique_trajectories[traj_key].append(idx + 1)

        colors = [cmap(i % 10) for i in range(len(unique_trajectories))]

        for ci, (traj_key, attempt_nums) in enumerate(unique_trajectories.items()):
            step_nums = [pt[0] for pt in traj_key]
            mses = [pt[1] for pt in traj_key]
            alpha_guesses = [pt[2] for pt in traj_key]
            color = colors[ci]
            label = f"Attempt(s) {', '.join(map(str, attempt_nums))}"

            ax1.plot(step_nums, mses, color=color, linewidth=2, marker='o', label=label, alpha=0.8)
            ax2.plot(step_nums, alpha_guesses, color=color, linewidth=2, marker='o', label=label, alpha=0.8)

        ax2.axhline(y=float(target_alpha), color="black", linestyle="--", linewidth=2.5,
                    label=f"True Alpha ({target_alpha})")
        ax1.axhline(y=1e-6, color="black", linestyle="--", linewidth=2.5,
                    label="Target MSE (1e-6)")

        ax1.set_xlabel("Agent Step", fontsize=12)
        ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
        ax1.set_title("Error Minimization", fontsize=13)
        ax1.grid(True, linestyle=":", alpha=0.7)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=9)

        ax2.set_xlabel("Agent Step", fontsize=12)
        ax2.set_ylabel("Guessed Alpha", fontsize=12)
        ax2.set_title("Alpha Search Trajectory", fontsize=13)
        ax2.grid(True, linestyle=":", alpha=0.7)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.45, 1.0), fontsize=9)

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        safe_dir = log_dir_name.replace("=", "").replace(" ", "_")
        out_path = f"figures/convergence_{safe_dir}_tfinal_{tf}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    → Saved: {out_path}")

print("\nDone!")
