import os
import json
import glob
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_alpha_logs():
    alphas = ["0.05", "0.005", "0.006"]
    os.makedirs("figures", exist_ok=True)
    
    # Group logs by (target_alpha, t_final)
    grouped_logs = collections.defaultdict(list)
    
    for target_alpha in alphas:
        log_dir = f"logs_alpha_{target_alpha}"
        if not os.path.isdir(log_dir):
            continue
            
        log_files = glob.glob(os.path.join(log_dir, "*.json"))
        log_files.sort()
        
        for log_file in log_files:
            with open(log_file, "r") as f:
                try:
                    data = json.load(f)
                    t_final = data.get("t_final", "unknown")
                    grouped_logs[(target_alpha, t_final)].append((log_file, data))
                except Exception as e:
                    print(f"Error reading {log_file}: {e}")
    
    out_paths = []
    
    for (target_alpha, t_final), logs in grouped_logs.items():
        if not logs:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Agent Convergence (Target Alpha = {target_alpha}, t_final = {t_final})", fontsize=16, fontweight='bold', y=1.05)
        
        # Use colormap to generate a unique color for each attempt
        cmap = cm.get_cmap('tab10')
        
        # Group identical trajectories
        unique_trajectories = collections.defaultdict(list)
        for idx, (log_file, data) in enumerate(logs):
            steps = data.get("steps", [])
            if not steps:
                continue
            traj_key = tuple((s["step"], s["mse"], s["alpha"]) for s in steps)
            unique_trajectories[traj_key].append(idx + 1)
            
        colors = [cmap(i % 10) for i in range(len(unique_trajectories))]
        
        for color_idx, (traj_key, attempt_nums) in enumerate(unique_trajectories.items()):
            step_nums = [pt[0] for pt in traj_key]
            mses = [pt[1] for pt in traj_key]
            alpha_guesses = [pt[2] for pt in traj_key]
            
            color = colors[color_idx]
            
            # Create a label grouping all attempts that took this exact path
            attempt_str = ", ".join(map(str, attempt_nums))
            label = f"Attempt(s) {attempt_str}"
            
            ax1.plot(step_nums, mses, color=color, linewidth=2, marker='o', label=label, alpha=0.8)
            ax2.plot(step_nums, alpha_guesses, color=color, linewidth=2, marker='o', label=label, alpha=0.8)
        
        # True alpha reference line
        ax2.axhline(y=float(target_alpha), color="black", linestyle="--", linewidth=2.5, 
                    label=f"True Alpha ({target_alpha})")
        # Convergence threshold reference
        ax1.axhline(y=1e-6, color="black", linestyle="--", linewidth=2.5, 
                    label="Target MSE (1e-6)")

        ax1.set_xlabel("Agent Step", fontsize=12)
        ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
        ax1.set_yscale("log")
        ax1.set_title("Error Minimization", fontsize=14)
        ax1.grid(True, linestyle=":", alpha=0.7)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
        
        ax2.set_xlabel("Agent Step", fontsize=12)
        ax2.set_ylabel("Guessed Alpha", fontsize=12)
        ax2.set_title("Alpha Search Trajectory", fontsize=14)
        ax2.set_yscale("log")
        ax2.grid(True, linestyle=":", alpha=0.7)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.45, 1.0))
        
        # Adjust layout to account for outside legends
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        out_path = f"figures/convergence_alpha_{target_alpha}_tfinal_{t_final}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        out_paths.append(out_path)
        print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_alpha_logs()
