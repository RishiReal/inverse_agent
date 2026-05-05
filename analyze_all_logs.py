"""
Comprehensive analysis script for both full-domain and 1-sensor inverse agent logs.
Outputs structured summary data to stdout for report generation.
"""
import os
import json
import glob
import collections

def load_logs(log_dir):
    """Load all JSON log files from a directory."""
    logs = []
    if not os.path.isdir(log_dir):
        return logs
    for f in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        try:
            with open(f) as fh:
                data = json.load(fh)
                data["_file"] = os.path.basename(f)
                logs.append(data)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
    return logs

def analyze_group(logs, label):
    """Analyze a group of logs (same alpha, same t_final)."""
    if not logs:
        return None

    total = len(logs)
    successes = [l for l in logs if l.get("success")]
    failures = [l for l in logs if not l.get("success")]
    
    n_steps_list = [l.get("n_steps", 0) for l in logs]
    final_alphas = [l.get("final_alpha", None) for l in logs]
    
    # Extract final MSE from each log
    final_mses = []
    for l in logs:
        steps = l.get("steps", [])
        if steps:
            final_mses.append(steps[-1].get("mse", float('inf')))
        else:
            final_mses.append(float('inf'))
    
    # Extract min MSE achieved
    min_mses = []
    for l in logs:
        steps = l.get("steps", [])
        if steps:
            min_mses.append(min(s.get("mse", float('inf')) for s in steps))
        else:
            min_mses.append(float('inf'))

    return {
        "label": label,
        "total_trials": total,
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": len(successes) / total * 100 if total > 0 else 0,
        "avg_steps": sum(n_steps_list) / len(n_steps_list) if n_steps_list else 0,
        "min_steps": min(n_steps_list) if n_steps_list else 0,
        "max_steps": max(n_steps_list) if n_steps_list else 0,
        "final_alphas": final_alphas,
        "final_mses": final_mses,
        "min_mses": min_mses,
        "avg_final_mse": sum(final_mses) / len(final_mses) if final_mses else 0,
        "avg_min_mse": sum(min_mses) / len(min_mses) if min_mses else 0,
        "success_steps": [l.get("n_steps", 0) for l in successes] if successes else [],
    }

def main():
    alphas = ["0.05", "0.005", "0.006"]
    t_finals = ["1.0", "5.0", "10.0"]
    
    prefixes = {
        "full_domain": "logs_alpha",
        "1sensor": "logs_1sensor_alpha",
    }
    
    all_results = {}
    
    for prefix_label, prefix in prefixes.items():
        all_results[prefix_label] = {}
        print(f"\n{'='*70}")
        print(f"  {prefix_label.upper()} MSE ANALYSIS")
        print(f"{'='*70}")
        
        for alpha in alphas:
            log_dir = f"{prefix}_{alpha}"
            logs = load_logs(log_dir)
            
            # Group by t_final
            by_tfinal = collections.defaultdict(list)
            for l in logs:
                tf = str(l.get("t_final", "unknown"))
                by_tfinal[tf].append(l)
            
            for tf in t_finals:
                group_logs = by_tfinal.get(tf, [])
                label = f"α={alpha}, t_final={tf}"
                result = analyze_group(group_logs, label)
                if result:
                    all_results[prefix_label][(alpha, tf)] = result
                    
                    print(f"\n  {label}")
                    print(f"    Trials: {result['total_trials']}")
                    print(f"    Success: {result['successes']}/{result['total_trials']} ({result['success_rate']:.0f}%)")
                    print(f"    Avg Steps: {result['avg_steps']:.1f} (range: {result['min_steps']}-{result['max_steps']})")
                    print(f"    Avg Final MSE: {result['avg_final_mse']:.2e}")
                    print(f"    Avg Min MSE:   {result['avg_min_mse']:.2e}")
                    if result['success_steps']:
                        print(f"    Success Steps: {result['success_steps']}")
                    print(f"    Final Alphas: {[f'{a:.6g}' for a in result['final_alphas']]}")
    
    # Comparison table
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON: FULL-DOMAIN vs 1-SENSOR")
    print(f"{'='*70}")
    print(f"\n  {'Config':<25} {'Full-Domain SR':<18} {'1-Sensor SR':<18} {'FD Avg Min MSE':<18} {'1S Avg Min MSE':<18}")
    print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*18} {'-'*18}")
    
    for alpha in alphas:
        for tf in t_finals:
            key = (alpha, tf)
            fd = all_results["full_domain"].get(key)
            s1 = all_results["1sensor"].get(key)
            
            label = f"α={alpha}, t={tf}"
            fd_sr = f"{fd['success_rate']:.0f}%" if fd else "N/A"
            s1_sr = f"{s1['success_rate']:.0f}%" if s1 else "N/A"
            fd_mse = f"{fd['avg_min_mse']:.2e}" if fd else "N/A"
            s1_mse = f"{s1['avg_min_mse']:.2e}" if s1 else "N/A"
            
            print(f"  {label:<25} {fd_sr:<18} {s1_sr:<18} {fd_mse:<18} {s1_mse:<18}")

if __name__ == "__main__":
    main()
