# run_trials.py
import asyncio
import os
import json
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dotenv import load_dotenv

load_dotenv()

ALPHAS   = [0.05, 0.005, 0.006]
T_FINALS = [1.0, 5.0, 10.0]
N_TRIALS = 6
BASE_LOG_DIR = "logs_1_point_bisection_search"
SLEEP_BETWEEN_TRIALS  = 5
SLEEP_BETWEEN_CONFIGS = 20

API_KEYS = [os.getenv("GROQ_API_KEY")]

_key_index = 0

def current_api_key() -> str:
    return API_KEYS[_key_index % len(API_KEYS)]

def rotate_api_key(reason: str = "rate limit"):
    global _key_index
    old = _key_index % len(API_KEYS)
    _key_index += 1
    new = _key_index % len(API_KEYS)
    print(f"  🔄  {reason} — rotating API key {old+1} → {new+1}")


def trial_is_done(log_dir: str) -> bool:
    if not os.path.exists(log_dir):
        return False
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
    return len(log_files) > 0


def generate_gif(log_path: str, gif_path: str, true_alpha: float, t_final: float):
    with open(log_path) as f:
        data = json.load(f)

    steps = data["steps"]
    if not steps:
        return

    alphas    = [s["alpha"] for s in steps]
    mses      = [s["mse"]   for s in steps]
    step_nums = [s["step"]  for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Agent Search — true α={true_alpha}, t_final={t_final}", fontsize=12)

    def update(i):
        ax1.cla(); ax2.cla()

        ax1.plot(step_nums[:i+1], alphas[:i+1], "o-", color="steelblue")
        ax1.axhline(true_alpha, color="red", linestyle="--", label=f"true α={true_alpha}")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("α guess")
        ax1.set_title("Alpha Guesses")
        ax1.legend()
        ax1.set_xlim(0, max(step_nums) + 1)
        ax1.set_ylim(0, max(alphas) * 1.2 + 1e-8)
        ax1.annotate(f"step {step_nums[i]}\nα={alphas[i]:.4e}",
                     xy=(step_nums[i], alphas[i]),
                     xytext=(10, 10), textcoords="offset points", fontsize=8)

        ax2.plot(step_nums[:i+1], mses[:i+1], "o-", color="darkorange")
        ax2.axhline(1e-6, color="red", linestyle="--", label="threshold 1e-6")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("MSE")
        ax2.set_yscale("log")
        ax2.set_title("MSE Convergence")
        ax2.legend()
        ax2.set_xlim(0, max(step_nums) + 1)
        ax2.annotate(f"MSE={mses[i]:.2e}",
                     xy=(step_nums[i], mses[i]),
                     xytext=(10, -15), textcoords="offset points", fontsize=8)

    ani = FuncAnimation(fig, update, frames=len(steps), repeat=False)
    ani.save(gif_path, writer=PillowWriter(fps=2))
    plt.close(fig)
    print(f"  GIF saved → {gif_path}")


async def run_single_trial(alpha: float, t_final: float, trial_num: int):
    log_dir = os.path.join(BASE_LOG_DIR, f"alpha_{alpha}", f"t_{t_final}", f"trial_{trial_num}")

    if trial_is_done(log_dir):
        print(f"  ⏭️  Skipping α={alpha} t={t_final} trial={trial_num} — already complete")
        return

    os.makedirs(log_dir, exist_ok=True)

    max_key_attempts = len(API_KEYS)

    for attempt in range(max_key_attempts):
        env = {
            **os.environ,
            "HEAT_TRUE_ALPHA": str(alpha),
            "HEAT_T_FINAL":    str(t_final),
            "LOGS_DIR":        log_dir,
            "GROQ_API_KEY":    current_api_key(),
        }

        print(f"\n{'='*60}")
        print(f"  α={alpha}  t_final={t_final}  trial={trial_num}  key={(_key_index % len(API_KEYS)) + 1}/{len(API_KEYS)}")
        print(f"{'='*60}")

        # capture_output MUST be False — capturing stdout breaks the
        # stdio pipe between mcp_client.py and mcp_server.py
        result = subprocess.run(
            ["python", "mcp_client.py"],
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
        )

        # if no log was written, the trial failed — rotate key and retry
        if not trial_is_done(log_dir):
            print(f"  ⚠️  No log written (returncode={result.returncode}) — rotating key and retrying...")
            rotate_api_key()
            await asyncio.sleep(5)
            continue

        break  # log exists, trial completed

    else:
        print(f"  ❌  All {len(API_KEYS)} keys exhausted for trial α={alpha} t={t_final} #{trial_num}.")
        return

    # generate GIF from the log that was just written
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
    if log_files:
        log_path = os.path.join(log_dir, log_files[-1])
        gif_path = os.path.join(log_dir, "convergence.gif")
        generate_gif(log_path, gif_path, true_alpha=alpha, t_final=t_final)

    print(f"  Waiting {SLEEP_BETWEEN_TRIALS}s before next trial...")
    await asyncio.sleep(SLEEP_BETWEEN_TRIALS)


async def main():
    if not API_KEYS:
        raise ValueError("No API keys configured.")
    print(f"  🔑  Loaded {len(API_KEYS)} API key(s).")

    done    = 0
    skipped = 0

    for alpha in ALPHAS:
        for t_final in T_FINALS:
            for trial in range(1, N_TRIALS + 1):
                log_dir = os.path.join(BASE_LOG_DIR, f"alpha_{alpha}", f"t_{t_final}", f"trial_{trial}")
                if trial_is_done(log_dir):
                    skipped += 1
                else:
                    await run_single_trial(alpha, t_final, trial)
                    done += 1

            print(f"\n  Config (α={alpha}, t={t_final}) done — waiting {SLEEP_BETWEEN_CONFIGS}s...\n")
            await asyncio.sleep(SLEEP_BETWEEN_CONFIGS)

    print(f"\n✅  All trials complete. Ran {done} new, skipped {skipped} already-done.")
    print(f"Logs saved under: {BASE_LOG_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())