import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

TRUE_ALPHA = 0.006

runs = [
    {"t": 1.0,  "alphas": (0.005, 0.004, 0.003, 0.006),          "exact": True},
    {"t": 1.0,  "alphas": (0.005, 0.004, 0.003, 0.006),          "exact": True},
    {"t": 1.0,  "alphas": (0.0055, 0.0056, 0.0057, 0.0058),      "exact": False},
    {"t": 1.0,  "alphas": (0.0055, 0.0056, 0.0057, 0.0058),      "exact": False},
    {"t": 1.0,  "alphas": (0.005, 0.004, 0.003, 0.006),          "exact": True},

    {"t": 5.0,  "alphas": (0.005, 0.004, 0.006),                  "exact": True},
    {"t": 5.0,  "alphas": (0.0055, 0.005, 0.0045, 0.006),         "exact": True},
    {"t": 5.0,  "alphas": (0.005, 0.004, 0.006),                  "exact": True},
    {"t": 5.0,  "alphas": (0.005, 0.004, 0.006),                  "exact": True},
    {"t": 5.0,  "alphas": (0.005, 0.004, 0.006),                  "exact": True},

    {"t": 10.0, "alphas": (0.0055, 0.0056, 0.0057),               "exact": False},
    {"t": 10.0, "alphas": (0.0055, 0.0056, 0.0057),               "exact": False},
    {"t": 10.0, "alphas": (0.005, 0.004, 0.006),                  "exact": True},
    {"t": 10.0, "alphas": (0.005, 0.004, 0.006),                  "exact": True},
    {"t": 10.0, "alphas": (0.005, 0.006),                         "exact": True},
]

CONFIG = {
    1.0:  {"color": "#378ADD", "label": "t_final = 1.0"},
    5.0:  {"color": "#1D9E75", "label": "t_final = 5.0"},
    10.0: {"color": "#D85A30", "label": "t_final = 10.0"},
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
fig.suptitle("α estimate per step  ·  dashed line = true α = 0.006", fontsize=12)

for ax, t_val in zip(axes, [1.0, 5.0, 10.0]):
    color = CONFIG[t_val]["color"]
    t_runs = [r for r in runs if r["t"] == t_val]

    # count how many times each unique trajectory appears
    counts = Counter(r["alphas"] for r in t_runs)
    exact_map = {r["alphas"]: r["exact"] for r in t_runs}

    for alphas, count in counts.items():
        xs = list(range(1, len(alphas) + 1))
        lw = 1.8 + count * 0.5  # thicker line = more trials
        ax.plot(xs, alphas, color=color, linewidth=lw, alpha=0.7)

        # endpoint marker
        ax.scatter(xs[-1], alphas[-1],
                   marker="o" if exact_map[alphas] else "X",
                   color=color, s=70, zorder=5,
                   edgecolors="white", linewidths=0.8)

        # count label just above the midpoint of the line
        mid = len(xs) // 2
        ax.text(xs[mid], alphas[mid], f"×{count}",
                fontsize=9, color=color, fontweight="bold",
                ha="center", va="bottom")

    ax.axhline(TRUE_ALPHA, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.set_title(CONFIG[t_val]["label"], fontsize=11, color=color)
    ax.set_xlabel("step #")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(True, linestyle="--", alpha=0.25)

axes[0].set_ylabel("α estimate")

plt.tight_layout()
plt.savefig("convergence.png", dpi=150, bbox_inches="tight")
plt.show()