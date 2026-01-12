#!/usr/bin/env python
"""
Visualize matmul benchmark speedup comparison.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
data = {
    "sizes": [
        "(4096, 1024, 1024)",
        "(1024, 4096, 1024)",
        "(4096, 2048, 2048)",
        "(2048, 4096, 2048)",
        "(8192, 1024, 1024)",
        "(1024, 8192, 1024)",
        "(8192, 2048, 2048)",
        "(2048, 8192, 2048)",
    ],
    "triton_tutorial": [1.04631, 1.03845, 0.840435, 0.883234, 0.886638, 1.07035, 0.783258, 0.818707],
    "matmul_partition_k": [0.0301898, 0.0254657, 0.0366286, 0.0307271, 0.0216396, 0.016944, 0.0326374, 0.019996],
    "aten_tunableop": [1.26454, 1.0111, 1.0301, 0.964566, 0.996127, 1.03367, 0.957606, 0.99756],
    "pt2_triton": [1.13679, 1.08928, 0.486921, 0.632348, 0.57634, 1.01977, 0.825569, 0.736276],
    "helion": [1.06953, 1.08969, 0.814427, 0.928959, 0.92504, 1.1414, 0.85886, 0.916783],
}

averages = {
    "triton_tutorial": 0.920922,
    "matmul_partition_k": 0.0267785,
    "aten_tunableop": 1.03191,
    "pt2_triton": 0.812911,
    "helion": 0.968085,
}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Matmul Benchmark: Speedup vs Baseline", fontsize=16, fontweight="bold")

# Colors for each implementation
colors = {
    "triton_tutorial": "#1f77b4",
    "aten_tunableop": "#2ca02c",
    "pt2_triton": "#ff7f0e",
    "helion": "#d62728",
}

# Plot 1: Grouped bar chart (excluding partition_k which is too small)
ax1 = axes[0, 0]
x = np.arange(len(data["sizes"]))
width = 0.2
implementations = ["triton_tutorial", "aten_tunableop", "pt2_triton", "helion"]
labels = ["Triton Tutorial", "ATen TunableOp", "PT2 Triton", "Helion"]

for i, (impl, label) in enumerate(zip(implementations, labels)):
    offset = (i - 1.5) * width
    bars = ax1.bar(x + offset, data[impl], width, label=label, color=colors[impl], alpha=0.8)

ax1.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Baseline (1.0x)")
ax1.set_xlabel("Matrix Size (M, N, K)")
ax1.set_ylabel("Speedup")
ax1.set_title("Speedup by Matrix Size")
ax1.set_xticks(x)
ax1.set_xticklabels([s.replace("(", "").replace(")", "") for s in data["sizes"]], rotation=45, ha="right", fontsize=8)
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, 1.4)

# Plot 2: Line chart showing trends
ax2 = axes[0, 1]
for impl, label in zip(implementations, labels):
    ax2.plot(range(len(data["sizes"])), data[impl], marker="o", label=label, color=colors[impl], linewidth=2, markersize=8)

ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Baseline")
ax2.set_xlabel("Matrix Size Index")
ax2.set_ylabel("Speedup")
ax2.set_title("Speedup Trend Across Sizes")
ax2.set_xticks(range(len(data["sizes"])))
ax2.set_xticklabels([f"{i}" for i in range(len(data["sizes"]))])
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

# Add size annotations at top
for i, size in enumerate(data["sizes"]):
    short = size.replace("(", "").replace(")", "").replace(" ", "")
    ax2.annotate(short, (i, 1.35), fontsize=6, ha="center", rotation=45)

# Plot 3: Average speedup bar chart
ax3 = axes[1, 0]
impl_names = ["Triton\nTutorial", "ATen\nTunableOp", "PT2\nTriton", "Helion"]
avg_values = [averages["triton_tutorial"], averages["aten_tunableop"], averages["pt2_triton"], averages["helion"]]
bar_colors = [colors["triton_tutorial"], colors["aten_tunableop"], colors["pt2_triton"], colors["helion"]]

bars = ax3.bar(impl_names, avg_values, color=bar_colors, alpha=0.8, edgecolor="black")
ax3.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Baseline (1.0x)")

# Add value labels on bars
for bar, val in zip(bars, avg_values):
    height = bar.get_height()
    ax3.annotate(f"{val:.3f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontweight="bold")

ax3.set_ylabel("Average Speedup")
ax3.set_title("Average Speedup Across All Sizes")
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_ylim(0, 1.3)

# Plot 4: Heatmap
ax4 = axes[1, 1]
heatmap_data = np.array([data[impl] for impl in implementations])
im = ax4.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.3)

ax4.set_xticks(range(len(data["sizes"])))
ax4.set_xticklabels([s.replace("(", "").replace(")", "").replace(" ", "") for s in data["sizes"]], rotation=45, ha="right", fontsize=8)
ax4.set_yticks(range(len(implementations)))
ax4.set_yticklabels(labels)
ax4.set_title("Speedup Heatmap (Green=Fast, Red=Slow)")

# Add text annotations
for i in range(len(implementations)):
    for j in range(len(data["sizes"])):
        val = heatmap_data[i, j]
        color = "white" if val < 0.7 or val > 1.1 else "black"
        ax4.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

plt.colorbar(im, ax=ax4, label="Speedup")

plt.tight_layout()

# Save
output_path = "/home/manoj/software/helion/openevolve_benchmark_artifacts/matmul_benchmark.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Visualization saved to: {output_path}")

# Also print summary
print("\n" + "="*80)
print("SUMMARY: Average Speedup vs Baseline")
print("="*80)
print(f"{'Implementation':<25} {'Avg Speedup':<15} {'Status'}")
print("-"*80)
for impl, label in zip(implementations, labels):
    avg = averages[impl]
    status = "FASTER" if avg > 1.0 else "SLOWER"
    emoji = "+" if avg > 1.0 else "-"
    print(f"{label:<25} {avg:<15.4f} {emoji} {abs(1-avg)*100:.1f}% {status}")
print("="*80)
