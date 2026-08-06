import matplotlib.pyplot as plt

# Empirical dataset: (Estimated Proving Time in Seconds, Test Accuracy %)
data = {
    # Model H (Ultra-Light MLP - 13.38s)
    "Model H (16-bit)": (13.38, 96.18), "Model H (12-bit)": (13.38, 96.14),
    "Model H (8-bit)":  (13.38, 93.62), "Model H (6-bit)":  (13.38, 71.39), "Model H (4-bit)":  (13.38, 14.16),

    # Model E (Deep MLP - 19.46s)
    "Model E (16-bit)": (19.46, 97.22), "Model E (12-bit)": (19.46, 97.22),
    "Model E (8-bit)":  (19.46, 94.62), "Model E (6-bit)":  (19.46, 74.45), "Model E (4-bit)":  (19.46, 10.14),

    # Model F (GELU MLP - Lookup Table Dependent)
    "Model F (16-bit)": (91.12, 97.44), "Model F (12-bit)": (31.12, 97.50),
    "Model F (8-bit)":  (27.37, 96.34), "Model F (6-bit)":  (27.18, 77.35), "Model F (4-bit)":  (27.14, 20.83),

    # Model C (Sigmoid MLP - Lookup Table Dependent)
    "Model C (16-bit)": (94.83, 97.08), "Model C (12-bit)": (34.83, 97.10),
    "Model C (8-bit)":  (31.08, 96.29), "Model C (6-bit)":  (30.89, 93.25), "Model C (4-bit)":  (30.84, 60.26),

    # Model A (Wide MLP - 41.67s)
    "Model A (16-bit)": (41.67, 97.16), "Model A (12-bit)": (41.67, 96.54),
    "Model A (8-bit)":  (41.67, 96.51), "Model A (6-bit)":  (41.67, 63.25), "Model A (4-bit)":  (41.67, 19.62),

    # Model D (Tiny CNN - 101.70s)
    "Model D (16-bit)": (101.70, 97.73), "Model D (12-bit)": (101.70, 97.53),
    "Model D (8-bit)":  (101.70, 91.50), "Model D (6-bit)":  (101.70, 47.10), "Model D (4-bit)":  (101.70, 16.73),

    # Model G (Medium CNN - 194.38s)
    "Model G (16-bit)": (194.38, 98.02), "Model G (12-bit)": (194.38, 98.03),
    "Model G (8-bit)":  (194.38, 92.81), "Model G (6-bit)":  (194.38, 64.52), "Model G (4-bit)":  (194.38, 35.42),

    # Model B (Heavy CNN - 410.11s)
    "Model B (16-bit)": (410.11, 97.83), "Model B (12-bit)": (410.11, 97.71),
    "Model B (8-bit)":  (410.11, 89.85), "Model B (6-bit)":  (410.11, 68.48), "Model B (4-bit)":  (410.11, 32.10)
}

labels = list(data.keys())
times = [data[label][0] for label in labels]
accuracies = [data[label][1] for label in labels]

# Calculate Empirical Pareto Frontier
sorted_points = sorted(zip(times, accuracies, labels), key=lambda x: x[0])
pareto_front_x = []
pareto_front_y = []
pareto_front_labels = []
max_accuracy_so_far = -1

for time, acc, label in sorted_points:
    if acc > max_accuracy_so_far:
        pareto_front_x.append(time)
        pareto_front_y.append(acc)
        pareto_front_labels.append(label)
        max_accuracy_so_far = acc

# Create Plot
plt.figure(figsize=(14, 9))

# Plot all 40 points
plt.scatter(times, accuracies, color='steelblue', s=50, alpha=0.5, label='Tested Configurations (40)')

# Highlight Pareto Optimal Points
plt.scatter(pareto_front_x, pareto_front_y, color='darkblue', s=130, edgecolors='black', zorder=5)
plt.plot(pareto_front_x, pareto_front_y, color='red', linestyle='--', linewidth=2.5, zorder=4, label='Empirical Pareto Frontier')

# Custom label
label_offsets = {
    "Model H (16-bit)": (12, -15),
    "Model E (16-bit)": (-50, 15),
    "Model F (12-bit)": (12, -15),
    "Model D (16-bit)": (12, -18),
    "Model G (16-bit)": (12, -15)
}

# Annotate Pareto Frontier points
for time, acc, label in zip(pareto_front_x, pareto_front_y, pareto_front_labels):
    offset = label_offsets.get(label, (10, -5))
    plt.annotate(
        label,
        (time, acc),
        textcoords="offset points",
        xytext=offset,
        ha='left',
        fontsize=9,
        fontweight='bold',
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5) if "Model E" in label else None
    )

# Formatting
plt.title("Empirical ZKML Pareto Frontier: Real Quantized Accuracy vs. Proving Time", fontsize=15, pad=15)
plt.xlabel("Predicted Proving Time (Seconds) - Lower is Better", fontsize=12)
plt.ylabel("Real MNIST Test Accuracy (%) - Higher is Better", fontsize=12)
plt.ylim(50, 100)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower right', fontsize=11)

plt.text(150, 55, "Note: 6-bit and 4-bit points collapsed to 10%–70% accuracy\nand appear below the view bounds.",
         fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"))

plt.tight_layout()
plt.savefig("pareto_frontier_final.png", dpi=300)
print("Saved final plot to 'pareto_frontier_final.png'")
