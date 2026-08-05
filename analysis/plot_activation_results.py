import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script lives (.../analysis)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the main repository folder
project_root = os.path.dirname(script_dir)
# Exact paths to the data and figures folders
file_path = os.path.join(project_root, "data", "activation_results_complete.csv")
save_path = os.path.join(project_root, "figures", "final_activation_analysis.png")

# Load the dataset
df = pd.read_csv(file_path)

# Set up colors
colors = {"Linear": "grey", "ReLU": "blue", "Sigmoid": "red", "Tanh": "purple"}
activations = ["Linear", "ReLU", "Sigmoid", "Tanh"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Add powers of 2 vertical lines
def add_power_of_2_lines(ax, max_val):
    power = 1
    while power <= max_val:
        ax.axvline(x=power, color='gray', linestyle='--', alpha=0.4, zorder=0)
        power *= 2

# Determine the maximum tensor size so we know where to stop drawing lines
max_tensor_size = df["Tensor_Size"].max() if not df.empty else 100000

# Graph 1: Proving Time vs Tensor Size
for act in activations:
    subset = df[df["Activation"] == act]
    subset = subset.sort_values(by="Tensor_Size")
    axes[0].plot(subset["Tensor_Size"], subset["Mean_Proving_Time_s"],
                 '-o', color=colors[act], label=act, markersize=4, zorder=2)

axes[0].set_title("Proving Time vs Tensor Size", fontsize=14, pad=10)
axes[0].set_xlabel("Tensor Elements (Size)", fontsize=12)
axes[0].set_ylabel("Proving Time (Seconds)", fontsize=12)
axes[0].grid(True, linestyle=':', alpha=0.6)
add_power_of_2_lines(axes[0], max_tensor_size)
axes[0].legend(fontsize=11)

# Graph 2: Peak RAM vs Tensor Size
for act in activations:
    subset = df[df["Activation"] == act]
    subset = subset.sort_values(by="Tensor_Size")
    axes[1].plot(subset["Tensor_Size"], subset["Peak_RAM_MB"],
                 '-o', color=colors[act], label=act, markersize=4, zorder=2)

axes[1].set_title("Peak RAM vs Tensor Size", fontsize=14, pad=10)
axes[1].set_xlabel("Tensor Elements (Size)", fontsize=12)
axes[1].set_ylabel("Megabytes (MB)", fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.6)
add_power_of_2_lines(axes[1], max_tensor_size)
axes[1].legend(fontsize=11)

# Graph 3: Proof Size vs Tensor Size
for act in activations:
    subset = df[df["Activation"] == act]
    subset = subset.sort_values(by="Tensor_Size")
    # Convert KB to MB for cleaner axis labels
    axes[2].plot(subset["Tensor_Size"], subset["Mean_Proof_Size_KB"] / 1024,
                 '-o', color=colors[act], label=act, markersize=4, zorder=2)

axes[2].set_title("Proof Size vs Tensor Size", fontsize=14, pad=10)
axes[2].set_xlabel("Tensor Elements (Size)", fontsize=12)
axes[2].set_ylabel("Megabytes (MB)", fontsize=12)
axes[2].grid(True, linestyle=':', alpha=0.6)
add_power_of_2_lines(axes[2], max_tensor_size)
axes[2].legend(fontsize=11)

# Clean up the layout and save
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f" Success! Saved image to {save_path}")

# Uncomment the line below if you are running this locally and want a popup window:
plt.show()
