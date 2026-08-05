import json
import matplotlib.pyplot as plt

# Load the JSON data file
with open("combinatorial_results.json", "r") as f:
    data = json.load(f)

sizes = [entry["size"] for entry in data]
times = [entry["proving_time"] for entry in data]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)

# Format the graph
plt.title("EZKL Proving Time vs Tensor Size (Combinatorial: Linear+ReLU+Linear)", fontsize=14)
plt.xlabel("Tensor Size (Elements)", fontsize=12)
plt.ylabel("Proving Time (Seconds)", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)

# plateaus
plt.axvline(x=3500, color='r', linestyle=':', label="Grid Jump (~3.5k)")
plt.axvline(x=7500, color='g', linestyle=':', label="Grid Jump (~7.5k)")
plt.legend()

# Save to an image file
plt.savefig("combinatorial_plateaus.png", dpi=300, bbox_inches="tight")
print("Plot successfully saved to combinatorial_plateaus.png")
