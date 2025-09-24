import pandas as pd
import matplotlib.pyplot as plt
import src.config as config

# Load CSV
csv_path = "/home/emanuele/Desktop/github/haptic/data/exp_2025-09-24_11-34-57/output_exp_2025-09-24_11-34-57.csv"
df = pd.read_csv(csv_path)

# Extract positions and velocities
x = df["px"].values
y = df["py"].values
vx = df["vx"].values
vy = df["vy"].values

# Plot points
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=30, c="blue", label="Position")

# Plot velocity vectors
scale = 0.01  # adjust this to control arrow length
plt.quiver(x, y, vx, vy, angles="xy", scale_units="xy", scale=1/scale, color="red", width=0.001, label="Velocity")

# Labels and grid
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Position and Velocity Vectors")
plt.legend()
plt.axis("equal")
plt.grid(False)

# Save figure in the same directory as CSV
fig_path = csv_path.replace("output_", "trajectory_").replace(".csv", ".png")
plt.savefig(fig_path)
print(f"Figure saved to {fig_path}")

plt.show()


