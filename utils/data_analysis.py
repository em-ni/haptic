import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# === Load your dataset ===
# Expecting CSV with columns: pm1..pm4, vm1..vm4, px, py
data = pd.read_csv("/home/emanuele/Desktop/github/haptic/data/exp_2025-10-01_11-59-24/output_exp_2025-10-01_11-59-24.csv")

X = data[["pm1", "pm2", "pm3", "pm4"]].values
y = data[["px", "py"]].values

# === Step 1: Estimate irreducible error via local variance ===
def local_variance(X, y, k=5):
    tree = cKDTree(X)
    vars_ = []
    means_ = []
    for i in range(len(X)):
        dists, idxs = tree.query(X[i], k=k)
        neigh_y = y[idxs]
        vars_.append(np.var(neigh_y, axis=0))
        means_.append(np.mean(neigh_y, axis=0))
    return np.mean(vars_, axis=0), np.max(vars_, axis=0), np.array(means_)

mean_var, max_var, neigh_means = local_variance(X, y, k=10)
print("=== Local Variance Estimates (Noise Floor) ===")
print(f"Mean variance per output: {mean_var}")
print(f"Max variance per output:  {max_var}")

# === Step 2: Visualize noise in output space ===
plt.figure(figsize=(8,8))
plt.scatter(y[:,0], y[:,1], alpha=0.3, s=10, label="Recorded points")
plt.scatter(neigh_means[:,0], neigh_means[:,1], alpha=0.3, s=10, label="Local means")
plt.xlabel("px")
plt.ylabel("py")
plt.title("Tip position with local neighborhood means")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("tip_positions_with_noise.png")

# === Step 3: Precision visualization (local neighborhoods) ===
# Plot spread of neighbors around some random samples
np.random.seed(42)
sample_idxs = np.random.choice(len(X), size=5, replace=False)

fig, axs = plt.subplots(1,5, figsize=(20,4))
for ax, idx in zip(axs, sample_idxs):
    dists, neigh_idxs = cKDTree(X).query(X[idx], k=20)
    neigh_y = y[neigh_idxs]
    ax.scatter(neigh_y[:,0], neigh_y[:,1], alpha=0.6, s=15)
    ax.scatter(y[idx,0], y[idx,1], color="red", marker="x", s=50)
    ax.set_title(f"Sample {idx}")
    ax.axis("equal")
plt.tight_layout()
plt.savefig("neighborhood_spread.png")

# === Step 4: Frequency plots of px and py ===
fig, axs = plt.subplots(1,2, figsize=(12,5))
axs[0].hist(y[:,0], bins=50, alpha=0.7)
axs[0].set_title("Frequency of px")
axs[0].set_xlabel("px")
axs[0].set_ylabel("Count")

axs[1].hist(y[:,1], bins=50, alpha=0.7)
axs[1].set_title("Frequency of py")
axs[1].set_xlabel("py")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("frequency_px_py.png")