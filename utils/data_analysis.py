import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Expecting CSV with columns: pm1..pm4, vm1..vm4, px, py
csv_path = "/home/emanuele/Desktop/github/haptic/data/exp_2025-12-15_11-53-06/output_exp_2025-12-15_11-53-06.csv"
data = pd.read_csv(csv_path)

X = data[["pm1", "pm2", "pm3", "pm4"]].values
y = data[["px", "py"]].values

# Estimate irreducible error via local variance
def local_variance(X, y, k=5):
    # Ensure k doesn't exceed the number of available data points
    k = min(k, len(X))
    
    # If we have fewer than 2 points, we can't compute meaningful variance
    if len(X) < 2:
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([])
    
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
print(f"Mean variance per output: {mean_var}")
print(f"Max variance per output:  {max_var}")

# Compute distance from the origin
data["dist_from_origin"] = np.sqrt(data["px"]**2 + data["py"]**2)
dist_min, dist_max = np.min(data["dist_from_origin"]), np.max(data["dist_from_origin"])

# Compute bins
n_bins = 50
bins = np.linspace(dist_min, dist_max, n_bins + 1)
data["dist_bin"] = np.digitize(data["dist_from_origin"], bins) - 1

# For each point compute distance from origin and assign to bin
bin_stats = []
for b in range(n_bins):
    subset = data[data["dist_bin"] == b]
    if len(subset) >= 2:  # Need at least 2 points for meaningful variance calculation
        bin_mean_var, bin_max_var, _ = local_variance(subset[["pm1", "pm2", "pm3", "pm4"]].values,
                                                      subset[["px", "py"]].values, k=10)
        bin_stats.append((b, len(subset), bin_mean_var, bin_max_var))
    else:
        bin_stats.append((b, len(subset), (np.nan, np.nan), (np.nan, np.nan)))

bin_stats = np.array(bin_stats, dtype=object)   

# Count points per bin
counts = bin_stats[:, 1].astype(int)

# Plot bins (x axis) vs number of points in bin (y axis)
plt.figure(figsize=(10, 5))
plt.bar(range(n_bins), counts)
plt.xlabel("Distance Bin")
plt.ylabel("Number of Points")
plt.title("Number of Points per Distance Bin")
plt.grid()
plt.show()

# Set max number of points to consider per bin
max_points_per_bin = 200
min_points_per_bin = 60

# For each bin if it has more than max_points_per_bin points, randomly sample max_points_per_bin points
# If it has less than min_points_per_bin points, discard the bin
for b in range(n_bins):
    if bin_stats[b][1] > max_points_per_bin:
        subset = data[data["dist_bin"] == b]
        sampled_subset = subset.sample(n=max_points_per_bin, random_state=42)
        data = data[data["dist_bin"] != b]
        data = pd.concat([data, sampled_subset], axis=0)
    elif bin_stats[b][1] < min_points_per_bin:
        data = data[data["dist_bin"] != b]

# Plot bins vs number of points in bin after sampling
counts = []
for b in range(n_bins):
    subset = data[data["dist_bin"] == b]
    counts.append(len(subset))
counts = np.array(counts)
plt.figure(figsize=(10, 5))
plt.bar(range(n_bins), counts)
plt.xlabel("Distance Bin")
plt.ylabel("Number of Points (after sampling)")
plt.title("Number of Points per Distance Bin (after sampling)")
plt.grid()
plt.show()

# Save the new dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
out_path = csv_path.replace("output_", "sampled_").replace(".csv", ".csv")
data.to_csv(out_path, index=False)
print(f"Sampled dataset saved to {out_path}")
