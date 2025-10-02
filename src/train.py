import argparse
import os
import random
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsRegressor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import math
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Model
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout=0.1):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Utils
def load_and_prepare(csv_path: str, include_vm: bool, predict_vel: bool) -> Tuple[np.ndarray, np.ndarray, list, list]:
    # Expected column names in CSV: timestamp,px,py,vx,vy,pm1,pm2,pm3,pm4,vm1,vm2,vm3,vm4
    df = pd.read_csv(csv_path)
    # Check required columns exist:
    req_cols = ['px', 'py', 'vx', 'vy', 'pm1', 'pm2', 'pm3', 'pm4', 'vm1', 'vm2', 'vm3', 'vm4']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}. Make sure your CSV has header row.")

    # Inputs
    input_cols = ['pm1', 'pm2', 'pm3', 'pm4']
    if include_vm:
        input_cols += ['vm1', 'vm2', 'vm3', 'vm4']

    # Outputs
    output_cols = ['px', 'py']
    if predict_vel:
        output_cols += ['vx', 'vy']

    X = df[input_cols].to_numpy(dtype=np.float32)
    Y = df[output_cols].to_numpy(dtype=np.float32)

    return X, Y, input_cols, output_cols

def create_scalers(X_train, Y_train, use_input_scaler=True, use_output_scaler=True):
    x_scaler = StandardScaler() if use_input_scaler else None
    y_scaler = StandardScaler() if use_output_scaler else None

    if x_scaler is not None:
        x_scaler.fit(X_train)
    if y_scaler is not None:
        y_scaler.fit(Y_train)
    return x_scaler, y_scaler

def rebalance_training_data(X_train, Y_train, threshold=0.2, oversample_factor=2):
    """Rebalance training set by oversampling data points far from origin."""
    # Find samples where |px| or |py| > threshold
    mask = (np.abs(Y_train[:, 0]) > threshold) | (np.abs(Y_train[:, 1]) > threshold)
    
    if np.sum(mask) == 0:
        print("No samples found beyond threshold for rebalancing")
        return X_train, Y_train
    
    # Oversample the rare non-zero regions
    n_oversample = min(int(np.sum(mask) * oversample_factor), len(X_train))
    X_extra, Y_extra = resample(X_train[mask], Y_train[mask], replace=True, n_samples=n_oversample, random_state=42)
    
    X_train_balanced = np.vstack([X_train, X_extra])
    Y_train_balanced = np.vstack([Y_train, Y_extra])
    
    print(f"Rebalanced training data: {len(X_train)} -> {len(X_train_balanced)} samples")
    print(f"Added {n_oversample} oversampled points from regions |px|,|py| > {threshold}")
    
    return X_train_balanced, Y_train_balanced

def analyze_noise_floor(X, Y, input_cols, output_cols, k=5, tolerance=1e-3):
    """Analyze irreducible error by examining repeated/similar inputs."""
    print("\n=== NOISE FLOOR ANALYSIS ===")
    
    # Method 1: Find exact repeated inputs
    print("1. Analyzing exact repeated inputs...")
    unique_x, inverse_indices, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    repeated_mask = counts > 1
    
    if np.sum(repeated_mask) > 0:
        print(f"Found {np.sum(repeated_mask)} unique input combinations with repeats")
        
        total_variance = 0
        n_groups = 0
        for i, unique_input in enumerate(unique_x[repeated_mask]):
            group_mask = (inverse_indices == np.where(repeated_mask)[0][n_groups])
            group_outputs = Y[group_mask]
            
            if len(group_outputs) > 1:
                variance_per_output = np.var(group_outputs, axis=0)
                print(f"  Input group {n_groups+1}: {counts[repeated_mask][n_groups]} repeats, variance: {variance_per_output}")
                total_variance += np.mean(variance_per_output)
                n_groups += 1
        
        if n_groups > 0:
            avg_variance = total_variance / n_groups
            print(f"Average variance across repeated inputs: {avg_variance:.6f}")
    else:
        print("No exact repeated inputs found")
    
    # Method 2: Nearest neighbor consistency
    print("\n2. Analyzing nearest neighbor consistency...")
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)  # +1 because first neighbor is itself
    distances, indices = nbrs.kneighbors(X)
    
    local_variances = []
    for i in range(len(X)):
        neighbor_indices = indices[i][1:]  # exclude self
        neighbor_outputs = Y[neighbor_indices]
        if len(neighbor_outputs) > 1:
            local_var = np.mean(np.var(neighbor_outputs, axis=0))
            local_variances.append(local_var)
    
    if local_variances:
        avg_local_variance = np.mean(local_variances)
        print(f"Average local variance (k={k} neighbors): {avg_local_variance:.6f}")
        print(f"Median local variance: {np.median(local_variances):.6f}")
        print(f"Max local variance: {np.max(local_variances):.6f}")
    
    # Method 3: kNN baseline
    print("\n3. kNN baseline comparison...")
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X, Y)
    knn_pred = knn.predict(X)
    knn_mse = np.mean((knn_pred - Y) ** 2)
    knn_mae = np.mean(np.abs(knn_pred - Y))
    
    print(f"kNN (k={k}) training MSE: {knn_mse:.6f}")
    print(f"kNN (k={k}) training MAE: {knn_mae:.6f}")
    
    return {
        "knn_mse": knn_mse,
        "knn_mae": knn_mae,
        "avg_local_variance": np.mean(local_variances) if local_variances else None
    }

def compute_density_weights(Y: np.ndarray, bin_size: float = 0.1, min_weight: float = 0.1, max_weight: float = 10.0):
    """
    Compute density-based weights for samples to reduce bias toward dense regions.
    
    Args:
        Y: Output data (N, output_dim) - typically (px, py) positions
        bin_size: Size of bins for histogram (default 0.1)
        min_weight: Minimum weight value to prevent extreme downweighting
        max_weight: Maximum weight value to prevent extreme upweighting
    
    Returns:
        weights: Array of weights (N,) for each sample
    """
    # Only use px, py for density calculation (first 2 columns)
    positions = Y[:, :2]  # (px, py)
    
    # Create 2D histogram to estimate density
    px_min, px_max = positions[:, 0].min(), positions[:, 0].max()
    py_min, py_max = positions[:, 1].min(), positions[:, 1].max()
    
    # Add small margin to avoid edge effects
    margin = bin_size * 0.1
    px_bins = np.arange(px_min - margin, px_max + margin + bin_size, bin_size)
    py_bins = np.arange(py_min - margin, py_max + margin + bin_size, bin_size)
    
    # Compute 2D histogram
    hist, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[px_bins, py_bins])
    
    # Find which bin each sample belongs to
    px_indices = np.digitize(positions[:, 0], px_bins) - 1
    py_indices = np.digitize(positions[:, 1], py_bins) - 1
    
    # Clamp indices to valid range
    px_indices = np.clip(px_indices, 0, hist.shape[0] - 1)
    py_indices = np.clip(py_indices, 0, hist.shape[1] - 1)
    
    # Get density for each sample
    densities = hist[px_indices, py_indices]
    
    # Compute weights as inverse density (with smoothing)
    epsilon = 1e-6
    weights = 1.0 / (densities + epsilon)
    
    # Normalize weights to have reasonable range
    weights = weights / np.mean(weights)
    
    # Clamp to prevent extreme values
    weights = np.clip(weights, min_weight, max_weight)
    
    print(f"Density weights - Min: {weights.min():.3f}, Mean: {weights.mean():.3f}, Max: {weights.max():.3f}")
    print(f"High-weight samples (>{weights.mean() + weights.std():.3f}): {np.sum(weights > weights.mean() + weights.std())}")
    
    return weights

def analyze_binwise_variance(X: np.ndarray, Y: np.ndarray, input_cols: list, output_cols: list, bin_size: float = 0.1):
    """
    Analyze variance within spatial bins to check for irreducible error.
    
    Args:
        X: Input data (N, input_dim)
        Y: Output data (N, output_dim) 
        input_cols: Names of input columns
        output_cols: Names of output columns
        bin_size: Size of bins for analysis
    
    Returns:
        dict with analysis results
    """
    print(f"\n=== BINWISE VARIANCE ANALYSIS (bin_size={bin_size}) ===")
    
    # Only use px, py for binning (first 2 columns of Y)
    positions = Y[:, :2]  # (px, py)
    
    # Create bins
    px_min, px_max = positions[:, 0].min(), positions[:, 0].max()
    py_min, py_max = positions[:, 1].min(), positions[:, 1].max()
    
    margin = bin_size * 0.1
    px_bins = np.arange(px_min - margin, px_max + margin + bin_size, bin_size)
    py_bins = np.arange(py_min - margin, py_max + margin + bin_size, bin_size)
    
    # Assign samples to bins
    px_indices = np.digitize(positions[:, 0], px_bins) - 1
    py_indices = np.digitize(positions[:, 1], py_bins) - 1
    
    # Clamp to valid range
    px_indices = np.clip(px_indices, 0, len(px_bins) - 2)
    py_indices = np.clip(py_indices, 0, len(py_bins) - 2)
    
    # Create bin identifiers
    bin_ids = px_indices * len(py_bins) + py_indices
    
    # Analyze each bin
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    
    variances = []
    high_variance_bins = []
    bin_stats = []
    
    for i, bin_id in enumerate(unique_bins):
        if counts[i] > 1:  # Need at least 2 samples for variance
            mask = bin_ids == bin_id
            bin_outputs = Y[mask]
            bin_inputs = X[mask]
            
            # Compute variance for each output dimension
            output_variances = np.var(bin_outputs, axis=0)
            mean_variance = np.mean(output_variances)
            
            # Check if inputs are similar in this bin
            input_variances = np.var(bin_inputs, axis=0) if counts[i] > 1 else np.zeros(bin_inputs.shape[1])
            mean_input_variance = np.mean(input_variances)
            
            # Reconstruct bin center
            px_idx = bin_id // len(py_bins)
            py_idx = bin_id % len(py_bins)
            bin_center_px = px_bins[px_idx] + bin_size / 2
            bin_center_py = py_bins[py_idx] + bin_size / 2
            
            bin_stats.append({
                'bin_center': (bin_center_px, bin_center_py),
                'count': counts[i],
                'output_variance': output_variances,
                'mean_output_variance': mean_variance,
                'input_variance': input_variances,
                'mean_input_variance': mean_input_variance
            })
            
            variances.append(mean_variance)
            
            # Flag high variance bins
            if mean_variance > 0.01:  # Threshold for "high" variance
                high_variance_bins.append({
                    'center': (bin_center_px, bin_center_py),
                    'count': counts[i],
                    'variance': mean_variance
                })
    
    if variances:
        mean_variance = np.mean(variances)
        max_variance = np.max(variances)
        high_var_count = len(high_variance_bins)
        
        print(f"Analyzed {len(variances)} bins with >1 sample")
        print(f"Mean output variance across bins: {mean_variance:.6f}")
        print(f"Max output variance: {max_variance:.6f}")
        print(f"High variance bins (>0.01): {high_var_count}")
        
        if high_variance_bins:
            print("\nTop high-variance bins:")
            sorted_bins = sorted(high_variance_bins, key=lambda x: x['variance'], reverse=True)[:5]
            for bin_info in sorted_bins:
                print(f"  Center: ({bin_info['center'][0]:.2f}, {bin_info['center'][1]:.2f}), "
                      f"Count: {bin_info['count']}, Variance: {bin_info['variance']:.6f}")
        
        return {
            'mean_variance': mean_variance,
            'max_variance': max_variance,
            'high_variance_bins': high_variance_bins,
            'bin_stats': bin_stats
        }
    else:
        print("No bins with multiple samples found")
        return None

def to_torch_dataset(X: np.ndarray, Y: np.ndarray, weights: np.ndarray = None):
    if weights is not None:
        return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(weights).float())
    else:
        return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

def mse_mae(y_pred: np.ndarray, y_true: np.ndarray):
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    return mse, mae

# Train and evaluate
def train_and_validate(model: nn.Module,
                       opt,
                       loss_fn,
                       train_loader,
                       val_loader,
                       device,
                       epochs: int = 150,
                       save_path: str = "best_model.pth",
                       use_weighted_loss: bool = False,
                       use_density_weights: bool = False):
    best_val_loss = math.inf
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_data in train_loader:
            if use_density_weights and len(batch_data) == 3:
                xb, yb, wb = batch_data
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
            else:
                xb, yb = batch_data
                xb = xb.to(device)
                yb = yb.to(device)
                wb = None
            
            opt.zero_grad()
            out = model(xb)
            
            if use_weighted_loss and not use_density_weights:
                # Original weighted loss: give higher weight to samples farther from origin
                diff = torch.norm(yb, dim=1)  # distance from zero
                weights = 1.0 + diff  # can tune this formula
                loss = (weights * (out - yb).pow(2).sum(1)).mean()
            elif use_density_weights and wb is not None:
                # Density-based weighted loss
                individual_losses = (out - yb).pow(2).sum(1)  # MSE per sample
                weighted_losses = wb * individual_losses
                loss = weighted_losses.mean()
            else:
                loss = loss_fn(out, yb)
            
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(1, n_batches)

        # validation
        model.eval()
        val_running = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                val_running += loss_fn(out, yb).item()
                n_val += 1
        val_loss = val_running / max(1, n_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, save_path)

    return history

def evaluate_on_test(model: nn.Module, test_loader: DataLoader, device, y_scaler=None):
    model.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            y_preds.append(out)
            y_trues.append(yb.numpy())
    y_pred = np.vstack(y_preds)
    y_true = np.vstack(y_trues)

    # if y_scaler is present reverse transform to original units
    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred)
        y_true = y_scaler.inverse_transform(y_true)

    mse = np.mean((y_pred - y_true) ** 2, axis=0)
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    overall_mse = float(np.mean(mse))
    overall_mae = float(np.mean(mae))
    return {"mse_per_output": mse, "mae_per_output": mae, "overall_mse": overall_mse, "overall_mae": overall_mae, "y_pred": y_pred, "y_true": y_true}


def main():
    parser = argparse.ArgumentParser(description="Train a small MLP mapping pm* (+ optional vm*) to px,py (+ optional vx,vy).")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file (header required).")
    parser.add_argument("--include-vm", action="store_true", help="Include vm1..vm4 in the inputs.")
    parser.add_argument("--predict-vel", action="store_true", help="Also predict vx,vy in outputs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save model and scalers. If None, uses parent directory of CSV file.")
    parser.add_argument("--no-output-scaling", action="store_true", help="Don't scale outputs (useful if you want raw output).")
    parser.add_argument("--rebalance", action="store_true", help="Rebalance training data by oversampling non-zero regions.")
    parser.add_argument("--rebalance-threshold", type=float, default=0.2, help="Threshold for rebalancing (samples with |px|,|py| > threshold).")
    parser.add_argument("--oversample-factor", type=float, default=2.0, help="Factor for oversampling non-zero regions.")
    parser.add_argument("--weighted-loss", action="store_true", help="Use weighted loss giving more weight to samples far from origin.")
    parser.add_argument("--density-weights", action="store_true", help="Use density-based weights to reduce bias toward center (0,0).")
    parser.add_argument("--density-bin-size", type=float, default=0.1, help="Bin size for density estimation (default: 0.1).")
    parser.add_argument("--density-min-weight", type=float, default=0.1, help="Minimum weight for density weighting (default: 0.1).")
    parser.add_argument("--density-max-weight", type=float, default=10.0, help="Maximum weight for density weighting (default: 10.0).")
    parser.add_argument("--alternative-loss", type=str, choices=["mse", "smoothl1", "mae"], default="mse", help="Loss function to use.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam optimizer.")
    parser.add_argument("--analyze-noise", action="store_true", help="Analyze noise floor and irreducible error.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X, Y, input_cols, output_cols = load_and_prepare(args.csv, include_vm=args.include_vm, predict_vel=args.predict_vel)
    print(f"Loaded data: X shape {X.shape}, Y shape {Y.shape}")
    print("Input columns:", input_cols)
    print("Output columns:", output_cols)

    # train/val/test split
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=args.seed)
    # split temp into train and val
    val_fraction_of_temp = args.val_size / (1.0 - args.test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=val_fraction_of_temp, random_state=args.seed)

    # Optional: Analyze noise floor before training
    if args.analyze_noise:
        noise_analysis = analyze_noise_floor(X, Y, input_cols, output_cols)
        # Also analyze binwise variance for irreducible error
        binwise_analysis = analyze_binwise_variance(X, Y, input_cols, output_cols, bin_size=args.density_bin_size)

    # Optional: Rebalance training data
    if args.rebalance:
        X_train, Y_train = rebalance_training_data(X_train, Y_train, 
                                                 threshold=args.rebalance_threshold,
                                                 oversample_factor=args.oversample_factor)

    # scalers
    x_scaler, y_scaler = create_scalers(X_train, Y_train, use_input_scaler=True, use_output_scaler=(not args.no_output_scaling))
    # transform
    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    if y_scaler is not None:
        Y_train_s = y_scaler.transform(Y_train)
        Y_val_s = y_scaler.transform(Y_val)
        Y_test_s = y_scaler.transform(Y_test)
    else:
        Y_train_s = Y_train.copy()
        Y_val_s = Y_val.copy()
        Y_test_s = Y_test.copy()

    # Compute density-based weights if requested
    train_weights = None
    if args.density_weights:
        print("\nComputing density-based weights...")
        # Use original (unscaled) Y_train for density calculation to maintain physical meaning
        train_weights = compute_density_weights(
            Y_train, 
            bin_size=args.density_bin_size,
            min_weight=args.density_min_weight,
            max_weight=args.density_max_weight
        )

    # Datasets
    train_ds = to_torch_dataset(X_train_s, Y_train_s, train_weights)
    val_ds = to_torch_dataset(X_val_s, Y_val_s)
    test_ds = to_torch_dataset(X_test_s, Y_test_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = X_train_s.shape[1]
    out_dim = Y_train_s.shape[1]

    model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Choose loss function
    if args.alternative_loss == "smoothl1":
        loss_fn = nn.SmoothL1Loss()
    elif args.alternative_loss == "mae":
        loss_fn = nn.L1Loss()
    else:  # mse
        loss_fn = nn.MSELoss()

    # Set default save directory if not provided
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.csv)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    scaler_path = os.path.join(args.save_dir, "scalers.pkl")

    print(f"Training model: in={in_dim} out={out_dim} hidden={args.hidden_sizes} epochs={args.epochs}")
    print(f"Loss function: {args.alternative_loss}, Weighted loss: {args.weighted_loss}, Density weights: {args.density_weights}, Weight decay: {args.weight_decay}")

    history = train_and_validate(model, opt, loss_fn, train_loader, val_loader, device, 
                                epochs=args.epochs, save_path=best_model_path, 
                                use_weighted_loss=args.weighted_loss,
                                use_density_weights=args.density_weights)

    # load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded best model from epoch {checkpoint.get('epoch')} with val_loss={checkpoint.get('val_loss'):.6f}")

    # evaluate on test set (note: y_scaler argument expects the scaler that transforms y from original units to scaled units.
    test_res = evaluate_on_test(model, test_loader, device, y_scaler if y_scaler is not None else None)
    print("Test overall MSE: {:.6f}, MAE: {:.6f}".format(test_res["overall_mse"], test_res["overall_mae"]))
    for i, col in enumerate(output_cols):
        print(f"  {col}: mse={test_res['mse_per_output'][i]:.6f}, mae={test_res['mae_per_output'][i]:.6f}")

    # Compare with kNN baseline on test set
    if args.analyze_noise:
        print("\n=== BASELINE COMPARISON ===")
        knn_test = KNeighborsRegressor(n_neighbors=5)
        knn_test.fit(X_train_s, Y_train_s if y_scaler is None else Y_train)
        knn_test_pred = knn_test.predict(X_test_s)
        
        if y_scaler is not None:
            knn_test_pred = y_scaler.inverse_transform(knn_test_pred)
            test_true = y_scaler.inverse_transform(Y_test_s)
        else:
            test_true = Y_test
            
        knn_test_mse = np.mean((knn_test_pred - test_true) ** 2)
        knn_test_mae = np.mean(np.abs(knn_test_pred - test_true))
        
        print(f"kNN test MSE: {knn_test_mse:.6f}, MAE: {knn_test_mae:.6f}")
        print(f"MLP vs kNN MSE ratio: {test_res['overall_mse'] / knn_test_mse:.3f}")

    # Save scalers and metadata
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "input_cols": input_cols,
            "output_cols": output_cols,
        }, f)
    print(f"Saved model to {best_model_path} and scalers to {scaler_path}")

    # plot train vs val loss
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Training Curve")
    plt.tight_layout()
    loss_plot_path = os.path.join(args.save_dir, "training_curve.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curve to {loss_plot_path}")
    plt.show()

    # scatter plot predictions vs true for px,py (if available)
    y_pred = test_res["y_pred"]
    y_true = test_res["y_true"]
    n_out = y_true.shape[1]
    cols_to_plot = min(4, n_out)
    fig, axs = plt.subplots(1, cols_to_plot, figsize=(4*cols_to_plot, 4))
    if cols_to_plot == 1:
        axs = [axs]
    for i in range(cols_to_plot):
        axs[i].scatter(y_true[:, i], y_pred[:, i], s=6, alpha=0.6)
        axs[i].plot([y_true[:, i].min(), y_true[:, i].max()], [y_true[:, i].min(), y_true[:, i].max()], '--')
        axs[i].set_xlabel("true")
        axs[i].set_ylabel("pred")
        axs[i].set_title(output_cols[i])
    plt.tight_layout()
    pred_plot_path = os.path.join(args.save_dir, "predictions_vs_actual.png")
    plt.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved predictions vs actual plot to {pred_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
