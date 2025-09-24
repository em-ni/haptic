import argparse
import os
import random
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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

def to_torch_dataset(X: np.ndarray, Y: np.ndarray):
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
                       epochs: int = 100,
                       save_path: str = "best_model.pth"):
    best_val_loss = math.inf
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save model and scalers. If None, uses parent directory of CSV file.")
    parser.add_argument("--no-output-scaling", action="store_true", help="Don't scale outputs (useful if you want raw output).")
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

    # Datasets
    train_ds = to_torch_dataset(X_train_s, Y_train_s)
    val_ds = to_torch_dataset(X_val_s, Y_val_s)
    test_ds = to_torch_dataset(X_test_s, Y_test_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = X_train_s.shape[1]
    out_dim = Y_train_s.shape[1]

    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden_sizes=tuple(args.hidden_sizes), dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Set default save directory if not provided
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.csv)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    scaler_path = os.path.join(args.save_dir, "scalers.pkl")

    print(f"Training model: in={in_dim} out={out_dim} hidden={args.hidden_sizes} epochs={args.epochs}")

    history = train_and_validate(model, opt, loss_fn, train_loader, val_loader, device, epochs=args.epochs, save_path=best_model_path)

    # load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded best model from epoch {checkpoint.get('epoch')} with val_loss={checkpoint.get('val_loss'):.6f}")

    # evaluate on test set (note: y_scaler argument expects the scaler that transforms y from original units to scaled units.
    test_res = evaluate_on_test(model, test_loader, device, y_scaler if y_scaler is not None else None)
    print("Test overall MSE: {:.6f}, MAE: {:.6f}".format(test_res["overall_mse"], test_res["overall_mae"]))
    for i, col in enumerate(output_cols):
        print(f"  {col}: mse={test_res['mse_per_output'][i]:.6f}, mae={test_res['mae_per_output'][i]:.6f}")

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
