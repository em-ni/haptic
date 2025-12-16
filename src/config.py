import csv
from datetime import datetime
import os
import time
import numpy as np

new_experiment = input("Is this a new experiment? (y/n): ").lower().strip() == 'y'
debug = False

# Conversion factor 155px = 33mm
px_to_mm = 33 / 155  # mm per pixel 

# Set experiment name and save directory
today = time.strftime("%Y-%m-%d")
time_now = time.strftime("%H-%M-%S")
experiment_name = "exp_" + today + "_" + time_now
save_dir = os.path.abspath(os.path.join(".", "data", experiment_name))
csv_path = os.path.abspath(os.path.join(save_dir, f"output_{experiment_name}.csv"))
data_dir = os.path.abspath(os.path.join(".", "data"))

# Set the csv file columns
csv_columns = [
    "timestamp",
    "px",
    "py",
    "vx",
    "vy",
    "pm1",
    "pm2",
    "pm3",
    "pm4",
    "vm1",
    "vm2",
    "vm3",
    "vm4",
]

if new_experiment:
    # If they dont exist, create the directories and the csv file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)

# Colors range for detection
lower_green = np.array([70, 55, 100])
upper_green = np.array([100, 255, 220])



