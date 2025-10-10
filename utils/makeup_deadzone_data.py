import csv
import random
import time
from tqdm import tqdm

def save_data_row(position, velocity, pm_values, vm_values, log_file):
    timestamp = time.time()
    row = [timestamp]
    if position is not None:
        row += [position[0], position[1]]
    else:
        row += [None, None]
    if velocity is not None:
        row += [velocity[0], velocity[1]]
    else:
        row += [None, None]
    row += pm_values + vm_values

    with open(log_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def initialize_log_file(log_file):
    header = ["timestamp", "px", "py", "vx", "vy"]
    header += [f"pm{i+1}" for i in range(4)]
    header += [f"vm{i+1}" for i in range(4)]

    with open(log_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

if __name__ == "__main__":
    log_file = "/home/emanuele/Desktop/github/haptic/data/deadzone/deadzone.csv"
    initialize_log_file(log_file)

    N = 8000
    position = [0.0, 0.0]
    velocity = [0.0, 0.0]
    vm_values = [0, 0, 0, 0]
    deadzone_range = [(-150, 150)]
    for i in tqdm(range(N)):
        # Generate random input combination inside the deadzone
        m1 = random.randint(-150, 150)
        m2 = random.randint(-150, 150)
        m3 = random.randint(-150, 150)
        m4 = random.randint(-150, 150)
        pm_values = [m1, m2, m3, m4]

        save_data_row(position, velocity, pm_values, vm_values, log_file)
        time.sleep(0.01)  # Simulate time delay between data points

    print(f"Data collection completed. Data saved to {log_file}.")
    
        