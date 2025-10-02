"""
Give the same input many times and measure all the ouputs.
Check the variance of the outputs to estimate the noise floor.
Plot some visualizations.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from random import random
from src import tracker
from src import controller
import threading
import time
import signal

N = 5  # Number of random points to test
n = 10  # Number of repetitions per point

def get_random_point():
    positions = list(range(255, 149, -10)) + list(range(-150, -255 + 1, -10))
    # Return string in format 'x y, z, w'
    return f"{positions[int(random()*len(positions))]}, {positions[int(random()*len(positions))]}, {positions[int(random()*len(positions))]}, {positions[int(random()*len(positions))]}"

def save_data(inputs, outputs, filename="test_accuracy_and_precision.csv"):
    import csv
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Input", "Output_X", "Output_Y", "Output_Z"])
        for inp, out in zip(inputs, outputs):
            writer.writerow([inp, *out])  # unpack tuple

def data_analysis():

    # For each point, compute mean and stddev of outputs
    import pandas as pd
    import matplotlib.pyplot as plt
    
    data = pd.read_csv("test_accuracy_and_precision.csv")
    grouped = data.groupby("Input")[["Output_X", "Output_Y", "Output_Z"]].agg(["mean","std"])
    print(grouped)
    print("\n=== Output Statistics per Input ===")

    # Get unique inputs for plotting
    unique_inputs = data["Input"].unique()
    
    # Create a summary plot for X and Y coordinates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot X coordinates
    x_means = []
    x_stds = []
    y_means = []
    y_stds = []
    input_labels = []
    
    for inp in unique_inputs:
        subset = data[data["Input"] == inp]
        if len(subset) > 0:
            x_mean = subset["Output_X"].mean()
            x_std = subset["Output_X"].std()
            y_mean = subset["Output_Y"].mean()
            y_std = subset["Output_Y"].std()
            
            x_means.append(x_mean)
            x_stds.append(x_std)
            y_means.append(y_mean)
            y_stds.append(y_std)
            input_labels.append(str(inp))
    
    # Plot X coordinates
    ax1.errorbar(range(len(input_labels)), x_means, yerr=x_stds, fmt='o', ecolor='r', capsize=5)
    ax1.set_xlabel("Input Index")
    ax1.set_ylabel("Output X Position")
    ax1.set_title("Output X Position vs Input with Error Bars")
    ax1.grid()
    ax1.set_xticks(range(len(input_labels)))
    ax1.set_xticklabels([f"Input {i+1}" for i in range(len(input_labels))], rotation=45)
    
    # Plot Y coordinates
    ax2.errorbar(range(len(input_labels)), y_means, yerr=y_stds, fmt='o', ecolor='b', capsize=5)
    ax2.set_xlabel("Input Index")
    ax2.set_ylabel("Output Y Position")
    ax2.set_title("Output Y Position vs Input with Error Bars")
    ax2.grid()
    ax2.set_xticks(range(len(input_labels)))
    ax2.set_xticklabels([f"Input {i+1}" for i in range(len(input_labels))], rotation=45)
    
    plt.tight_layout()
    plt.savefig("accuracy_and_precision.png")
    print("Plot saved to accuracy_and_precision.png")
    plt.show()

    # # Plot scatter plots for each input point showing the spread of outputs
    # for i, inp in enumerate(unique_inputs):
    #     subset = data[data["Input"] == inp]
    #     if len(subset) > 0:
    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(subset["Output_X"], subset["Output_Y"], alpha=0.6, s=50)
            
    #         # Add mean point with error bars
    #         x_mean = subset["Output_X"].mean()
    #         y_mean = subset["Output_Y"].mean()
    #         x_std = subset["Output_X"].std()
    #         y_std = subset["Output_Y"].std()
            
    #         plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='ro', markersize=8, capsize=5, label='Mean ± Std')
            
    #         plt.xlabel("Output X Position")
    #         plt.ylabel("Output Y Position")
    #         plt.title(f"Spread of Outputs for Input: {inp}")
    #         plt.legend()
    #         plt.grid()
            
    #         # Create safe filename by replacing problematic characters
    #         safe_filename = str(inp).replace(", ", "_").replace("-", "neg")
    #         plt.savefig(f"spread_input_{safe_filename}.png")
    #         print(f"Plot saved to spread_input_{safe_filename}.png")
    #         plt.show()

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    tracker_instance.quit = True
    controller_instance.close()
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

try:
    # Start tracker in a separate thread
    tracker_instance = tracker.Tracker(cam_index=0)
    tracker_thread = threading.Thread(target=tracker_instance.track, daemon=False)
    tracker_thread.start()
    
    # Give tracker time to initialize
    print("Initializing tracker, please wait...")
    time.sleep(3)
    print("Tracker initialized.")

    # Initialize controller with tracker instance
    controller_instance = controller.Controller(arduino_port="/dev/ttyUSB0", tracker=tracker_instance)

    # Print initial coordinates of the green tip
    if controller_instance.tracker is not None:
        origin = controller_instance.tracker.get_position()
        print("Initial tip position:", origin)
        controller_instance.init_coordinates = origin
    else:
        print("No tracker provided, skipping initial position print.")

    
    # Send random points and collect outputs
    all_inputs = []
    all_outputs = []
    for p in range(N):
        point = get_random_point()
        # point = '245, 225, -210, 245'
        print(f"\nMoving to random point {p+1}/{N}: {point}")
        for i in range(n):
            print(f"  Repetition {i+1}/{n}")
            temp_point = get_random_point()
            controller_instance.send_arduino(temp_point, bomb=1)
            time.sleep(3)
            start = time.time()
            controller_instance.send_arduino(point, bomb=1)
            end = time.time()
            # print(f"  Command sent in {end - start:.2f} seconds")
            time.sleep(3)  # Wait for the system to stabilize
            position = controller_instance.tracker.get_position()
            all_inputs.append(point)
            all_outputs.append(position)
            print(f"Recorded position: {position}")

    # Save collected data
    save_data(all_inputs, all_outputs)
    
    # Analyze data
    data_analysis()

    # Keep main thread alive and wait for threads to complete
    print("System running. Press Ctrl+C to stop.")
    
    # Stop tracker
    tracker_instance.quit = True
    tracker_thread.join()
    
    print("\nExperiment completed successfully.")

except Exception as e:
    print(f"Error occurred: {e}")
    if 'tracker_instance' in locals():
        tracker_instance.quit = True
    if 'controller_instance' in locals():
        controller_instance.close()
    sys.exit(1)

finally:
    # Cleanup
    if 'controller_instance' in locals():
        controller_instance.close()
