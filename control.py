import math
from src import config, tracker
from src import controller
from src.mpc import MPCController, MPCConfig
import threading
import time
import signal
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


STEP = False
LINE = False
CIRCLE = False
LETTER = True

def letter_U(N=100):
    """
    Generate trajectory points forming the letter U in the XY plane.
    """
    # Calculate proportional distribution for N total points
    length_left = 2.0
    length_bottom = math.pi * 1
    length_right = 2.0
    total_length = length_left + length_bottom + length_right

    N_left = int(N * length_left / total_length)
    N_bottom = int(N * length_bottom / total_length)
    N_right = N - N_left - N_bottom  # Ensure exactly N points

    # Left line from -1,1 to -1,-1
    left_line_x_values = np.linspace(-1, -1, N_left)
    left_line_y_values = np.linspace(1, -1, N_left)
    left_line = np.column_stack((left_line_x_values, left_line_y_values))

    # Bottom semicircle from -1,-1 to 1,-1
    theta = np.linspace(np.pi, 0, N_bottom)
    bottom_curve_x = np.cos(theta)
    bottom_curve_y = -1 - np.sin(theta)
    bottom_curve = np.column_stack((bottom_curve_x, bottom_curve_y))

    # Right line from 1,-1 to 1,1
    right_line_x_values = np.linspace(1, 1, N_right)
    right_line_y_values = np.linspace(-1, 1, N_right)
    right_line = np.column_stack((right_line_x_values, right_line_y_values))
    return np.vstack((left_line, bottom_curve, right_line))

def letter_N(N=100):
    """
    Generate trajectory points forming the letter N in the XY plane.
    """
    # Calculate proportional distribution for N total points
    length_left = 3.0
    length_diag = math.sqrt((2.0)**2 + (3.0)**2)
    length_right = 3.0
    total_length = length_left + length_diag + length_right

    N_left = int(N * length_left / total_length)
    N_diag = int(N * length_diag / total_length)
    N_right = N - N_left - N_diag  # Ensure exactly N points

    # Left line from -1,-1.5 to -1,1.5
    left_line_x_values = np.linspace(-1, -1, N_left)
    left_line_y_values = np.linspace(-1.5, 1.5, N_left)
    left_line = np.column_stack((left_line_x_values, left_line_y_values))

    # Diagonal line from -1,1.5 to 1,-1.5
    diag_x_values = np.linspace(-1, 1, N_diag)
    diag_y_values = np.linspace(1.5, -1.5, N_diag)
    diag_line = np.column_stack((diag_x_values, diag_y_values))

    # Right line from 1,-1.5 to 1,1.5
    right_line_x_values = np.linspace(1, 1, N_right)
    right_line_y_values = np.linspace(-1.5, 1.5, N_right)
    right_line = np.column_stack((right_line_x_values, right_line_y_values))
    return np.vstack((left_line, diag_line, right_line))

def letter_S(N=100):
    """
    Generate trajectory points forming the letter S in the XY plane.
    """
    # Calculate proportional distribution for N total points
    length_top = (3/4) * 2 * math.pi * 1
    length_bottom = (3/4) * 2 * math.pi * 1
    total_length = length_top + length_bottom

    N_top = int(N * length_top / total_length)
    N_bottom = N - N_top  # Ensure exactly N points

    # Top 3/4 circle centred at 0,1
    theta_top = np.linspace(0, (3/4) * 2 * np.pi, N_top)
    top_curve_x = np.cos(theta_top)
    top_curve_y = 1 + np.sin(theta_top)
    top_curve = np.column_stack((top_curve_x, top_curve_y))

    # Bottom 3/4 circle centred at 0,-1
    theta_bottom = np.linspace(np.pi/2, -np.pi, N_bottom)
    bottom_curve_x = np.cos(theta_bottom)
    bottom_curve_y = -1 + np.sin(theta_bottom)
    bottom_curve = np.column_stack((bottom_curve_x, bottom_curve_y))
    
    return np.vstack((top_curve, bottom_curve))

def letter_W(N=100):
    """
    Generate trajectory points forming the letter W in the XY plane.
    """
    # Calculate proportional distribution for N total points
    length1 = math.sqrt((1)**2 + (2.0)**2)
    length2 = math.sqrt((1)**2 + (2.0)**2)
    length3 = math.sqrt((1)**2 + (2.0)**2)
    length4 = math.sqrt((1)**2 + (2.0)**2)
    total_length = length1 + length2 + length3 + length4

    N1 = int(N * length1 / total_length)
    N2 = int(N * length2 / total_length)
    N3 = int(N * length3 / total_length)
    N4 = N - N1 - N2 - N3  # Ensure exactly N points

    # Line 1 from -2,1.0 to -1,-1
    line1_x_values = np.linspace(-2, -1, N1)
    line1_y_values = np.linspace(1.0, -1.0, N1)
    line1 = np.column_stack((line1_x_values, line1_y_values))

    # Line 2 from -1,-1 to 0,1.0
    line2_x_values = np.linspace(-1, 0, N2)
    line2_y_values = np.linspace(-1.0, 1.0, N2)
    line2 = np.column_stack((line2_x_values, line2_y_values))

    # Line 3 from 0,1.0 to 1,-1
    line3_x_values = np.linspace(0, 1, N3)
    line3_y_values = np.linspace(1.0, -1.0, N3)
    line3 = np.column_stack((line3_x_values, line3_y_values))

    # Line 4 from 1,-1 to 2,1.0
    line4_x_values = np.linspace(1, 2, N4)
    line4_y_values = np.linspace(-1.0, 1.0, N4)
    line4 = np.column_stack((line4_x_values, line4_y_values))   

    return np.vstack((line1, line2, line3, line4))


def line_trajectory(start, end, N):
    # Return a 2D trajectory between the start and end point with N points
    x_values = np.linspace(start[0], end[0], N)
    y_values = np.linspace(start[1], end[1], N)
    return np.vstack((x_values, y_values)).T

def circle_trajectory(center, radius, N):
    # Return a circular trajectory around center with given radius and N points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_values = center[0] + radius * np.cos(angles)
    y_values = center[1] + radius * np.sin(angles)
    return np.vstack((x_values, y_values)).T

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

    # Initialize MPC controller
    print("Initializing MPC controller...")
    mpc_controller = MPCController()
    print("MPC controller initialized.")

    # Print initial coordinates of the green tip
    if controller_instance.tracker is not None:
        origin = controller_instance.tracker.get_position()
        print("Initial tip position:", origin)
        controller_instance.init_coordinates = origin
    else:
        print("No tracker provided, skipping initial position print.")
    
    # Initialize target position variable
    target_position = np.array([0.0, 0.0])  # Default target position
    
    if STEP:
        # Define target trajectory (simple step reference)
        target_position = np.array([0.9, 0.5])  # Target px, py in mm
        print(f"Target position: {target_position} mm")

    if LINE:
        # Define target trajectory (simple line reference)
        line_start = np.array([1.7, -1.7])
        line_end = np.array([-1.7, 1.7])
        target_trajectory = line_trajectory(line_start, line_end, N=100)
        print(f"Target trajectory (line): {target_trajectory} mm")
        # Initialize target_position with first point of trajectory
        target_position = target_trajectory[0]

    if CIRCLE:
        # Define target trajectory (circle reference, 2mm diameter = 1mm radius)
        circle_center = np.array([0.0, 0.0])  # Center at origin
        circle_radius = 2.0  # mm
        target_trajectory = circle_trajectory(circle_center, circle_radius, N=100)
        print(f"Target trajectory (circle): {target_trajectory} mm")

        # # Line from center to first point of circle
        # line_start = np.array([0.0, 0.0])
        # line_end = target_trajectory[0]
        # approach_trajectory = line_trajectory(line_start, line_end, N=10)

        # # Prepend approach trajectory to circle trajectory
        # target_trajectory = np.vstack((approach_trajectory, target_trajectory))

        # Initialize target_position with first point of trajectory
        target_position = target_trajectory[0]

    if LETTER:
        letter_choice = 'W'  # U, N, S, or W

        if letter_choice == 'U':
            target_trajectory = letter_U()
        elif letter_choice == 'N':
            target_trajectory = letter_N()
        elif letter_choice == 'S':
            target_trajectory = letter_S()
        elif letter_choice == 'W':
            target_trajectory = letter_W()

        target_position = target_trajectory[0]
        print(f"Target trajectory (letter {letter_choice}): {target_trajectory}")


    # MPC control loop
    def mpc_control_loop(initial_target_position, target_trajectory_param=None):
        """Run MPC control loop"""
        print("Starting MPC control loop...")
        
        # Initialize target position for this function
        current_target_position = initial_target_position.copy()
        
        dt = MPCConfig.DT
        T = MPCConfig.SIM_TIME
        control_rate = int(1.0 / dt)
        max_iterations = int(T * control_rate)

        line_step = max_iterations // len(target_trajectory_param) if LINE and target_trajectory_param is not None else 1
        
        for i in range(max_iterations):
            try:
                # Get current position for feedback
                if controller_instance.tracker is not None:
                    current_pos = controller_instance.tracker.get_position()
                    current_pos_mm = (current_pos - controller_instance.init_coordinates) * config.px_to_mm
                else:
                    current_pos_mm = np.array([0.0, 0.0])

                # Determine target position
                if LINE and target_trajectory_param is not None:
                    trajectory_index = min(i // line_step, len(target_trajectory_param) - 1)
                    current_target_position = target_trajectory_param[trajectory_index]

                elif CIRCLE:
                    current_target_position = target_trajectory[i % len(target_trajectory)]

                elif LETTER:
                    current_target_position = target_trajectory[i % len(target_trajectory)]

                # Get optimal control from MPC with current position feedback
                u_optimal = mpc_controller.step(current_target_position, current_pos_mm)
                
                # Convert to integer PWM values and send to Arduino
                pwm_values = [int(round(u)) for u in u_optimal]
                command = ",".join(str(pwm) for pwm in pwm_values)
                
                controller_instance.send_arduino(command)
                time.sleep(1)
                
                # Logging current position
                if controller_instance.tracker is not None:
                    error = np.linalg.norm(current_pos_mm - current_target_position)

                    if i % 1 == 0:  # Print every 5 steps
                        print(f"Step {i+1}/{max_iterations}: Sending PWM: {pwm_values} Current pos: [{current_pos_mm[0]:.2f}, {current_pos_mm[1]:.2f}] mm, Error: {error:.3f} mm")
                
                # Store data for MPC history (use consistent scaling)
                if controller_instance.tracker is not None:
                    mpc_controller.history_y.append(current_pos_mm)
                    mpc_controller.history_u.append(u_optimal)
                
                time.sleep(dt)
                
            except KeyboardInterrupt:
                print("\nControl loop interrupted by user.")
                break
            except Exception as e:
                print(f"Error in MPC control loop: {e}")
                break
        
        # Return to zero position
        print("Returning to zero position...")
        zero_command = "0,0,0,0"
        controller_instance.send_arduino(zero_command)
        
        print("MPC control loop completed.")

    # Run MPC control in a separate thread
    trajectory_param = target_trajectory if LINE else None
    mpc_thread = threading.Thread(target=mpc_control_loop, args=(target_position, trajectory_param), daemon=False)
    mpc_thread.start()

    # Keep main thread alive and wait for threads to complete
    print("MPC system running. Press Ctrl+C to stop.")
    
    # Wait for MPC thread to finish
    mpc_thread.join()
    
    # Stop tracker
    tracker_instance.quit = True
    tracker_thread.join()
    
    # Plot MPC results
    print("Plotting MPC results...")
    mpc_controller.plot_results()

    # Plot 2D trajectory and target trajectory
    import matplotlib.pyplot as plt

    if mpc_controller.history_y:
        # Extract x and y coordinates from history
        actual_trajectory = np.array(mpc_controller.history_y)

        # Smooth actual trajectory
        actual_trajectory = gaussian_filter1d(actual_trajectory, sigma=2, axis=0)

        plt.figure(figsize=(10, 6))

        # Plot actual trajectory points
        plt.scatter(actual_trajectory[:, 0], actual_trajectory[:, 1], c='b', label='Actual Trajectory', s=30)

        # Plot target trajectory points
        if STEP:
            plt.scatter(target_position[0], target_position[1], c='r', s=80, marker='o', label='Target Position')
        else:
            plt.scatter(target_trajectory[:, 0], target_trajectory[:, 1], c='r', label='Target Trajectory', s=30, marker='o')
        # Mark start and end points
        plt.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], c='g', s=60, marker='o', label='Start')
        plt.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], c='k', s=60, marker='o', label='End')

        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.title('2D Trajectory Tracking')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.savefig('results/trajectory.png')
        print("Results saved to results/trajectory.png")
        plt.show()



    
    print("\nMPC experiment completed successfully.")

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
