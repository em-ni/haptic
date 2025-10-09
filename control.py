from src import config, tracker
from src import controller
from src.mpc import MPCController, MPCConfig
import threading
import time
import signal
import sys
import numpy as np

STEP = False
LINE = True
CIRCLE = False

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
        line_start = np.array([0.7, -0.7])
        line_end = np.array([-0.7, 0.7])
        target_trajectory = line_trajectory(line_start, line_end, N=100)
        print(f"Target trajectory (line): {target_trajectory} mm")
        # Initialize target_position with first point of trajectory
        target_position = target_trajectory[0]

    if CIRCLE:
        # Define target trajectory (circle reference, 2mm diameter = 1mm radius)
        circle_center = np.array([0.0, 0.0])  # Center at origin
        circle_radius = 0.9  # mm
        target_trajectory = circle_trajectory(circle_center, circle_radius, N=40)
        print(f"Target trajectory (circle): {target_trajectory} mm")

        # Line from center to first point of circle
        line_start = np.array([0.0, 0.0])
        line_end = target_trajectory[0]
        approach_trajectory = line_trajectory(line_start, line_end, N=10)

        # Prepend approach trajectory to circle trajectory
        target_trajectory = np.vstack((approach_trajectory, target_trajectory))
        # Initialize target_position with first point of trajectory
        target_position = target_trajectory[0]

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

                # Get optimal control from MPC with current position feedback
                u_optimal = mpc_controller.step(current_target_position, current_pos_mm)
                
                # Convert to integer PWM values and send to Arduino
                pwm_values = [int(round(u)) for u in u_optimal]
                command = ",".join(str(pwm) for pwm in pwm_values)
                
                controller_instance.send_arduino(command)
                time.sleep(3)
                
                # Logging current position
                if controller_instance.tracker is not None:
                    error = np.linalg.norm(current_pos_mm - current_target_position)

                    if i % 5 == 0:  # Print every 5 steps
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

        plt.figure(figsize=(10, 6))

        # Plot actual trajectory points
        plt.scatter(actual_trajectory[:, 0], actual_trajectory[:, 1], c='b', label='Actual Trajectory', s=30)

        # Plot target trajectory points
        if LINE or CIRCLE:
            plt.scatter(target_trajectory[:, 0], target_trajectory[:, 1], c='r', label='Target Trajectory', s=30, marker='o')
        elif STEP:
            plt.scatter(target_position[0], target_position[1], c='r', s=80, marker='o', label='Target Position')

        # Mark start and end points
        plt.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], c='g', s=60, marker='o', label='Start')
        plt.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], c='k', s=60, marker='o', label='End')

        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.title('2D Trajectory Tracking')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
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
