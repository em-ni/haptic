from src import config, tracker
from src import controller
from src.mpc import MPCController, MPCConfig
import threading
import time
import signal
import sys
import numpy as np

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    tracker_instance.quit = True
    controller_instance.close()
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

try:
    # Start tracker in a separate thread
    tracker_instance = tracker.Tracker(cam_index=2)
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

    # Define target trajectory (simple step reference)
    target_position = np.array([-0.9, -1.5])  # Target px, py in mm
    print(f"Target position: {target_position} mm")

    # MPC control loop
    def mpc_control_loop():
        """Run MPC control loop"""
        print("Starting MPC control loop...")
        
        dt = MPCConfig.DT
        T = MPCConfig.SIM_TIME
        control_rate = int(1.0 / dt)
        max_iterations = int(T * control_rate)
        
        for i in range(max_iterations):
            try:
                # Get current position for feedback
                if controller_instance.tracker is not None:
                    current_pos = controller_instance.tracker.get_position()
                    current_pos_mm = (current_pos - controller_instance.init_coordinates) * config.px_to_mm
                else:
                    current_pos_mm = np.array([0.0, 0.0])
                
                # Get optimal control from MPC with current position feedback
                u_optimal = mpc_controller.step(target_position, current_pos_mm)
                
                # Convert to integer PWM values and send to Arduino
                pwm_values = [int(round(u)) for u in u_optimal]
                command = ",".join(str(pwm) for pwm in pwm_values)
                
                print(f"Step {i+1}/{max_iterations}: Sending PWM: {pwm_values}")
                controller_instance.send_arduino(command)
                
                # Logging current position
                if controller_instance.tracker is not None:
                    error = np.linalg.norm(current_pos_mm - target_position)
                    
                    if i % 5 == 0:  # Print every 5 steps
                        print(f"  Current pos: [{current_pos_mm[0]:.2f}, {current_pos_mm[1]:.2f}] mm, Error: {error:.3f} mm")
                
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
    mpc_thread = threading.Thread(target=mpc_control_loop, daemon=False)
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
