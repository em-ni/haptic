from src import tracker
from src import controller
import threading
import time
import signal
import sys

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
    time.sleep(2)

    # Initialize controller with tracker instance
    controller_instance = controller.Controller(arduino_port="/dev/ttyUSB0", tracker=tracker_instance)

    # Sweep workspace in a separate thread
    # sweep_thread = threading.Thread(target=controller_instance.polygon_sweep, args=(50, 0.2, 255), daemon=False)
    sweep_thread = threading.Thread(target=controller_instance.concentric_polygons_sweep, args=([255, 220, 185, 150],), daemon=False)
    sweep_thread.start()

    # Keep main thread alive and wait for threads to complete
    print("System running. Press Ctrl+C to stop.")
    
    # Wait for threads to finish
    sweep_thread.join()
    
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
