import threading
import time
import cv2
import numpy as np
import platform
from matplotlib.animation import FuncAnimation
try:
    import config
except ImportError:
    import src.config as config

# Threaded camera stream for fast frame grabbing
class CameraStream:
    def __init__(self, cam_index):
        system = platform.system()
        if system == "Windows":
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(cam_index)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


class Tracker:
    def __init__(self, cam_index):
        self.cam_index = cam_index

        # Smoothing factor for the exponential moving average filter (lower is smoother)
        self.alpha = 0.2

        # Tracking variables
        self.cur_pos = None
        self.prev_pos = None
        self.cur_vel = None
        self.prev_vel = None

        self.quit = False

    def detect_tip(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the green tip of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make masks for green color
        mask_green = cv2.inRange(hsv, config.lower_green, config.upper_green)

        # Find contours in the mask.
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("WARNING: No green tip detected.")
            return None

        # Choose the largest contour.
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        return cx, cy
    
    def filter_value(self, pre_filtered, new_value, alpha):
        """
        General exponential moving average filter for any value (array or scalar).
        """
        if pre_filtered is None:
            return new_value
        else:
            return alpha * new_value + (1 - alpha) * pre_filtered
        
    def get_position(self):
        return self.cur_pos

    def get_velocity(self):
        return self.cur_vel

    def track(self):
        """
        Main function to track the tip position and velocity.
        """
        cam = CameraStream(self.cam_index)
        time.sleep(1)

        # Check if camera opened successfully
        if not cam.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        start_time = time.time()
        frame_count = 0
        while not self.quit:
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Detect the tip position
            pos = self.detect_tip(frame)
            if pos is not None:
                self.cur_pos = np.array(pos)
                # Filter the position
                self.cur_pos = self.filter_value(self.prev_pos, self.cur_pos, alpha=self.alpha)

                # Calculate velocity
                if self.prev_pos is not None:
                    dt = time.time() - start_time
                    velocity = (self.cur_pos - self.prev_pos) / dt
                    self.cur_vel = self.filter_value(self.prev_vel, velocity, alpha=self.alpha)
                    self.prev_vel = self.cur_vel

                self.prev_pos = self.cur_pos
                start_time = time.time()

                # Draw the detected tip on the frame
                cv2.circle(frame, (int(self.cur_pos[0]), int(self.cur_pos[1])), 5, (0, 255, 0), -1)
                if self.cur_pos is not None:
                    cv2.putText(frame, f"Pos: ({int(self.cur_pos[0])}, {int(self.cur_pos[1])})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.cur_vel is not None:
                    cv2.putText(frame, f"Vel: {self.cur_vel}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Tracking", frame)
            frame_count += 1

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        elapsed_time = time.time() - start_time

    def draw(self):
        """
        Draw in real time the current position on a 2D plane
        """

        import matplotlib.pyplot as plt

        # Set up the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 640)  # Adjust based on your camera resolution
        ax.set_ylim(0, 480)  # Adjust based on your camera resolution
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Real-time Tip Position')
        ax.invert_yaxis()  # Invert y-axis to match camera coordinates

        # Initialize empty line for trajectory
        line, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1)
        point, = ax.plot([], [], 'ro', markersize=8)

        # Store trajectory points
        trajectory_x = []
        trajectory_y = []

        def update_plot(frame):
            if self.cur_pos is not None:
                # Add current position to trajectory
                trajectory_x.append(self.cur_pos[0])
                trajectory_y.append(self.cur_pos[1])
                
                # Keep only last 100 points for performance
                if len(trajectory_x) > 100:
                    trajectory_x.pop(0)
                    trajectory_y.pop(0)
                
                # Update line and point
                line.set_data(trajectory_x, trajectory_y)
                point.set_data([self.cur_pos[0]], [self.cur_pos[1]])
            
            return line, point

        # Start camera stream
        cam = CameraStream(self.cam_index)
        time.sleep(1)

        if not cam.cap.isOpened():
            print("Error: Could not open camera.")
            return

        # Start tracking in a separate thread
        def track_position():
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                
                pos = self.detect_tip(frame)
                if pos is not None:
                    self.cur_pos = np.array(pos)
                    self.cur_pos = self.filter_value(self.prev_pos, self.cur_pos, alpha=0.2)
                    self.prev_pos = self.cur_pos

        tracking_thread = threading.Thread(target=track_position, daemon=True)
        tracking_thread.start()

        # Create animation
        ani = FuncAnimation(fig, update_plot, interval=50, blit=True)
        plt.show()

        cam.release()

if __name__ == "__main__":
    cam_index = 6  # Change this based on your camera index
    tracker = Tracker(cam_index)
    tracker.track()
    tracker.draw()