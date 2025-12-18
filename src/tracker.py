import threading
import time
import cv2
import numpy as np
import platform
import argparse
import depthai as dai
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


# Threaded OAK camera stream
class OakCameraStream:
    def __init__(self):
        # Initialize device first
        self.device = dai.Device()
        self.pipeline = dai.Pipeline(self.device)
        
        # Define sources and outputs
        self.cam = self.pipeline.create(dai.node.Camera)
        
        # Configure Camera
        # Try to find Cam A properties
        found_cam_a = False
        for sensor in self.device.getConnectedCameraFeatures():
            if sensor.socket == dai.CameraBoardSocket.CAM_A:
                self.cam.build(sensor.socket)
                found_cam_a = True
                break
        
        if not found_cam_a:
             # Fallback
             self.cam.build(dai.CameraBoardSocket.CAM_A)
        
        # Request output (NV12 enables getCvFrame)
        self.cam_out = self.cam.requestOutput((1920, 1080), dai.ImgFrame.Type.NV12)
        
        # Create output queue
        self.video_q = self.cam_out.createOutputQueue(maxSize=1, blocking=False)
        
        # Start pipeline
        self.pipeline.start()
        
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            in_video = self.video_q.tryGet()
            
            if in_video is not None:
                frame = in_video.getCvFrame()
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)
    
    def release(self):
        self.running = False
        self.thread.join()
        # No explicit device close method in some versions, but pipeline cleanups automatically
        # or we can try closing if method exists
        pass


class Tracker:
    def __init__(self, cam_index=0, use_oak=True, video_output='video.mp4'):
        self.cam_index = cam_index
        self.use_oak = use_oak
        self.video_output = video_output

        # Smoothing factor for the exponential moving average filter (lower is smoother)
        self.alpha = 0.3

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
        if self.use_oak:
            cam = OakCameraStream()
        else:
            cam = CameraStream(self.cam_index)
        time.sleep(1)

        # Check if camera opened successfully
        if not self.use_oak and not cam.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        start_time = time.time()
        frame_count = 0
        video_writer = None

        while not self.quit:
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Save raw video if requested
            if self.video_output is not None:
                if video_writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(self.video_output, fourcc, 30.0, (width, height))
                video_writer.write(frame)

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

        if video_writer is not None:
            video_writer.release()
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
        if self.use_oak:
            cam = OakCameraStream()
        else:
            cam = CameraStream(self.cam_index)
        time.sleep(1)

        if not self.use_oak and not cam.cap.isOpened():
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
    parser = argparse.ArgumentParser(description="Tip Tracker")
    parser.add_argument("--oak", action="store_true", help="Use Oak camera (Cam A)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--save_video", type=str, default=None, help="Path to save raw video (e.g., output.mp4)")
    args = parser.parse_args()

    tracker = Tracker(cam_index=args.cam, use_oak=args.oak, video_output=args.save_video)
    tracker.track()
    tracker.draw()