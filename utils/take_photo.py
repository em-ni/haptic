import cv2
import os
import platform
import argparse
import time
import threading
import depthai as dai
import numpy as np

# Threaded OAK camera stream (Reused from tracker.py)
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
        pass

def take_photo(cam_index, use_oak=False):
    save_dir = os.path.join(".", "data", "test")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if use_oak:
        print("Initializing OAK Camera...")
        cap = OakCameraStream()
        # Give it some time to warm up and get a frame
        time.sleep(2) 
    else:
        # Detect OS
        print(f"Initializing Webcam {cam_index}...")
        system = platform.system()
        if system == "Windows":
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:  # Linux, macOS, etc.
            cap = cv2.VideoCapture(cam_index)

        if not cap.isOpened():
            print("Error: Cannot access the camera")
            return

    # For OAK, read is custom, for cv2 it's standard.
    # To make it uniform, we can use a small wrapper or just if/else depending on object type,
    # but since OakCameraStream has a .read() method that mimics cv2 (mostly), checking it works.
    # OakCameraStream.read() returns (ret, frame).
    
    # However, standard cv2.VideoCapture doesn't need sleep usually, but reading a few frames helps auto-exposure.
    
    print("Capturing frame...")
    ret, frame = cap.read()
    
    # Retry a few times if using OAK and frame is empty initially
    if use_oak and not ret:
         for _ in range(10):
             time.sleep(0.1)
             ret, frame = cap.read()
             if ret: break

    if not ret:
        print("Error: Could not read frame")
        cap.release()
        if not use_oak:
            cv2.destroyAllWindows()
        return

    photo_path = os.path.join(save_dir, "photo.png")
    cv2.imwrite(photo_path, frame)
    print(f"Photo saved at {photo_path}")

    cap.release()
    if not use_oak:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take a photo with Webcam or OAK Camera")
    parser.add_argument("--oak", action="store_true", help="Use Oak camera")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    take_photo(cam_index=args.cam, use_oak=args.oak)
