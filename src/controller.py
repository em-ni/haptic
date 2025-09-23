import time
import serial
import random
import numpy as np

class Controller:
    def __init__(self, arduino_port):
        self.arduino = serial.Serial(arduino_port, 9600)
        # Allow Arduino to initialize
        time.sleep(2)  

    def close(self):
        if self.arduino.is_open:
            self.arduino.close()
        else:
            print("Arduino port is already closed.")
        time.sleep(0.1)
        print("Arduino connection closed.")

    def generate_random_trajectory(self, N):
        """
        Generate random trajectories for four motors, each in [-255,-150] U [150,255] skipping the dead zone [-149,149]. 
        Ensures that at each point, the four motors do not have the same sign.
        """
        valid_ranges = [(-255, -150), (150, 255)]
        
        while True:  # keep trying until we get valid trajectories
            trajectories = []
            for motor_idx in range(4):
                start_range = random.choice(valid_ranges)
                end_range = random.choice(valid_ranges)
                start = random.randint(start_range[0], start_range[1])
                end = random.randint(end_range[0], end_range[1])
                traj = self.interpolate_skip_deadzone(start, end, N)
                trajectories.append(traj)

            # Check sign constraint at every point
            valid = True
            for i in range(N):
                values = [trajectories[m][i] for m in range(4)]
                signs = [v > 0 for v in values]
                if all(signs) or not any(signs):
                    valid = False
                    break

            if valid:
                return trajectories
            # If not valid, try again
            print("Generated trajectories are not valid. Retrying...")

    def interpolate_skip_deadzone(self, start, end, N):
        """
        Interpolate N points from start to end, skipping dead zone [-149,149].
        """
        points = []
        # If start and end on same side, linear interpolation normally
        if (start < -149 and end < -149) or (start > 149 and end > 149):
            points = np.linspace(start, end, N, dtype=int).tolist()
        else:
            # Jump across dead zone
            # Determine edges
            left_edge = -150
            right_edge = 150

            # Distance before dead zone and after dead zone
            dist1 = abs(start - (left_edge if start < 0 else right_edge))
            dist2 = abs(end - (right_edge if start < 0 else left_edge))
            total_dist = dist1 + dist2

            N1 = max(1, int(N * dist1 / total_dist))
            N2 = N - N1

            # First half
            first_half = np.linspace(start, left_edge if start < 0 else right_edge, N1, dtype=int).tolist()
            # Second half
            second_half = np.linspace(right_edge if start < 0 else left_edge, end, N2, dtype=int).tolist()

            points = first_half + second_half

        return points

    def send_arduino(self, command):
        if self.arduino.is_open:
            self.arduino.write(command.encode())
        else:
            print("Error: Arduino port is not open.")
        time.sleep(0.1)

    def send_trajectory(self, trajectories, delay):
        N = len(trajectories[0])
        for i in range(N):
            command = ",".join(str(trajectories[m][i]) for m in range(4))
            # print(f"Sending command: {command}")
            self.send_arduino(command)
            time.sleep(delay)

    def workspace_sweep(self, step, delay):
        """
        Sweep the workspace by moving each motor from -255 to 255 in steps, while keeping others constant and avoiding the dead zone.
        Ensures that at each point, the four motors do not have the same sign.
        """

        def sweep_motor(m_id, cur_positions, step, direction):
            # negative to positive
            if direction == "np":  
                positions = list(range(-255, -149, step)) + list(range(150, 256, step))
            # positive to negative
            else:  
                positions = list(range(255, 149, -step)) + list(range(-150, -256, -step))

            for pos in positions:
                cur_positions[m_id-1] = pos
                command = ",".join(str(p) for p in cur_positions)
                print(f"Sending command: {command}")
                self.send_arduino(command)
                time.sleep(delay)

        # Order of motor movements
        m_ids = [1,3,2,4,3,1,4]

        # Initial position
        init_p = [255,255,-255,-255]
        init_command = ",".join(str(p) for p in init_p)
        print(f"Sending initial command: {init_command}")
        self.send_arduino(init_command)
        print("Initial position set. Waiting for 2 seconds...")
        time.sleep(2)

        # Current positions
        curr_p = init_p.copy()

        for m_id in m_ids:
            print(f"\nMoving motor {m_id}...")
            direction = "np" if curr_p[m_id-1] < 0 else "pn"
            sweep_motor(m_id, curr_p, step, direction)
            time.sleep(2)

            


if __name__ == "__main__":
    arduino_port = "/dev/ttyUSB0"  
    controller = Controller(arduino_port)
    random_trajectory = False

    if random_trajectory == True:
        # Random trajectory
        trajectories = controller.generate_random_trajectory(N=100)

        # Go to the first position immediately
        initial_command = ",".join(str(traj[0]) for traj in trajectories)
        print(f"Sending initial command: {initial_command}")
        controller.send_arduino(initial_command)
        print("Initial position set. Waiting for 2 seconds...")
        time.sleep(2)

        for idx, traj in enumerate(trajectories):
            print(f"Motor {idx+1} trajectory: {traj}")

        print("\nSending trajectories to Arduino...")
        controller.send_trajectory(trajectories, delay=0.1)

        # Go back to the inital position (all zeros) interpolating from the last position
        print("\nReturning to initial position...")
        final_trajectories = []
        for traj in trajectories:
            final_trajectories.append(controller.interpolate_skip_deadzone(traj[-1], 0, 100))
        controller.send_trajectory(final_trajectories, delay=0.1)

    else:
        # Workspace sweep
        print("Starting workspace sweep...")
        controller.workspace_sweep(step=50, delay=0.2)

    print("Done.")

    controller.close()
