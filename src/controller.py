import csv
import time
import serial
import random
import numpy as np
try:
    import config as config
except ImportError:
    import src.config as config

debug = config.debug

class Controller:
    def __init__(self, arduino_port, tracker):
        self.arduino = serial.Serial(arduino_port, 9600)
        # Allow Arduino to initialize
        time.sleep(2)

        # Initialize tracker
        self.tracker = tracker
        self.save_data = False

        # Init csv to log positions and velocities if tracker is provided
        if self.tracker is not None:
            self.save_data = True
            self.log_file = config.csv_path
            self.log_columns = config.csv_columns


    def close(self):
        if self.arduino.is_open:
            self.arduino.close()
        else:
            print("Arduino port is already closed.")
        time.sleep(0.1)
        print("Arduino connection closed.")

    def concentric_polygons_sweep(self, radii):
        for radius in radii:
            if debug: print(f"\nStarting sweep for radius: {radius}")
            self.polygon_sweep(perimeter_step=50, delay=0.2, max_radius=radius)
            time.sleep(2)
            if debug: print(f"Finished sweep for radius: {radius}")
        if debug: print("\nFinished all polygons.")

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
    
    def save_data_row(self, position, velocity, pm_values, vm_values):
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

        with open(self.log_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

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

    def polygon_sweep(self, perimeter_step, delay, max_radius):
        """
        Sweep the workspace by moving each motor from -255 to 255 in perimeter_steps, while keeping others constant and avoiding the dead zone.
        Ensures that at each point, the four motors do not have the same sign.
        """

        def sweep_motor(m_id, cur_positions, perimeter_step, direction, max_radius):
            # negative to positive
            if direction == "np":  
                positions = list(range(-max_radius, -149, perimeter_step)) + list(range(150, max_radius + 1, perimeter_step))
            # positive to negative
            else:  
                positions = list(range(max_radius, 149, -perimeter_step)) + list(range(-150, -max_radius + 1, -perimeter_step))

            for pos in positions:
                cur_positions[m_id-1] = pos
                command = ",".join(str(p) for p in cur_positions)
                if debug: print(f"Sending command: {command}")
                self.send_arduino(command)

                if self.save_data and self.tracker is not None:
                    position = self.tracker.get_position()
                    velocity = self.tracker.get_velocity()
                    pm_values = command.split(",")
                    vm_values = [0, 0, 0, 0]  # Placeholder for velocity measurements
                    self.save_data_row(position, velocity, pm_values, vm_values)
                    if debug: print(f"Logged data: Pos {position}, Vel {velocity}, PM {pm_values}, VM {vm_values}")

                time.sleep(delay)

        # Order of motor movements
        m_ids = [1,3,2,4,3,1,4]

        # Initial position
        init_p = [max_radius,max_radius,-max_radius,-max_radius]
        init_command = ",".join(str(p) for p in init_p)
        print(f"Sending initial command: {init_command}")
        self.send_arduino(init_command)
        print("Initial position set. Waiting for 2 seconds...")
        time.sleep(2)

        # Current positions
        curr_p = init_p.copy()

        for m_id in m_ids:
            if debug: print(f"\nMoving motor {m_id}...")
            direction = "np" if curr_p[m_id-1] < 0 else "pn"
            sweep_motor(m_id, curr_p, perimeter_step, direction, max_radius)
            time.sleep(2)



if __name__ == "__main__":
    arduino_port = "/dev/ttyUSB0"  
    controller = Controller(arduino_port, tracker=None)
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
        controller.polygon_sweep(perimeter_step=50, delay=0.2, max_radius=255)

    print("Done.")

    controller.close()
