from controller import Robot, Camera
import numpy as np
from PIL import Image
import time
import requests

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

class Mavic(Robot):
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    K_VERTICAL_OFFSET = 0.6  # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 30.0          # P constant of the pitch PID.
    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    target_precision = 0.5  # Precision between the target position and the robot position in meters

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)
        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0
        self.last_image_capture_time = 0

    def set_position(self, pos):
        self.current_pose = pos

    def capture_image(self):
        image = self.camera.getImage()
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        pil_image = Image.fromarray(image_array, 'RGBA')
        return pil_image

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=False):
        if self.target_position[0:2] == [0, 0]:  # Initialization
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):
            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ", self.target_position[0:2])
                self.target_altitude = 0

        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        angle_left = self.target_position[2] - self.current_pose[5]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance

    def run(self):
        t1 = self.getTime()
        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0
        waypoints = [[0, 0], [0, 20], [20, 20], [20, 0], [0, 0]]
        self.target_altitude = 1

        while self.step(self.time_step) != -1:
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])
            if altitude > self.target_altitude - 1:
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance = self.move_to_target(
                        waypoints)
                    t1 = self.getTime()

            roll_input = self.K_ROLL_P * \
                clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * \
                clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(
                self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * \
                pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input - yaw_input - pitch_input + roll_input

            print(vertical_input, yaw_input, pitch_input, roll_input)
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)

            current_time = time.time()
            if current_time - self.last_image_capture_time > 1:
                captured_image = self.capture_image()
                file_path = "captured_image.png"
                captured_image.save(file_path)
                self.last_image_capture_time = current_time

                # Send the image to the Flask app
                url = 'http://127.0.0.1:5000/predict'
                files = {'file': open(file_path, 'rb')}
                response = requests.post(url, files=files)
                print('Prediction:', response.text)

robot = Mavic()
robot.run()
