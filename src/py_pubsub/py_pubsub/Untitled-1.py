#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import math as m

def scale(angle):
    return (angle + m.pi) % (2 * m.pi) - m.pi

class EKF_SLAM(Node):
    def __init__(self):
        super().__init__('ekf_slam')
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/filtered_data', 10)
        self.timer = self.create_timer(0.25, self.timer_cb)

        # --- Initial State Setup ---
        # Start with robot pose only: [x, y, theta]
        self.x_prev, self.y_prev, self.theta_prev = 0.0, 0.0, 0.0
        self.Mu = np.zeros((3, 1))
        self.Sigma = np.eye(3) * 0.01

        # Landmarks: will be appended as (x, y) tuples.
        self.landmarks = []  
        self.state_size = 3  # Dynamic state size: 3 + 2*num_landmarks

        # --- Noise Parameters ---
        # Motion noise parameters
        self.alpha1, self.alpha2 = 0.005, 0.05
        self.alpha3, self.alpha4 = 0.005, 0.05
        # Measurement noise parameters (for [range, bearing])
        self.sigma_r, self.sigma_phi = 0.1, 0.01
        self.Qt = np.diag([self.sigma_r**2, self.sigma_phi**2])

    # --- Odometry Callback ---
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        _, _, self.theta = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        # Compute control inputs from odometry
        self.Dt = m.sqrt((self.x - self.x_prev)**2 + (self.y - self.y_prev)**2)
        self.Dr1 = scale(m.atan2(self.y - self.y_prev, self.x - self.x_prev) - self.theta_prev)
        self.Dr2 = scale(self.theta - self.theta_prev - self.Dr1)

    # --- Laser Scan Callback ---
    def scan_callback(self, scan):
        # Extract range measurements; simple feature extraction using differences
        self.x_scan = [0.0]*len(scan.ranges)
        self.diff = [0.0]*len(scan.ranges)
        for i in range(len(scan.ranges)):
            if m.isinf(scan.ranges[i]) or m.isnan(scan.ranges[i]):
                self.x_scan[i] = scan.range_max
            else:
                self.x_scan[i] = scan.ranges[i]
            if i > 0 and self.x_scan[i] > 0.2 and self.x_scan[i-1] > 0.2:
                diff_val = self.x_scan[i] - self.x_scan[i-1]
            else:
                diff_val = 0.0
            self.diff[i] = diff_val if abs(diff_val) > 0.15 else 0.0

        # Extract potential landmark measurements from the scan.
        self.lidar_dist, self.lidar_angle = [], []
        obj_detected = False
        indices, angle_sum, range_sum, j = 0, 0, 0, -1
        while j < len(self.diff) - 1:
            j += 1
            if self.diff[j] < 0 and not obj_detected:
                obj_detected = True
                indices += 1
                angle_sum += j
                range_sum += self.x_scan[j]
            elif self.diff[j] < 0 and obj_detected:
                obj_detected = False
                indices, angle_sum, range_sum = 0, 0, 0
                j -= 1
            elif self.diff[j] == 0 and obj_detected:
                indices += 1
                angle_sum += j
                range_sum += self.x_scan[j]
            elif self.diff[j] > 0 and obj_detected:
                obj_detected = False
                if indices > 0:
                    avg_angle = (angle_sum * m.pi / 180) / indices
                    self.lidar_angle.append(scale(avg_angle))
                    self.lidar_dist.append(range_sum / indices)
                indices, angle_sum, range_sum = 0, 0, 0
        self.global_coord_conv()

    # --- Global Coordinate Conversion ---
    def global_coord_conv(self):
        detected_landmarks = []
        for i in range(len(self.lidar_dist)):
            # Transform the local measurement (from the previous robot pose) to global coordinates.
            lx = self.x_prev + self.lidar_dist[i] * m.cos(scale(self.theta_prev + self.lidar_angle[i]))
            ly = self.y_prev + self.lidar_dist[i] * m.sin(scale(self.theta_prev + self.lidar_angle[i]))
            detected_landmarks.append((lx, ly))
        self.update_landmarks(detected_landmarks)

    def update_landmarks(self, detected_landmarks):
        # If no landmarks are detected, reset to robot-only state.
        if len(detected_landmarks) == 0:
                #self.get_logger().info("No landmarks detected. Resetting to robot-only state.")
                self.landmarks = []
                self.state_size = 3
                self.Mu = self.Mu[:3, :]
                self.Sigma = self.Sigma[:3, :3]
                return
        # add new landmarks if not already present.
        for (lx, ly) in detected_landmarks:
            # Check if the detected landmark is already in the list (within a threshold)
            if not any(m.sqrt((lx - Lx)**2 + (ly - Ly)**2) < 0.5 for (Lx, Ly) in self.landmarks):
                self.landmarks.append((lx, ly))
                #print(f"New landmark detected: ({lx:.2f}, {ly:.2f})")
                self.extend_state()

    def extend_state(self):
        # Increase the state vector by 2 for each new landmark.
        self.state_size = 3 + 2 * len(self.landmarks)
        self.Mu = np.vstack((self.Mu, np.zeros((2, 1))))
        new_Sigma = np.zeros((self.state_size, self.state_size))
        new_Sigma[:self.Sigma.shape[0], :self.Sigma.shape[1]] = self.Sigma
        new_Sigma[-2:, -2:] = np.eye(2) * 1e6  # High initial uncertainty for new landmarks.
        self.Sigma = new_Sigma
        #print(f"Extended state size: {self.state_size}")
        #print(f"Extended state vector: {self.Sigma}")


     # --- EKF Prediction and Correction ---
    def timer_cb(self):
        self.motion_update()
        self.state_covariance()
        self.control_covariance()
        self.process_noise()
        self.correction()
#         self.publish_state()

    def motion_update(self):
        # Compute control inputs from odometry (only update robot pose; landmarks remain unchanged)
        dx = self.Dt * m.cos(self.theta_prev + self.Dr1)
        dy = self.Dt * m.sin(self.theta_prev + self.Dr1)
        dtheta = self.Dr1 + self.Dr2
        self.Mu[0, 0] += dx
        self.Mu[1, 0] += dy
        self.Mu[2, 0] = scale(self.Mu[2, 0] + dtheta)
        #print(f"Mu: {self.Sigma}")

    def state_covariance(self):
        # Build Jacobian Gt for the full state.
        Gt = np.eye(self.state_size)
        Gt[0, 2] = -self.Dt * m.sin(self.theta_prev + self.Dr1)
        Gt[1, 2] = self.Dt * m.cos(self.theta_prev + self.Dr1)
        self.Sigma1 = Gt @ self.Sigma @ Gt.T

    def control_covariance(self):
        # Build the control Jacobian (v) and append zeros for landmarks.
        v = np.array([
            [-self.Dt * m.sin(self.theta_prev + self.Dr1), m.cos(self.theta_prev + self.Dr1), 0],
            [self.Dt * m.cos(self.theta_prev + self.Dr1), m.sin(self.theta_prev + self.Dr1), 0],
            [1, 0, 1]
        ])
        n_landmarks = (self.Mu.shape[0] - 3) // 2
        lm_control = np.zeros((2 * n_landmarks, 3))
        self.Vt = np.block([[v], [lm_control]])
        print(f"Vt: {self.Vt}")

    def process_noise(self):
        # Compute process noise covariance.
        min_noise = 1e-6
        self.Mt = np.array([
            [max((self.alpha1 * self.Dr1**2) + (self.alpha2 * self.Dt**2), min_noise), 0, 0],
            [0, max((self.alpha3 * self.Dt**2) + (self.alpha4 * (self.Dr2**2 + self.Dr1**2)), min_noise), 0],
            [0, 0, max((self.alpha1 * self.Dr1**2) + (self.alpha2 * self.Dt**2), min_noise)]
        ])
        self.Rt = self.Vt @ self.Mt @ self.Vt.T
        self.Sigma  = self.Sigma1 + self.Rt
        #print(f"Sigma: {self.Sigma}")
    
    def correction(self):
        # For each landmark in the state, try to associate a measurement.
        for i in range(len(self.landmarks)):
            # Landmark state is stored at indices 3+2*i and 4+2*i in Mu.
            landmark_state = self.Mu[3+2*i:3+2*i+2, 0]  # [l_ix, l_iy]
            robot_state = self.Mu[0:3, 0]  # [x, y, theta]
            delta_x = landmark_state[0] - robot_state[0]
            delta_y = landmark_state[1] - robot_state[1]
            q = delta_x**2 + delta_y**2
            if q == 0:
                continue
            range_pred = m.sqrt(q)
            bearing_pred = scale(m.atan2(delta_y, delta_x) - robot_state[2])
            z_hat = np.array([[range_pred], [bearing_pred]])

            best_j = None
            best_dist = float('inf')
            for j in range(len(self.lidar_dist)):
                z_meas = np.array([[self.lidar_dist[j]], [self.lidar_angle[j]]])
                innovation = z_meas - z_hat
                dist = float(innovation.T @ np.linalg.inv(self.Qt) @ innovation)
                #self.get_logger().info(f"Landmark {i}, meas {j}: Mahalanobis dist = {dist:.2f}")
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            threshold = 16.0  # Adjust threshold as needed
            #self.get_logger().info(f"Landmark {i}: best_meas = {best_j}, best_dist = {best_dist:.2f}, threshold = {threshold}")
            if best_j is not None and best_dist < threshold:
                z = np.array([[self.lidar_dist[best_j]], [self.lidar_angle[best_j]]])
                H = np.zeros((2, self.state_size))
                # Robot portion:
                H[0, 0] = -delta_x / range_pred
                H[0, 1] = -delta_y / range_pred
                H[1, 0] = delta_y / q
                H[1, 1] = -delta_x / q
                H[1, 2] = -1
                # Landmark portion:
                H[0, 3+2*i] = delta_x / range_pred
                H[0, 4+2*i] = delta_y / range_pred
                H[1, 3+2*i] = -delta_y / q
                H[1, 4+2*i] = delta_x / q

                S = H @ self.Sigma @ H.T + self.Qt
                K = self.Sigma @ H.T @ np.linalg.inv(S)
                self.Mu = self.Mu + K @ (z - z_hat)
                self.Mu[2, 0] = scale(self.Mu[2, 0])
                self.Sigma = (np.eye(self.state_size) - K @ H) @ self.Sigma
            else:
                self.get_logger().info(f"Landmark {i} not updated; best_dist = {best_dist:.2f}")
            print(f"Mu after correction: {self.Mu}")

        # Update landmarks in the state vector for the next iteration.
        for i in range(len(self.landmarks)):
            self.landmarks[i] = (self.Mu[3+2*i, 0], self.Mu[4+2*i, 0])
        #self.get_logger().info(f"Updated landmarks: {self.landmarks}")

    def publish_state(self):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.pose.position.x = self.Mu[0, 0]
        pose_msg.pose.pose.position.y = self.Mu[1, 0]
        q = quaternion_from_euler(0, 0, self.Mu[2, 0])
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]
        self.pub.publish(pose_msg)
        print(f"Published Pose: x={self.Mu[0,0]:.2f}, y={self.Mu[1,0]:.2f}, theta={self.Mu[2,0]:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = EKF_SLAM()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()