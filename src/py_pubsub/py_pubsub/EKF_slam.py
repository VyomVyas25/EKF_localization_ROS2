#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Float32MultiArray
import numpy as np
import math as m


def scale(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + m.pi) % (2 * m.pi) - m.pi


class EKF_SLAM(Node):
    def __init__(self):
        super().__init__('ekf_slam')

        # Subscriptions & publisher
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/filtered_data', 10)
        self.state_pub = self.create_publisher(Float32MultiArray, '/slam_state', 10)
        self.timer = self.create_timer(0.25, self.timer_cb)

        # Initial State
        self.Dt = 0.0
        self.Dr1 = 0.0
        self.Dr2 = 0.0

        self.m_x = 0.0
        self.m_y = 0.0
        self.m_0 = 0.0
        self.m_x_prev = 0.0
        self.m_prev_y = 0.0
        self.m_prev_0 = 0.0

        self.prev_state = None
        self.state_matrix = None
        self.n = 3  # initial state size

        # Noise parameters
        self.sigma_x = 0.1
        self.sigma_y = 0.1
        self.sigma_theta = 0.1

        # Covariance matrix
        self.P = np.eye(self.n)

        # Lidar data storage
        self.lidar_angle = []
        self.lidar_range = []
        self.diff = []
        
        # Initialize predicted measurements array
        self.predicted_measurements = None

    def odom_callback(self, msg: Odometry):
        self.m_x = msg.pose.pose.position.x
        self.m_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.m_0 = scale(yaw)

        self.Dt = m.sqrt((self.m_x - self.m_x_prev) ** 2 + (self.m_y - self.m_prev_y) ** 2)
        self.Dr1 = scale(m.atan2(self.m_y - self.m_prev_y, self.m_x - self.m_x_prev) - self.m_prev_0)
        self.Dr2 = scale(self.m_0 - self.m_prev_0 - self.Dr1)

        # Update prev values for next iteration
        self.m_x_prev = self.m_x
        self.m_prev_y = self.m_y
        self.m_prev_0 = self.m_0

    def update_state(self):
        # Initialize robot state
        robot_state = np.array([[self.m_x], [self.m_y], [self.m_0]])
        
        if self.state_matrix is None:
            # First time - initialize with robot pose only
            self.state_matrix = robot_state.copy()
        else:
            # Update robot pose while preserving landmarks
            current_landmarks = self.state_matrix[3:, :] if self.state_matrix.shape[0] > 3 else np.array([]).reshape(0, 1)
            
            # Combine updated robot pose with existing landmarks
            if current_landmarks.size > 0:
                self.state_matrix = np.vstack([robot_state, current_landmarks])
            else:
                self.state_matrix = robot_state.copy()
        
        # Update state size
        self.n = self.state_matrix.shape[0]

    def state_transition_matrix(self):
        self.Gt = np.array([
            [1, 0, -self.Dt * m.sin(self.m_prev_0 + self.Dr1)],
            [0, 1,  self.Dt * m.cos(self.m_prev_0 + self.Dr1)],
            [0, 0, 1]
        ])

        if self.state_matrix is not None:
            self.n = self.state_matrix.shape[0]
            upper_right = np.zeros((3, self.n - 3))
            bottom_left = np.zeros((self.n - 3, 3))
            bottom_right = np.eye(self.n - 3)
            self.F = np.block([
                [self.Gt, upper_right],
                [bottom_left, bottom_right]
            ])
        else:
            self.n = 3
            self.F = self.Gt

    def control_covariance_matrix(self):
        # V matrix maps control noise to state space
        # Only affects robot pose (first 3 states), landmarks are not affected by control
        self.V = np.zeros((self.n, 3))
        self.V[0, 0] = m.cos(self.m_prev_0 + self.Dr1)
        self.V[0, 1] = -self.Dt * m.sin(self.m_prev_0 + self.Dr1)
        self.V[1, 0] = m.sin(self.m_prev_0 + self.Dr1)
        self.V[1, 1] = self.Dt * m.cos(self.m_prev_0 + self.Dr1)
        self.V[2, 1] = 1  # Dr1 affects robot orientation
        self.V[2, 2] = 1  # Dr2 affects robot orientation

        # Control noise covariance (for motion model uncertainties)
        self.Q = np.diag([
            self.sigma_x ** 2,    # uncertainty in translation
            self.sigma_y ** 2,    # uncertainty in rotation Dr1
            self.sigma_theta ** 2 # uncertainty in rotation Dr2
        ])

        # Process noise covariance in state space
        self.R = self.V @ self.Q @ self.V.T

    def covariance_matrix(self):
        self.P = self.F @ self.P @ self.F.T + self.R
        print(f"Covariance matrix P:\n{self.P}")

    def scan_callback(self, msg: LaserScan):
        self.scan = [0.0] * len(msg.ranges)
        self.diff = [0.0] * len(msg.ranges)
        self.lidar_range = []
        self.lidar_angle = []

        for i in range(len(msg.ranges)):
            if m.isinf(msg.ranges[i]) or m.isnan(msg.ranges[i]):
                self.scan[i]  = 3.5
            else:
                self.scan[i] = msg.ranges[i]

            if i > 0 and self.scan[i] > 0.2 and self.scan[i - 1] > 0.2:
                diff_val = self.scan[i] - self.scan[i - 1]
            else:
                diff_val = 0.0
            self.diff[i] = diff_val if abs(diff_val) > 0.15 else 0.0

    def detect_landmarks(self):
        obj_detected = 0
        add_ranges = 0.0
        indices = 0
        add_index = 0
        
        # Reset landmark lists
        self.lidar_angle = []
        self.lidar_range = []
        
        j = 0
        while j < len(self.diff):
            if self.diff[j] < 0 and obj_detected == 0:
                obj_detected = 1
                indices = 1
                add_index = j
                add_ranges = self.scan[j]
            elif self.diff[j] < 0 and obj_detected == 1:
                obj_detected = 0
                indices = 0
                add_index = 0
                add_ranges = 0.0
                j -= 1
            elif self.diff[j] == 0 and obj_detected == 1:
                indices += 1
                add_ranges += self.scan[j]
                add_index += j
            elif self.diff[j] > 0 and obj_detected == 1:
                obj_detected = 0
                if indices > 0:
                    avg_angle = scale((add_index * (m.pi/180)) / indices)
                    avg_range = (add_ranges / indices) + 0.15377
                    
                    # Only add valid landmarks
                    if avg_range > 0.1 and avg_range < 10.0:  # reasonable range limits
                        self.lidar_angle.append(avg_angle)
                        self.lidar_range.append(avg_range)
                
                indices = 0
                add_index = 0
                add_ranges = 0.0
            j += 1

        # Create measurement matrix
        if self.lidar_range:
            self.zm = np.vstack([self.lidar_range, self.lidar_angle])
        else:
            self.zm = np.array([]).reshape(2, 0)  # Proper empty shape
        
        if len(self.lidar_range) > 0:
            self.get_logger().info(f"Detected {len(self.lidar_range)} landmarks with ranges: {[f'{r:.2f}' for r in self.lidar_range[:5]]}")  # Show first 5

    def calculate_predicted_measurements(self):
        """Calculate predicted range and bearing for existing landmarks."""
        predicted_range = []
        predicted_bearing = []
        
        # Start from index 3 (after robot pose), step by 2 for each landmark
        for i in range(3, self.state_matrix.shape[0], 2):
            # Extract landmark coordinates
            landmark_x = self.state_matrix[i, 0]
            landmark_y = self.state_matrix[i + 1, 0]
            
            # Extract robot pose
            robot_x = self.state_matrix[0, 0]
            robot_y = self.state_matrix[1, 0]
            robot_theta = self.state_matrix[2, 0]
            
            # Calculate relative position
            dx = landmark_x - robot_x
            dy = landmark_y - robot_y
            
            # Calculate range and bearing
            range_pred = np.sqrt(dx**2 + dy**2)
            bearing_pred = scale(np.arctan2(dy, dx) - robot_theta)
            
            predicted_range.append(range_pred)
            predicted_bearing.append(bearing_pred)
        
        if predicted_range:
            self.predicted_measurements = np.vstack([predicted_range, predicted_bearing])
        else:
            self.predicted_measurements = np.array([]).reshape(2, 0)
            
    def match_state(self):
        """Match detected landmarks with existing landmarks in state and identify new ones."""
        # Initialize empty lists
        self.matched_landmarks = []
        self.new_landmarks = []
        
        # Check if we have any observations
        if self.zm.size == 0 or len(self.lidar_range) == 0:
            return
        
        # Check if we have existing landmarks in state
        if self.state_matrix is None or self.state_matrix.shape[0] <= 3:
            # No existing landmarks, all observations are new
            self.new_landmarks = list(range(len(self.lidar_range)))
            self.add_new_landmarks()
            return
        
        # Calculate predicted measurements for existing landmarks
        self.calculate_predicted_measurements()
        
        # Check if predicted measurements are empty
        if self.predicted_measurements.size == 0:
            self.new_landmarks = list(range(len(self.lidar_range)))
            self.add_new_landmarks()
            return
        
        matched_observations = set()
        matched_landmarks = set()  # Track which landmarks are already matched
        
        # Association thresholds
        range_threshold = 0.5  # meters
        bearing_threshold = 0.3  # radians (~17 degrees)
        
        # Create list of all possible matches with their distances
        possible_matches = []
        
        for i in range(self.zm.shape[1]):  # For each observation
            for j in range(self.predicted_measurements.shape[1]):  # For each existing landmark
                # Calculate distance metrics
                range_diff = abs(self.zm[0, i] - self.predicted_measurements[0, j])
                bearing_diff = abs(scale(self.zm[1, i] - self.predicted_measurements[1, j]))
                
                # Check if within thresholds
                if range_diff < range_threshold and bearing_diff < bearing_threshold:
                    # Combined distance metric
                    distance = np.sqrt(range_diff**2 + bearing_diff**2)
                    possible_matches.append((distance, i, j))
        
        # Sort matches by distance (best matches first)
        possible_matches.sort(key=lambda x: x[0])
        
        # Greedily assign matches, ensuring one-to-one correspondence
        for distance, obs_idx, landmark_idx in possible_matches:
            # Check if both observation and landmark are still available
            if obs_idx not in matched_observations and landmark_idx not in matched_landmarks:
                self.matched_landmarks.append((obs_idx, landmark_idx))
                matched_observations.add(obs_idx)
                matched_landmarks.add(landmark_idx)
        
        # Identify new landmarks (unmatched observations)
        self.new_landmarks = [i for i in range(self.zm.shape[1]) if i not in matched_observations]
        
        # Add new landmarks to state
        if self.new_landmarks:
            self.add_new_landmarks()
        
        # Debug information
        if len(self.matched_landmarks) > 0 or len(self.new_landmarks) > 0:
            self.get_logger().info(f"Matched {len(self.matched_landmarks)} landmarks, "
                                f"Added {len(self.new_landmarks)} new landmarks, "
                                f"Total landmarks in state: {(self.n - 3) // 2}")

    def add_new_landmarks(self):
        """Add new landmarks to the state vector and covariance matrix."""
        num_new_landmarks = len(self.new_landmarks)
        if num_new_landmarks == 0:
            return
        
        # Current state size
        current_size = self.state_matrix.shape[0]
        
        # Each landmark adds 2 states (x, y coordinates)
        new_size = current_size + 2 * num_new_landmarks
        
        # Extend state matrix
        new_state = np.zeros((new_size, 1))
        new_state[:current_size, 0] = self.state_matrix[:, 0]
        
        # Add new landmark positions
        robot_x = self.state_matrix[0, 0]
        robot_y = self.state_matrix[1, 0]
        robot_theta = self.state_matrix[2, 0]
        
        self.get_logger().info(f"Robot pose: ({robot_x:.2f}, {robot_y:.2f}, {robot_theta:.2f})")
        
        for idx, obs_idx in enumerate(self.new_landmarks):
            landmark_idx = current_size + 2 * idx
            
            # Convert polar observation to global Cartesian coordinates
            range_val = self.zm[0, obs_idx]
            bearing = self.zm[1, obs_idx]
            
            # Global landmark position
            landmark_x = robot_x + range_val * np.cos(robot_theta + bearing)
            landmark_y = robot_y + range_val * np.sin(robot_theta + bearing)
            
            new_state[landmark_idx, 0] = landmark_x
            new_state[landmark_idx + 1, 0] = landmark_y
            
            self.get_logger().info(f"Adding landmark {idx}: range={range_val:.2f}, bearing={bearing:.2f} -> ({landmark_x:.2f}, {landmark_y:.2f})")
        
        self.state_matrix = new_state
        
        # Extend covariance matrix
        new_P = np.zeros((new_size, new_size))
        new_P[:current_size, :current_size] = self.P
        
        # Initialize covariance for new landmarks
        # High uncertainty for new landmarks
        landmark_uncertainty = 1.0  # meters^2
        for idx in range(num_new_landmarks):
            landmark_idx = current_size + 2 * idx
            new_P[landmark_idx, landmark_idx] = landmark_uncertainty
            new_P[landmark_idx + 1, landmark_idx + 1] = landmark_uncertainty
        
        self.P = new_P
        self.n = new_size  # Update state size
        
        self.get_logger().info(f"Added {num_new_landmarks} new landmarks. State size now: {new_size}")
        self.get_logger().info(f"Current state: {self.state_matrix.flatten()[:min(10, len(self.state_matrix.flatten()))]}")  # Show first 10 elements

    def correction_step(self):
        """Perform EKF correction step for matched landmarks."""
        if not hasattr(self, 'matched_landmarks') or not self.matched_landmarks:
            return
        
        # Measurement noise covariance
        R_sensor = np.diag([0.1**2, 0.05**2])  # range and bearing noise
        
        for obs_idx, landmark_idx in self.matched_landmarks:
            # Observed measurement
            z = np.array([[self.zm[0, obs_idx]], [self.zm[1, obs_idx]]])
            
            # Predicted measurement
            h = np.array([[self.predicted_measurements[0, landmark_idx]], 
                         [self.predicted_measurements[1, landmark_idx]]])
            
            # Innovation
            y = z - h
            y[1, 0] = scale(y[1, 0])  # Normalize bearing innovation
            
            # Compute Jacobian H (measurement model)
            robot_x = self.state_matrix[0, 0]
            robot_y = self.state_matrix[1, 0]
            robot_theta = self.state_matrix[2, 0]
            
            landmark_state_idx = 3 + 2 * landmark_idx
            landmark_x = self.state_matrix[landmark_state_idx, 0]
            landmark_y = self.state_matrix[landmark_state_idx + 1, 0]
            
            dx = landmark_x - robot_x
            dy = landmark_y - robot_y
            q = dx**2 + dy**2
            sqrt_q = np.sqrt(q)
            
            # Jacobian matrix
            H = np.zeros((2, self.n))
            H[0, 0] = -dx / sqrt_q  # ∂r/∂x_robot
            H[0, 1] = -dy / sqrt_q  # ∂r/∂y_robot
            H[0, landmark_state_idx] = dx / sqrt_q      # ∂r/∂x_landmark
            H[0, landmark_state_idx + 1] = dy / sqrt_q  # ∂r/∂y_landmark
            
            H[1, 0] = dy / q        # ∂θ/∂x_robot
            H[1, 1] = -dx / q       # ∂θ/∂y_robot
            H[1, 2] = -1            # ∂θ/∂θ_robot
            H[1, landmark_state_idx] = -dy / q     # ∂θ/∂x_landmark
            H[1, landmark_state_idx + 1] = dx / q  # ∂θ/∂y_landmark
            
            # Innovation covariance
            S = H @ self.P @ H.T + R_sensor
            
            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # State update
            self.state_matrix += K @ y
            
            # Covariance update
            I = np.eye(self.n)
            self.P = (I - K @ H) @ self.P

    def timer_cb(self):
        """Main EKF-SLAM loop."""
        try:
            self.update_state()
            self.state_transition_matrix()
            self.control_covariance_matrix()
            self.covariance_matrix()
            self.detect_landmarks()
            
            # Data association and state augmentation
            self.match_state()
            
            # Correction step for matched landmarks
            self.correction_step()
            
            # Optional: Publish filtered pose
            if self.state_matrix is not None:
                pose_msg = PoseWithCovarianceStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.pose.pose.position.x = self.state_matrix[0, 0]
                pose_msg.pose.pose.position.y = self.state_matrix[1, 0]
                orientation = quaternion_from_euler(0, 0, self.state_matrix[2, 0])
                pose_msg.pose.pose.orientation.x = orientation[0]
                pose_msg.pose.pose.orientation.y = orientation[1]
                pose_msg.pose.pose.orientation.z = orientation[2]
                pose_msg.pose.pose.orientation.w = orientation[3]
                self.pub.publish(pose_msg)
                state_msg = Float32MultiArray()
                state_msg.data = self.state_matrix.flatten().tolist()
                self.state_pub.publish(state_msg)
                self.get_logger().info(f"Published state with {self.state_matrix.shape[0]} elements, "
                                    f"{(self.state_matrix.shape[0]-3)//2} landmarks")
        
        except Exception as e:
            self.get_logger().error(f"Error in timer_cb: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKF_SLAM()
    try:
        rclpy.spin(ekf_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        ekf_node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        ekf_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()