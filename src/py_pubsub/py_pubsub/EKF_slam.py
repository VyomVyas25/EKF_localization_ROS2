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
    """Normalize angle to [-pi, pi]."""
    return (angle + m.pi) % (2 * m.pi) - m.pi


class EKF_SLAM(Node):
    def __init__(self):
        super().__init__('ekf_slam')

        # Subscriptions & publisher
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/filtered_data', 10)
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
        self.scan = []
        self.diff = []

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
        if self.prev_state is None:
            self.prev_state = np.array([[self.m_x_prev], [self.m_prev_y], [self.m_prev_0]])
        else:
            self.prev_state = np.array([[self.m_x], [self.m_y], [self.m_0]])

        self.state_matrix = self.prev_state + np.array([[self.Dt], [self.Dr1], [self.Dr2]])

        rows = self.state_matrix.shape[0]
        extra_cols = max(0, rows - 3)  # columns for landmarks
        self.state_matrix = np.hstack((self.state_matrix, np.zeros((rows, extra_cols))))

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
        self.V = np.zeros((self.n, 3))
        self.V[0, 0] = -self.Dt * m.sin(self.m_prev_0 + self.Dr1)
        self.V[0, 1] = self.Dt * m.cos(self.m_prev_0 + self.Dr1)
        self.V[1, 0] = self.Dt * m.cos(self.m_prev_0 + self.Dr1)
        self.V[1, 1] = self.Dt * m.sin(self.m_prev_0 + self.Dr1)
        self.V[2, 0] = 1
        self.V[2, 2] = 1

        self.Q = np.diag([
            self.sigma_x ** 2,
            self.sigma_y ** 2,
            self.sigma_theta ** 2
        ])

        self.R = self.V @ self.Q @ self.V.T

    def covariance_matrix(self):
        self.P = self.F @ self.P @ self.F.T + self.R

    def scan_callback(self, msg: LaserScan):
        self.scan = []
        self.diff = []

        for i, r in enumerate(msg.ranges):
            if m.isinf(r) or m.isnan(r):
                self.scan.append(0.0)
            else:
                self.scan.append(r)

            if i > 0 and self.scan[i] > 0.2 and self.scan[i - 1] > 0.2:
                diff_val = self.scan[i] - self.scan[i - 1]
            else:
                diff_val = 0.0
            self.diff.append(diff_val if abs(diff_val) > 0.15 else 0.0)

    def detect_landmarks(self):
        self.lidar_angle = []
        self.lidar_range = []
        obj_detected = False
        indices = 0
        add_index = 0
        add_ranges = 0.0

        for j, d in enumerate(self.diff):
            if d < 0.0 and not obj_detected:
                obj_detected = True
                indices = 1
                add_index = j
                add_ranges = self.scan[j]
            elif d == 0.0 and obj_detected:
                indices += 1
                add_index += j
                add_ranges += self.scan[j]
            elif d > 0.0 and obj_detected:
                avg_angle = scale((add_index / indices) * (m.pi / 180.0))
                self.lidar_angle.append(avg_angle)
                self.lidar_range.append((add_ranges / indices) + 0.1537)
                obj_detected = False
                indices = 0
                add_index = 0
                add_ranges = 0.0

        if self.lidar_angle:
            self.zm = np.vstack((self.lidar_angle, self.lidar_range)).T
        else:
            self.zm = np.empty((0, 2))

    def timer_cb(self):
        self.get_logger().info('Timer callback triggered')
        try:
            self.update_state()
            self.state_transition_matrix()
            self.control_covariance_matrix()
            self.covariance_matrix()

            if self.scan:
                self.detect_landmarks()
                self.get_logger().info(f'Detected landmarks: {len(self.lidar_angle)}')

            # Publish filtered pose
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.pose.position.x = self.m_x
            pose_msg.pose.pose.position.y = self.m_y
            orientation = quaternion_from_euler(0, 0, self.m_0)
            pose_msg.pose.pose.orientation.x = orientation[0]
            pose_msg.pose.pose.orientation.y = orientation[1]
            pose_msg.pose.pose.orientation.z = orientation[2]
            pose_msg.pose.pose.orientation.w = orientation[3]
            self.pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f"Unexpected error in timer_cb: {e}")


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