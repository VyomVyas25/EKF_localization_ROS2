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
        self.m_x_prev = 0.0
        self.m_prev_y = 0.0
        self.m_prev_0 = 0.0

        self.prev_state = None
        self.state_matrix = None
        self.n = 3

        self.sigma_x = 0.1  # Standard deviation for x
        self.sigma_y = 0.1  # Standard deviation for y
        self.sigma_theta = 0.1  # Standard deviation for theta
       
        
    def odom_callback(self, msg:Odometry):
        self.m_x = msg.pose.pose.position.x
        self.m_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.m_0 = scale(yaw)
        self.Dt = m.sqrt((self.m_x - self.m_x_prev) ** 2 + (self.m_y - self.m_y_prev) ** 2)
        self.Dr1 = scale(m.atan2(self.m_y - self.m_prev_y, self.m_x - self.m_prev_x) - self.m_prev_0)
        self.Dr2 = scale(self.m_0 - self.m_prev_0 - self.Dr1)


    def update_state(self):
        if self.prev_state is None:
            self.prev_state = np.array([[self.m_x_prev], [self.m_prev_y], [self.m_prev_0]])
        else:
            self.prev_state = np.array([[self.m_x], [self.m_y], [self.m_0]])
        self.state_matrix = self.prev_state + np.array([[self.Dt], [self.Dr1], [self.Dr2]])

        final_state = np.block([
            [self.state_matrix],
            np.zeros((len(self.state_matrix), len(self.state_matrix) - 3))
        ])

        self.state_matrix = final_state


    def state_transition_matrix(self):
        # Update the state transition matrix Gt based on the current state 
        self.Gt = np.array([[1, 0, -self.Dt * m.sin(self.m_prev_0 + self.Dr1)],
                            [0, 1, self.Dt * m.cos(self.m_prev_0 + self.Dr1)],
                            [0, 0, 1]])
        
        # If state_matrix is not None, adjust the size of Gt accordingly
        if self.state_matrix is not None:
            self.n = len(self.state_matrix)  
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
        self.V = np.zeros((self.n,3))
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

    def covariamce_matrix(self):
        self.P = self.F @ self.P @ self.F.T + self.R
    

    def scan_callback(self, msg:LaserScan):
        # Process the laser scan data
        # This is a placeholder for actual SLAM logic
        self.get_logger().info('Processing LaserScan data')
        
        self.scan = [0]*(len(msg.ranges))
        for i in range(len(msg.ranges)):
            if m.isinf(msg.ranges[i]) or m.isnan(msg.ranges[i]):
                self.scan[i] = 0.0
            else:
                self.scan[i] = msg.ranges[i]
            
        
    def detect_landmarks(self):

        j = -1
        while j < len(self.scan - 1):
            if self.scan[j] > 0.0:
                # Process landmark detection logic here
                pass
            j += 1