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
    def odom_callback(self, msg):
        self.m_x = msg.pose.pose.position.x
        self.m_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.m_0 = scale(yaw)
        self.Dt = m.sqrt((self.m_x - self.m_x_prev) ** 2 + (self.m_y - self.m_y_prev) ** 2)