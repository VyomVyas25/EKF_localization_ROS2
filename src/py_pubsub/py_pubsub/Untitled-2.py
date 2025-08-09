#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math as m
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, PoseStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from rclpy.clock import Clock

def scale(angle):
    return (angle + m.pi) % (2 * m.pi) - m.pi

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_localization_node')
        # Subscriptions
        self.create_subscription(Float32MultiArray, '/cylinder_coordinates', self.cylinder_coords_callback, 10)
        self.create_subscription(Odometry, '/odom', self.pos_measurement_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.pos_lidar_callback, 10)
        # Publisher for filtered pose
        self.pub_filtered = self.create_publisher(PoseWithCovarianceStamped, '/filtered_data', 10)
        # Publishers for visualization markers:
        # Cylinder markers are now published as a MarkerArray on the topic '/visualization_marker_array'
        self.marker_array_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        # Error ellipse publisher
        self.ellipse_marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        # Publisher for trajectory markers (for path visualization)
        self.marker_pub = self.create_publisher(Marker, '/trajectory_marker', 10)
        # Publishers for trajectories as Path messages
        self.odom_path_pub = self.create_publisher(Path, '/orig_path', 10)
        self.filtered_path_pub = self.create_publisher(Path, '/filtered_path', 10)
        
        # Initialize state variables:
        self.m_prev_x = 0.0
        self.m_prev_y = 0.0
        self.m_prev_0 = 0.0
        self.Dt = 0.0
        self.Dr1 = 0.0
        self.Dr2 = 0.0
        self.m_x = 0.0
        self.m_y = 0.0
        self.m_0 = 0.0
        self.covariance_prev = np.eye(3)
        # Motion model noise parameters:
        self.alpha1 = 0.03  
        self.alpha2 = 0.01  
        self.alpha3 = 0.03
        self.alpha4 = 0.01  
        # Lidar processing:
        self.lidar_distance = []
        self.lidar_angle = []
        self.diff = []

        self.map_cylinders = []

        self.sigma_r = 0.1
        self.sigma_phi = 0.01
        self.zm = []
        self.meas = np.array([])  
        self.o = np.array([])     
        self.difff = np.array([])  

        self.odom_path = []      # raw odometry (real robot trajectory)
        self.filtered_path = []  # filtered EKF trajectory

        self.timer = self.create_timer(0.5, self.timer_callback)

    def cylinder_coords_callback(self, msg: Float32MultiArray):
        """
        Split the flat array [x1, y1, x2, y2, ...] into coordinate pairs.
        Log the received coordinates and publish cylinder markers.
        """
        data = msg.data
        if len(data) % 2 != 0:
            self.get_logger().error("Received cylinder coordinates with uneven length!")
            return
        coords = []
        for i in range(0, len(data), 2):
            coords.append([data[i], data[i+1]])
        self.map_cylinders = coords
        self.get_logger().info(f"Received cylinder coordinates: {coords}")
        self.publish_cylinder_marker_array()

    def pos_lidar_callback(self, scan: LaserScan):
        n = len(scan.ranges)
        self.x = [0.0] * n
        self.lidar_distance = []
        self.lidar_angle = []
        self.diff = [0.0] * n
        for i in range(n):
            if m.isnan(scan.ranges[i]) or m.isinf(scan.ranges[i]):
                self.x[i] = 3.0
            else:
                self.x[i] = scan.ranges[i]
            if i > 0 and self.x[i] > 0.2 and self.x[i-1] > 0.2:
                diff_val = self.x[i] - self.x[i-1]
            else:
                diff_val = 0.0
            self.diff[i] = diff_val if abs(diff_val) > 0.15 else 0.0

    def pos_measurement_callback(self, msg: Odometry):
        self.m_x = msg.pose.pose.position.x
        self.m_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.m_0 = scale(yaw)
        self.Dt = m.sqrt((self.m_x - self.m_prev_x)**2 + (self.m_y - self.m_prev_y)**2)
        self.Dr1 = scale(m.atan2(self.m_y - self.m_prev_y, self.m_x - self.m_prev_x) - self.m_prev_0)
        self.Dr2 = scale(self.m_0 - self.m_prev_0 - self.Dr1)
        # Append current odometry point
        self.odom_path.append(Point(x=self.m_x, y=self.m_y, z=0.0))

    # --- Motion Prediction & Covariance ---
    def state_covariance(self):
        self.Gt = np.array([
            [1, 0, -self.Dt * m.sin(self.m_prev_0 + self.Dr1)],
            [0, 1,  self.Dt * m.cos(self.m_prev_0 + self.Dr1)],
            [0, 0, 1]
        ])

    def control_covariance(self):
        self.Vt = np.array([
            [-self.Dt * m.sin(self.m_prev_0 + self.Dr1), m.cos(self.m_prev_0 + self.Dr1), 0],
            [ self.Dt * m.cos(self.m_prev_0 + self.Dr1), m.sin(self.m_prev_0 + self.Dr1), 0],
            [1, 0, 1]
        ])

    def process_noise(self):
        self.Mt = np.array([
            [self.alpha1*(self.Dr1**2) + self.alpha2*(self.Dt**2), 0, 0],
            [0, self.alpha3*(self.Dt**2) + self.alpha4*((self.Dr2**2)+(self.Dr1**2)), 0],
            [0, 0, self.alpha1*(self.Dr1**2) + self.alpha2*(self.Dt**2)]
        ])

    def mu_mean(self):
        self.x_bel = self.m_prev_x + self.Dt * m.cos(self.m_prev_0 + self.Dr1)
        self.y_bel = self.m_prev_y + self.Dt * m.sin(self.m_prev_0 + self.Dr1)
        self.theta_bel = scale(self.m_prev_0 + self.Dr1 + self.Dr2)
        self.muBar = np.array([[self.x_bel],
                               [self.y_bel],
                               [self.theta_bel]])
        # Append filtered pose to EKF trajectory:
        self.filtered_path.append(Point(x=self.muBar[0, 0], y=self.muBar[1, 0], z=0.0))

    def sigma_covariance(self):
        self.state_covariance()
        self.control_covariance()
        self.sigmaBar = self.Gt @ self.covariance_prev @ self.Gt.T + self.Vt @ self.Mt @ self.Vt.T

    # --- LIDAR Observations & Matching ---
    def observed_landmarks(self):
        # A simple grouping based on jumps in LIDAR data
        obj_detected = 0
        add_ranges = 0.0
        indices = 0
        add_index = 0
        j = -1
        while j < (len(self.diff) - 1):
            if self.diff[j] < 0 and obj_detected == 0:
                obj_detected = 1
                indices += 1
                add_index += j
                add_ranges += self.x[j]
            elif self.diff[j] < 0 and obj_detected == 1:
                obj_detected = 0
                indices = 0
                add_index = 0
                add_ranges = 0.0
                j -= 1
            elif self.diff[j] == 0 and obj_detected == 1:
                indices += 1
                add_ranges += self.x[j]
                add_index += j
            elif self.diff[j] > 0 and obj_detected == 1:
                obj_detected = 0
                avg_angle = scale((add_index * (m.pi/180)) / indices)
                self.lidar_angle.append(avg_angle)
                self.lidar_distance.append((add_ranges) / indices + 0.25)
                indices = 0
                add_index = 0
                add_ranges = 0.0
            j += 1
        self.zm = np.vstack([self.lidar_distance, self.lidar_angle])
    
    def obs_meas_difference(self):
        # Compare each map cylinder with the robot’s predicted pose (muBar)
        self.orig_dist = []
        self.orig_angle = []
        self.diff_x = []
        self.diff_y = []
        self.meas_dist = []
        self.meas_angle = []
        self.o_dist = []
        self.o_angle = []
        self.difff_x = []
        self.difff_y = []
        for i in range(len(self.map_cylinders)):
            diff_x = self.map_cylinders[i][0] - self.x_bel
            diff_y = self.map_cylinders[i][1] - self.y_bel
            dist = m.sqrt(diff_x**2 + diff_y**2)
            angle = scale(m.atan2(diff_y, diff_x) - self.theta_bel)
            self.orig_dist.append(dist)
            self.orig_angle.append(angle)
            self.diff_x.append(diff_x)
            self.diff_y.append(diff_y)
        self.orig = np.vstack((self.orig_dist, self.orig_angle))
        self.diff_1 = np.vstack((self.diff_x, self.diff_y))
        if len(self.map_cylinders) == 0 or len(self.zm[0]) == 0:
            self.get_logger().warn("No landmarks detected from lidar!")
            return
        for i in range(np.shape(self.orig)[1]):
            for j in range(np.shape(self.zm)[1]):
                if abs(self.zm[0, j] - self.orig[0, i]) < 0.2 and abs(self.zm[1, j] - self.orig[1, i]) < 0.2:
                    self.meas_dist.append(self.zm[0][j])
                    self.meas_angle.append(self.zm[1][j])
                    self.o_dist.append(self.orig[0][i])
                    self.o_angle.append(self.orig[1][i])
                    self.difff_x.append(self.diff_1[0][i])
                    self.difff_y.append(self.diff_1[1][i])
                    break
        if self.meas_angle and self.meas_dist:
            self.meas = np.vstack((self.meas_dist, self.meas_angle))
        else:
            self.meas = np.array([])
        if self.o_dist and self.o_angle:
            self.o = np.vstack((self.o_dist, self.o_angle))
        else:
            self.o = np.array([])
        if self.difff_x and self.difff_y:
            self.difff = np.vstack((self.difff_x, self.difff_y))
        else:
            self.difff = np.array([])

    def correction_matrices(self):
        if self.meas.size == 0:
            self.Mu = self.muBar
            self.covariance_prev = self.sigmaBar
        else:
            for i in range(self.meas.shape[1]):
                measDist = self.meas[0][i]
                measAngle = scale(self.meas[1][i])
                Zt = np.array([[measDist],
                               [measAngle]])
                hx = np.array([[self.o[0][i]],
                               [scale(self.o[1][i])]])
                dx = self.difff[0][i]
                dy = self.difff[1][i]
                r = self.o[0][i]
                Ht = np.array([
                    [-dx / r, -dy / r, 0],
                    [dy / (r**2), -dx / (r**2), -1]
                ])
                Qt = np.array([[self.sigma_r**2, 0],
                               [0, self.sigma_phi**2]])
                S_t = Ht @ self.sigmaBar @ Ht.T + Qt
                innovation = np.array([[Zt[0][0] - hx[0][0]],
                                       [scale(Zt[1][0] - hx[1][0])]])
                K_t = self.sigmaBar @ Ht.T @ np.linalg.inv(S_t)
                correction = K_t @ innovation
                I_KH = np.eye(3) - (K_t @ Ht)
                self.Mu = self.muBar + correction
                self.covariance_prev = I_KH @ self.sigmaBar
        self.m_prev_x = self.m_x
        self.m_prev_y = self.m_y
        self.m_prev_0 = self.m_0

    # --- Publishing Functions ---
    def publish_mu_sigma(self):
        if self.pub_filtered.get_subscription_count() == 0:
            return
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose.pose.position.x = self.Mu[0, 0]
        pose.pose.pose.position.y = self.Mu[1, 0]
        quat = quaternion_from_euler(0, 0, self.Mu[2, 0])
        pose.pose.pose.orientation.z = quat[2]
        pose.pose.pose.orientation.w = quat[3]
        full_cov = np.zeros((6, 6))
        full_cov[:3, :3] = self.covariance_prev
        pose.pose.covariance = full_cov.ravel().tolist()
        self.pub_filtered.publish(pose)
        # Publish error ellipse, path markers, and trajectories
        self.publish_error_ellipse(pose.pose.pose.position.x, pose.pose.pose.position.y, self.covariance_prev)
        self.publish_path_markers()
        self.publish_trajectory()

    def publish_cylinder_marker_array(self):
        """
        Publish a MarkerArray with each marker representing one cylinder.
        The markers are now set to use frame "odom", namespace "estimated_cylinders",
        and are positioned slightly above ground (z = 0.1) for better visualization.
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        for idx, coord in enumerate(self.map_cylinders):
            marker = Marker()
            marker.header.frame_id = "odom"  # Changed from "map" to "odom"
            marker.header.stamp = now
            marker.ns = "estimated_cylinders"  # Updated namespace
            marker.id = idx  # Unique marker ID per cylinder.
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = coord[0]
            marker.pose.position.y = coord[1]
            marker.pose.position.z = 0.1  # Slightly above ground
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
            self.get_logger().info(f"Publishing cylinder marker {idx} at: {coord}")
        self.marker_array_pub.publish(marker_array)
    
    def publish_path_markers(self):
        """
        Publish two path markers (raw odometry in blue, filtered EKF in green)
        as Marker LINE_STRIPs in the "odom" frame.
        """
        now = self.get_clock().now().to_msg()
        # Odometry path marker (blue)
        marker_odom = Marker()
        marker_odom.header.frame_id = "odom"
        marker_odom.header.stamp = now
        marker_odom.ns = "odom_path"
        marker_odom.id = 1
        marker_odom.type = Marker.LINE_STRIP
        marker_odom.action = Marker.ADD
        marker_odom.scale.x = 0.05
        marker_odom.color.a = 1.0
        marker_odom.color.b = 1.0  # Blue
        for pt in self.odom_path:
            marker_odom.points.append(pt)
        self.marker_pub.publish(marker_odom)
        # Filtered EKF path marker (green)
        marker_filtered = Marker()
        marker_filtered.header.frame_id = "odom"
        marker_filtered.header.stamp = now
        marker_filtered.ns = "filtered_path"
        marker_filtered.id = 2
        marker_filtered.type = Marker.LINE_STRIP
        marker_filtered.action = Marker.ADD
        marker_filtered.scale.x = 0.05
        marker_filtered.color.a = 1.0
        marker_filtered.color.g = 1.0  # Green
        for pt in self.filtered_path:
            marker_filtered.points.append(pt)
        self.marker_pub.publish(marker_filtered)
    
    def publish_trajectory(self):
        """
        Publish the accumulated trajectories as Path messages for both
        raw odometry and filtered EKF.
        """
        now = Clock().now().to_msg()
        # Filtered EKF trajectory Path
        filtered_path_msg = Path()
        filtered_path_msg.header.frame_id = "odom"
        filtered_path_msg.header.stamp = now
        for pt in self.filtered_path:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "odom"
            ps.pose.position = pt
            quat = quaternion_from_euler(0, 0, self.Mu[2, 0])
            ps.pose.orientation.x = quat[0]
            ps.pose.orientation.y = quat[1]
            ps.pose.orientation.z = quat[2]
            ps.pose.orientation.w = quat[3]
            filtered_path_msg.poses.append(ps)
        self.filtered_path_pub.publish(filtered_path_msg)
        # Raw odometry trajectory Path
        odom_path_msg = Path()
        odom_path_msg.header.frame_id = "odom"
        odom_path_msg.header.stamp = now
        for pt in self.odom_path:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "odom"
            ps.pose.position = pt
            odom_path_msg.poses.append(ps)
        self.odom_path_pub.publish(odom_path_msg)
    
    def publish_error_ellipse(self, x, y, cov):
        """
        Compute an error ellipse from the 2x2 covariance submatrix and
        publish it as a Marker (LINE_STRIP) in the "odom" frame.
        """
        cov2d = cov[0:2, 0:2]
        eigvals, eigvecs = np.linalg.eigh(cov2d)
        chisq = 5.991  # 95% confidence for 2D
        axis1 = m.sqrt(eigvals[0] * chisq)
        axis2 = m.sqrt(eigvals[1] * chisq)
        max_idx = np.argmax(eigvals)
        angle = m.atan2(eigvecs[1, max_idx], eigvecs[0, max_idx])
        ellipse_points = []
        num_pts = 50
        for i in range(num_pts+1):
            theta = 2 * m.pi * i / num_pts
            ex = axis1 * m.cos(theta)
            ey = axis2 * m.sin(theta)
            rx = x + ex * m.cos(angle) - ey * m.sin(angle)
            ry = y + ex * m.sin(angle) + ey * m.cos(angle)
            ellipse_points.append(Point(x=rx, y=ry, z=0.0))
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = Clock().now().to_msg()
        marker.ns = "error_ellipse"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.points = ellipse_points
        self.ellipse_marker_pub.publish(marker)

    def run(self):
        # Prediction step:
        self.control_covariance()
        self.state_covariance()
        self.process_noise()    
        self.mu_mean()
        self.sigma_covariance()
        #correction step:
        self.observed_landmarks()
        self.obs_meas_difference()
        self.correction_matrices()
        # Publish filtered pose:
        self.publish_mu_sigma()
        # Publish visualization markers:
        self.publish_cylinder_marker_array()
        self.publish_path_markers()
        self.publish_trajectory()
    
    def timer_callback(self):
        self.run()

def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKFNode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()