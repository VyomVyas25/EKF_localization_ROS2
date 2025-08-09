#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math as m
import math
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
        # Publishers for visualization markers
        self.marker_array_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.marker_pub = self.create_publisher(Marker, '/trajectory_marker', 10)
        # Error ellipse publisher (a single Marker, as in arr_det.py)
        self.error_ellipse_pub = self.create_publisher(Marker, '/error_ellipse', 10)
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
        # Noise parameters
        self.alpha1 = 0.03  
        self.alpha2 = 0.01  
        self.alpha3 = 0.03
        self.alpha4 = 0.01  
        # Lidar processing:
        self.lidar_distance = []
        self.lidar_angle = []
        self.diff = []
        # Map-based cylinder coordinates:
        self.map_cylinders = []
        # Measurement noise parameters:
        self.sigma_r = 0.1
        self.sigma_phi = 0.01
        self.zm = []
        self.meas = np.array([])  
        self.o = np.array([])     
        self.difff = np.array([])  
        # For trajectories:
        self.odom_path = []      # raw odometry trajectory
        self.filtered_path = []  # filtered EKF trajectory
        # Timer (update at 2 Hz)
        self.timer = self.create_timer(0.5, self.timer_callback)
        # Initialize Mu to prevent errors on first publish
        self.Mu = np.array([[0.0], [0.0], [0.0]])
        # Flag to track if we've received data yet
        self.received_data = False

    # --- Subscription Callbacks ---
    def cylinder_coords_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) % 2 != 0:
            self.get_logger().error("Received cylinder coordinates with uneven length!")
            return
        coords = []
        for i in range(0, len(data), 2):
            coords.append([data[i], data[i+1]])
        self.map_cylinders = coords
        #self.get_logger().info(f"Received cylinder coordinates: {coords}")
        self.publish_cylinder_marker_array()
        self.received_data = True

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
        self.received_data = True

    def pos_measurement_callback(self, msg: Odometry):
        self.m_x = msg.pose.pose.position.x
        self.m_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x,
                                           orientation.y,
                                           orientation.z,
                                           orientation.w])
        self.m_0 = scale(yaw)
        self.Dt = m.sqrt((self.m_x - self.m_prev_x)**2 + (self.m_y - self.m_prev_y)**2)
        self.Dr1 = scale(m.atan2(self.m_y - self.m_prev_y, self.m_x - self.m_prev_x) - self.m_prev_0)
        self.Dr2 = scale(self.m_0 - self.m_prev_0 - self.Dr1)
        self.odom_path.append(Point(x=self.m_x, y=self.m_y, z=0.0))
        self.received_data = True

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
        self.filtered_path.append(Point(x=self.muBar[0, 0], y=self.muBar[1, 0], z=0.0))

    def sigma_covariance(self):
        self.state_covariance()
        self.control_covariance()
        self.sigmaBar = self.Gt @ self.covariance_prev @ self.Gt.T + self.Vt @ self.Mt @ self.Vt.T

    # --- LIDAR Observations & Matching ---
    def observed_landmarks(self):
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
                self.lidar_distance.append((add_ranges) / indices + 0.15377 )
                indices = 0
                add_index = 0
                add_ranges = 0.0
            j += 1
        self.zm = np.vstack([self.lidar_distance, self.lidar_angle]) if self.lidar_distance else np.array([])
    
    def obs_meas_difference(self):
        # If no LIDAR measurements have been collected, skip pairing.
        if not isinstance(self.zm, np.ndarray) or self.zm.size == 0:
            self.meas = np.array([])
            self.o = np.array([])
            self.difff = np.array([])
            return

        orig_dist = []
        orig_angle = []
        diff_x = []
        diff_y = []
        for coord in self.map_cylinders:
            dx = coord[0] - self.x_bel
            dy = coord[1] - self.y_bel
            orig_dist.append(m.sqrt(dx**2 + dy**2))
            orig_angle.append(scale(m.atan2(dy, dx) - self.theta_bel))
            diff_x.append(dx)
            diff_y.append(dy)
        if len(orig_dist) == 0:
            self.meas = np.array([])
            self.o = np.array([])
            self.difff = np.array([])
            return

        orig = np.vstack((orig_dist, orig_angle))
        diff_1 = np.vstack((diff_x, diff_y))
        
        paired_indices = []
        # For each predicted measurement, find a valid LIDAR measurement.
        for i in range(orig.shape[1]):
            diff = np.abs(self.zm - orig[:, i].reshape(2, 1))
            valid_idx = np.where((diff[0, :] < 0.2) & (diff[1, :] < 0.2))[0]
            if valid_idx.size > 0:
                j = valid_idx[np.argmin(diff[0, valid_idx] + diff[1, valid_idx])]
                paired_indices.append((i, j))
        
        if paired_indices:
            paired_meas = np.array([[orig[0, i] for (i, j) in paired_indices],
                                      [orig[1, i] for (i, j) in paired_indices]])
            paired_lidar = np.array([[self.zm[0, j] for (i, j) in paired_indices],
                                     [self.zm[1, j] for (i, j) in paired_indices]])
            paired_diff = np.array([[diff_1[0, i] for (i, j) in paired_indices],
                                    [diff_1[1, i] for (i, j) in paired_indices]])
        else:
            paired_meas = np.array([])
            paired_lidar = np.array([])
            paired_diff = np.array([])
        
        self.meas = paired_lidar  # the LIDAR measurements that were paired
        self.o = paired_meas      # the corresponding predicted measurements (from the map)
        self.difff = paired_diff   # the difference vectors for the paired measurements


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
                self.muBar = self.muBar + correction
                self.sigmaBar = I_KH @ self.sigmaBar
                #print(f"z: {Zt}, hx: {hx}, {i}")
        self.Mu = self.muBar
        self.covariance_prev = self.sigmaBar    
        self.m_prev_x = self.m_x
        self.m_prev_y = self.m_y
        self.m_prev_0 = self.m_0

    def publish_mu_sigma(self):
        # Publish filtered pose
        if self.pub_filtered.get_subscription_count() > 0:
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
        
        self.publish_error_ellipse(self.Mu[0, 0], self.Mu[1, 0], self.covariance_prev)

    def publish_cylinder_marker_array(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        for idx, coord in enumerate(self.map_cylinders):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = now
            marker.ns = "estimated_cylinders"
            marker.id = idx
            marker.type = Marker.CYLINDER  # Changed from SPHERE to CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = coord[0]
            marker.pose.position.y = coord[1]
            marker.pose.position.z = 0.2  # Bottom of cylinder at z=0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # Diameter of cylinder
            marker.scale.y = 0.2  # Diameter of cylinder
            marker.scale.z = 0.4  # Height of cylinder
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.3
            marker.color.b = 1.0
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
        
        if marker_array.markers:
            self.marker_array_pub.publish(marker_array)
            #self.get_logger().info(f"Published {len(marker_array.markers)} cylinder markers")
    
    def publish_path_markers(self):
        now = self.get_clock().now().to_msg()
        
        # Publish raw odometry path
        if self.odom_path:
            marker_odom = Marker()
            marker_odom.header.frame_id = "odom"
            marker_odom.header.stamp = now
            marker_odom.ns = "odom_path"
            marker_odom.id = 1
            marker_odom.type = Marker.LINE_STRIP
            marker_odom.action = Marker.ADD
            marker_odom.scale.x = 0.05
            marker_odom.color.a = 1.0
            marker_odom.color.b = 1.0
            for pt in self.odom_path:
                marker_odom.points.append(pt)
            self.marker_pub.publish(marker_odom)
        
        # Publish filtered path
        if self.filtered_path:
            marker_filtered = Marker()
            marker_filtered.header.frame_id = "odom"
            marker_filtered.header.stamp = now
            marker_filtered.ns = "filtered_path"
            marker_filtered.id = 2
            marker_filtered.type = Marker.LINE_STRIP
            marker_filtered.action = Marker.ADD
            marker_filtered.scale.x = 0.05
            marker_filtered.color.a = 1.0
            marker_filtered.color.g = 1.0
            for pt in self.filtered_path:
                marker_filtered.points.append(pt)
            self.marker_pub.publish(marker_filtered)
    
    def publish_trajectory(self):
        now = self.get_clock().now().to_msg() 
        # Publish filtered path
        if self.filtered_path:
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
        # Publish raw odometry path
        if self.odom_path:
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

    def compute_error_ellipse(self, cov):
        cov_2d = np.array([[cov[0, 0], cov[0, 1]],
                           [cov[1, 0], cov[1, 1]]])
        eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
        sort_indices = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        a = math.sqrt(eigenvals[0])
        b = math.sqrt(eigenvals[1])
        angle = math.atan2(eigenvecs[1, 0], eigenvecs[0, 0])
        return a, b, angle

    def generate_ellipse_points(self, center, a, b, angle, num_pts=50):
        points = []
        for t in np.linspace(0, 2 * m.pi, num_pts):
            x = a * m.cos(t)
            y = b * m.sin(t)
            x_rot = center[0] + x * m.cos(angle) - y * m.sin(angle)
            y_rot = center[1] + x * m.sin(angle) + y * m.cos(angle)
            points.append(Point(x=x_rot, y=y_rot, z=0.0))
        points.append(points[0])
        return points
    
    def publish_error_ellipse(self, x, y, cov):
        a, b, angle = self.compute_error_ellipse(cov)
        scale_factor = 1.0  # Adjust if ellipse is too small/large
        a *= scale_factor
        b *= scale_factor
        if a > 0.001 and b > 0.001 and a < 10.0 and b < 10.0:
            center = (x, y)
            ellipse_points = self.generate_ellipse_points(center, a, b, angle, num_pts=50)
            
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "error_ellipse"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Increased line thickness for visibility
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.points = ellipse_points
            self.error_ellipse_pub.publish(marker)
        else:
            self.get_logger().warn(f"Invalid ellipse parameters: a={a:.3f}, b={b:.3f}. Not publishing.")
    
    # --- Main run() Function ---
    def run(self):
        if not self.received_data:
            self.get_logger().info("Waiting for initial data...")
            return
            
        self.control_covariance()
        self.state_covariance()
        self.process_noise()
        self.mu_mean()
        self.sigma_covariance()
        self.observed_landmarks()
        self.obs_meas_difference()
        self.correction_matrices()
        self.publish_mu_sigma()
        self.publish_cylinder_marker_array()
        self.publish_path_markers()
        self.publish_trajectory()
        
    def timer_callback(self):
        self.run()

def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKFNode()
    
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