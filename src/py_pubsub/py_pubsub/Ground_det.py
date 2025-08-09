import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import random
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler, quaternion_multiply

class GroundPlaneDetector(Node):
    def __init__(self):
        super().__init__('ground_plane_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        
        # Intrinsic parameters (adjust if necessary)
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 320.0
        self.cy = 240.0

        # Camera mounting parameters:
        self.camera_height = 0.45  # in meters
        self.camera_tilt_deg = 15  # in degrees, positive means camera looks down
        self.camera_tilt_rad = np.deg2rad(self.camera_tilt_deg)
        
        # Field of view parameters
        self.horizontal_fov = np.deg2rad(60)  # Horizontal field of view in radians
        self.depth_range = 5.0  # Maximum depth range in meters
        
        # RANSAC parameters - increased for better stability
        self.ransac_iterations = 400  # Increased for better coverage
        self.ransac_distance_threshold = 0.035  # Slightly increased for more inclusive planes
        self.ransac_min_inliers = 100  # Minimum points needed to consider a valid plane

        # Ground filtering parameters
        self.ground_height_max = 0.15  # Maximum height above estimated ground to be considered ground
        self.ground_tolerance = 0.25  # Increased for more robustness across the full FOV
        
        # Temporal consistency parameters
        self.previous_plane = None
        self.plane_consistency_threshold = 0.9  # Cosine similarity threshold for plane normals
        self.plane_smoothing_factor = 0.7  # Weight of previous plane in smoothing (lower = more adaptive)
        
        # Store the latest depth image
        self.latest_depth_image = None

        # Publisher for markers
        self.marker_pub = self.create_publisher(Marker, 'ground_markers', 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'ground_marker_array', 10)
        
        # Create a timer callback to process the image every 0.5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        # Set up static transform broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        
        self.get_logger().info("Ground Plane Detector initialized")

    def publish_static_transform(self):
        """Publish static transform from base_link to camera frame"""
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = "base_link"
        static_transform.child_frame_id = "camera_depth_optical_frame"
        
        # Set the position - camera is mounted at height on the robot, typically forward of center
        static_transform.transform.translation.x = 0.1  # Slightly forward of the base_link
        static_transform.transform.translation.y = 0.0  # Centered
        static_transform.transform.translation.z = 0.1
        
        # The optical frame in ROS has z forward, x right, y down
        # We need to rotate from the base_link frame (x forward, y left, z up)
        # This requires a -90 degree rotation around x, followed by camera tilt
        
        # First create a quaternion for optical frame alignment (-90 around X)
        q_optical = quaternion_from_euler(-np.pi/2.0, -0.26, -np.pi/2.0)
        
        # Then create a quaternion for the tilt (around X in the new frame)
        q_tilt = quaternion_from_euler(self.camera_tilt_rad, 0, 0)
        
        # Combine the rotations (optical frame alignment first, then tilt)
        q_combined = quaternion_multiply(q_tilt, q_optical)
        
        static_transform.transform.rotation.x = q_combined[0]
        static_transform.transform.rotation.y = q_combined[1]
        static_transform.transform.rotation.z = q_combined[2]
        static_transform.transform.rotation.w = q_combined[3]
        
        self.tf_broadcaster.sendTransform(static_transform)
        self.get_logger().info("Published static transform for camera frame")

    def image_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            
            # Apply some basic preprocessing to the depth image
            # Remove any NaN or inf values
            mask = np.isfinite(depth_image)
            depth_image[~mask] = 0.0
            
            self.latest_depth_image = depth_image
            self.get_logger().debug(f"Stored new depth image with shape: {depth_image.shape}")
        except Exception as e:
            self.get_logger().error("Could not convert depth image: " + str(e))

    def timer_callback(self):
        if self.latest_depth_image is None:
            self.get_logger().warn("No depth image available yet.")
            return
        
        depth_image = self.latest_depth_image.copy()  # Make a copy to avoid modifying original
        self.get_logger().debug(f"Processing depth image with shape: {depth_image.shape}")

        # Generate the point cloud with improved density sampling
        points = self.create_point_cloud(depth_image)
        if points.size == 0:
            self.get_logger().warn("No valid points in depth image.")
            return

        # Apply initial filtering - remove NaN values and points too far away
        valid_indices = ~np.isnan(points).any(axis=1)
        points = points[valid_indices]
        
        # Remove points that are too far away
        distances = np.linalg.norm(points, axis=1)
        points = points[distances < self.depth_range]
        
        if points.shape[0] < self.ransac_min_inliers:
            self.get_logger().warn(f"Not enough valid points after filtering: {points.shape[0]}")
            return
        
        # Initial filtering to find potential ground points across the full FOV
        # Calculate the expected ground plane considering camera parameters
        z_distances = points[:, 2]
        
        # Calculate expected ground plane Y value considering radial distance
        # In camera optical frame, ground points should follow a pattern based on Z distance and tilt
        radial_distance = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 2]))
        expected_ground_y = np.sin(self.camera_tilt_rad) * z_distances - self.camera_height/np.cos(self.camera_tilt_rad)
        
        # Adjust ground tolerance for far points (points at edges of FOV may have larger variation)
        # This creates a distance-dependent tolerance that increases with distance from camera
        adaptive_tolerance = self.ground_tolerance * (1.0 + 0.5 * radial_distance / self.depth_range)
        
        # Points are likely ground if they are close to the expected ground plane
        ground_diff = np.abs(points[:, 1] - expected_ground_y)
        potential_ground = points[ground_diff < adaptive_tolerance]
        
        self.get_logger().debug(f"Points before filtering: {points.shape[0]}, potential ground: {potential_ground.shape[0]}")
        
        if potential_ground.shape[0] < self.ransac_min_inliers:
            self.get_logger().warn("Not enough potential ground points for RANSAC.")
            # Fall back to using all points with relaxed filtering
            potential_ground = points
        
        # Use a stratified sampling approach for RANSAC to ensure coverage across the full FOV
        # Divide the FOV into regions and sample from each region
        x_points = potential_ground[:, 0]
        z_points = potential_ground[:, 2]
        
        # Run RANSAC to fit a plane on the potential ground points
        plane, inlier_indices = self.stratified_ransac_plane(
            potential_ground, 
            iterations=self.ransac_iterations, 
            distance_threshold=self.ransac_distance_threshold
        )
        
        if plane is None or len(inlier_indices) < self.ransac_min_inliers:
            self.get_logger().warn(f"No plane found or too few inliers: {0 if inlier_indices is None else len(inlier_indices)}")
            return
        
        # Apply temporal consistency check
        plane = self.ensure_temporal_consistency(plane)
        
        # Use the fitted plane to classify all points in the full FOV
        normal, d = plane
        distances_to_plane = np.abs(np.dot(points, normal) + d)
        
        # Points are ground if close to the plane and not too high above it
        # Scale threshold with distance for better coverage at edges
        radial_distance = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 2]))
        adaptive_threshold = self.ransac_distance_threshold * (1.0 + 0.5 * radial_distance / self.depth_range)
        ground_mask = distances_to_plane < adaptive_threshold
        
        # Additional check: project points onto plane normal to determine if they're above or below
        # This helps distinguish objects on ground from the ground itself
        signed_distances = np.dot(points, normal) + d
        
        # Points significantly above the plane (positive distance) are objects
        # Using a height threshold that scales with distance from camera
        height_threshold = self.ground_height_max * (1.0 + 0.2 * radial_distance / self.depth_range)
        above_plane = signed_distances > height_threshold
        
        # Final ground mask: close to plane AND not significantly above it
        ground_mask = ground_mask & ~above_plane
        
        # Split points into ground and non-ground
        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]
        
        self.get_logger().info(f"Ground points: {ground_points.shape[0]}, Other objects: {non_ground_points.shape[0]}")

        # Publish markers
        self.publish_enhanced_markers(ground_points, non_ground_points)
        self.publish_enhanced_marker_array(ground_points, non_ground_points)

    def create_point_cloud(self, depth_image):
        """Convert depth image to 3D point cloud with improved density sampling."""
        # Basic depth image filtering
        depth_image[depth_image > self.depth_range] = 0.0  # Remove very far points
        depth_image[depth_image < 0.1] = 0.0   # Remove very close points (likely noise)
        
        height, width = depth_image.shape
        
        # Create evenly distributed sampling grid
        # This gives us better coverage across the entire FOV
        step = 2  # Adjust step size based on desired density and performance
        v, u = np.mgrid[0:height:step, 0:width:step]
        
        # Flatten for easier processing
        u = u.flatten()
        v = v.flatten()
        
        # Sample the depth image at grid points
        indices = v * width + u
        valid_indices = (indices >= 0) & (indices < width * height)
        u = u[valid_indices]
        v = v[valid_indices]
        
        # Get depth values
        z = np.zeros_like(u, dtype=np.float32)
        for i in range(len(u)):
            if v[i] < height and u[i] < width:
                z[i] = depth_image[v[i], u[i]]
        
        # Convert to 3D coordinates using camera intrinsics
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Filter out invalid points
        valid = z > 0.01
        points = np.vstack((x[valid], y[valid], z[valid])).T
        
        return points

    def stratified_ransac_plane(self, points, iterations=300, distance_threshold=0.03):
        """Fit a plane to points using RANSAC with stratified sampling across FOV."""
        best_inliers = []
        best_plane = None
        num_points = points.shape[0]
        
        if num_points < 3:
            return None, None

        # Divide the FOV into regions based on X and Z coordinates
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        
        # Create a grid of regions
        x_regions = 3  # Number of regions along X axis
        z_regions = 3  # Number of regions along Z axis
        
        x_edges = np.linspace(x_min, x_max, x_regions + 1)
        z_edges = np.linspace(z_min, z_max, z_regions + 1)
        
        # Create region assignments for each point
        x_bins = np.digitize(points[:, 0], x_edges) - 1
        z_bins = np.digitize(points[:, 2], z_edges) - 1
        
        # Ensure valid bin indices
        x_bins = np.clip(x_bins, 0, x_regions - 1)
        z_bins = np.clip(z_bins, 0, z_regions - 1)
        
        # Create a combined region ID for each point
        region_ids = x_bins * z_regions + z_bins
        
        # Count how many iterations to allocate per region based on point density
        unique_regions, region_counts = np.unique(region_ids, return_counts=True)
        iterations_per_region = {}
        total_regions = len(unique_regions)
        
        if total_regions == 0:
            return None, None
            
        # Allocate iterations proportionally to region point count, with a minimum
        for region, count in zip(unique_regions, region_counts):
            iterations_per_region[region] = max(10, int(iterations * count / num_points))

        for i in range(iterations):
            # Choose a region based on the current iteration
            current_region = i % total_regions
            region_id = unique_regions[current_region]
            
            # Get points from this region
            region_points_mask = region_ids == region_id
            region_points = points[region_points_mask]
            
            if len(region_points) < 3:
                continue
                
            # Randomly sample 3 points from this region
            idx = random.sample(range(len(region_points)), 3)
            sample = region_points[idx, :]
            p1, p2, p3 = sample

            # Calculate plane from 3 points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_val = np.linalg.norm(normal)
            
            if norm_val == 0:
                continue
                
            normal = normal / norm_val
            d = -np.dot(normal, p1)

            # Ensure normal points "up" in camera frame
            # In camera optical frame, "up" is negative Y
            if normal[1] < 0:  # In camera optical frame, ground normal should point up (negative Y)
                normal = -normal
                d = -d

            # Calculate distances from all points to the plane
            distances = np.abs(np.dot(points, normal) + d)
            inlier_indices = np.where(distances < distance_threshold)[0]

            # Update best model if we found more inliers
            if len(inlier_indices) > len(best_inliers):
                best_inliers = inlier_indices
                best_plane = (normal, d)

        return best_plane, best_inliers

    def ensure_temporal_consistency(self, current_plane):
        """Ensure the detected plane is consistent with previous frames."""
        if current_plane is None:
            return self.previous_plane
            
        current_normal, current_d = current_plane
        
        if self.previous_plane is None:
            self.previous_plane = current_plane
            return current_plane
            
        previous_normal, previous_d = self.previous_plane
        
        # Check consistency using cosine similarity between normals
        cos_similarity = np.dot(current_normal, previous_normal)
        
        if cos_similarity > self.plane_consistency_threshold:
            # Smooth the transition
            alpha = self.plane_smoothing_factor
            smooth_normal = alpha * previous_normal + (1 - alpha) * current_normal
            smooth_normal = smooth_normal / np.linalg.norm(smooth_normal)
            smooth_d = alpha * previous_d + (1 - alpha) * current_d
            
            self.previous_plane = (smooth_normal, smooth_d)
            return self.previous_plane
        else:
            self.get_logger().warn(f"Plane changed significantly! Cos similarity: {cos_similarity:.3f}")
            
            # Keep the previous plane if the change is too drastic
            if cos_similarity < 0.5:  # Very different planes
                return self.previous_plane
            else:
                # Accept new plane but with heavy smoothing
                alpha = 0.9  # Even higher weight to previous
                smooth_normal = alpha * previous_normal + (1 - alpha) * current_normal
                smooth_normal = smooth_normal / np.linalg.norm(smooth_normal)
                smooth_d = alpha * previous_d + (1 - alpha) * current_d
                
                self.previous_plane = (smooth_normal, smooth_d)
                return self.previous_plane

    def publish_enhanced_markers(self, ground_points, non_ground_points):
        """Publish enhanced markers for ground and non-ground points"""
        # Clear previous markers
        self.clear_markers()
        
        # Sample points if there are too many
        max_ground_points = 3000  # Increased for better ground coverage
        max_object_points = 2000  # Increased for better object visualization
        
        if len(ground_points) > max_ground_points:
            indices = np.random.choice(len(ground_points), max_ground_points, replace=False)
            ground_points_sample = ground_points[indices]
        else:
            ground_points_sample = ground_points
            
        if len(non_ground_points) > max_object_points:
            indices = np.random.choice(len(non_ground_points), max_object_points, replace=False)
            non_ground_points_sample = non_ground_points[indices]
        else:
            non_ground_points_sample = non_ground_points
        
        # GROUND POINTS (RED)
        ground_marker = Marker()
        ground_marker.header = Header()
        ground_marker.header.stamp = self.get_clock().now().to_msg()
        ground_marker.header.frame_id = "camera_depth_optical_frame"
        ground_marker.ns = "ground_points"
        ground_marker.id = 0
        ground_marker.type = Marker.POINTS
        ground_marker.action = Marker.ADD
        ground_marker.pose.orientation.w = 1.0
        
        # Make points larger for better visibility
        ground_marker.scale.x = 0.02  # Slightly smaller for better performance with more points
        ground_marker.scale.y = 0.02
        
        # RED for ground plane
        ground_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        
        # Add points
        for pt in ground_points_sample:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = float(pt[2])
            ground_marker.points.append(point)
        
        # NON-GROUND POINTS (GREEN)
        object_marker = Marker()
        object_marker.header = Header()
        object_marker.header.stamp = self.get_clock().now().to_msg()
        object_marker.header.frame_id = "camera_depth_optical_frame"
        object_marker.ns = "object_points"
        object_marker.id = 1
        object_marker.type = Marker.POINTS
        object_marker.action = Marker.ADD
        object_marker.pose.orientation.w = 1.0
        
        # Make points larger for better visibility
        object_marker.scale.x = 0.02
        object_marker.scale.y = 0.02
        
        # GREEN for objects
        object_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        
        # Add points
        for pt in non_ground_points_sample:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = float(pt[2])
            object_marker.points.append(point)
        
        # Publish markers
        self.marker_pub.publish(ground_marker)
        self.marker_pub.publish(object_marker)
        self.get_logger().debug(f"Published ground marker (RED) with {len(ground_marker.points)} points")
        self.get_logger().debug(f"Published object marker (GREEN) with {len(object_marker.points)} points")

    def publish_enhanced_marker_array(self, ground_points, non_ground_points):
        """Publish enhanced marker array for ground and non-ground points"""
        # Sample points for visualization
        max_ground_points = 3000
        max_object_points = 2000
        
        if len(ground_points) > max_ground_points:
            indices = np.random.choice(len(ground_points), max_ground_points, replace=False)
            ground_points_sample = ground_points[indices]
        else:
            ground_points_sample = ground_points
            
        if len(non_ground_points) > max_object_points:
            indices = np.random.choice(len(non_ground_points), max_object_points, replace=False)
            non_ground_points_sample = non_ground_points[indices]
        else:
            non_ground_points_sample = non_ground_points
        
        # GROUND POINTS (RED)
        ground_marker = Marker()
        ground_marker.header = Header()
        ground_marker.header.stamp = self.get_clock().now().to_msg()
        ground_marker.header.frame_id = "camera_depth_optical_frame"
        ground_marker.ns = "ground_points_array"
        ground_marker.id = 0
        ground_marker.type = Marker.POINTS
        ground_marker.action = Marker.ADD
        ground_marker.pose.orientation.w = 1.0
        ground_marker.scale.x = 0.02
        ground_marker.scale.y = 0.02
        ground_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # RED
        
        for pt in ground_points_sample:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = float(pt[2])
            ground_marker.points.append(point)
        
        # NON-GROUND POINTS (GREEN)
        object_marker = Marker()
        object_marker.header = Header()
        object_marker.header.stamp = self.get_clock().now().to_msg()
        object_marker.header.frame_id = "camera_depth_optical_frame"
        object_marker.ns = "object_points_array"
        object_marker.id = 1
        object_marker.type = Marker.POINTS
        object_marker.action = Marker.ADD
        object_marker.pose.orientation.w = 1.0
        object_marker.scale.x = 0.02
        object_marker.scale.y = 0.02
        object_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # GREEN
        
        for pt in non_ground_points_sample:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = float(pt[2])
            object_marker.points.append(point)
        
        # Create marker array and publish
        marker_array = MarkerArray()
        marker_array.markers = [ground_marker, object_marker]
        self.marker_array_pub.publish(marker_array)
        self.get_logger().debug(f"Published marker array with {len(ground_marker.points)} ground points (RED) and {len(object_marker.points)} object points (GREEN)")

    def clear_markers(self):
        """Clear all previous markers to avoid visual clutter"""
        # Create and publish deletion markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "camera_depth_optical_frame"
        delete_marker.action = Marker.DELETEALL
        self.marker_pub.publish(delete_marker)
        
        delete_all_array = MarkerArray()
        delete_marker_ground = Marker()
        delete_marker_ground.header.frame_id = "camera_depth_optical_frame"
        delete_marker_ground.ns = "ground_points_array"
        delete_marker_ground.id = 0
        delete_marker_ground.action = Marker.DELETE
        
        delete_marker_objects = Marker()
        delete_marker_objects.header.frame_id = "camera_depth_optical_frame"
        delete_marker_objects.ns = "object_points_array"
        delete_marker_objects.id = 1
        delete_marker_objects.action = Marker.DELETE
        
        delete_all_array.markers = [delete_marker_ground, delete_marker_objects]
        self.marker_array_pub.publish(delete_all_array)
        self.get_logger().debug("Cleared previous markers")

def main(args=None):
    rclpy.init(args=args)
    node = GroundPlaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("GroundPlaneDetector node stopped cleanly.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()