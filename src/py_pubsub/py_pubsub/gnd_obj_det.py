import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class GroundPlaneDetector(Node):
    def __init__(self):
        super().__init__('ground_plane_detector')
        
        # Create subscription to depth image
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.image_callback,
            10
        )
        
        # Create publisher for the binary ground mask
        self.binary_pub = self.create_publisher(Image, 'ground_binary_image', 10)
        self.bridge = CvBridge()
        
        # Parameters for ground plane detection
        self.max_depth = 8.0        # Maximum depth in meters for valid points
        self.height_threshold = 0.10   # Allowed variation around the estimated ground height
        self.min_points_ratio = 0.05   # Min fraction of depth pixels needed for reliable detection
        
        # Camera intrinsics (adjust to your camera)
        self.fx = 525.0
        self.fy = 525.0
        # cx and cy will be set dynamically once we know image width, height
        
        # Parameters for morphological filtering
        self.noise_removal_kernel = 5  # Kernel size for morphological open/close
        
        # Ground plane estimation parameters
        self.ground_plane_model = None  # Will store (a, b, c, d) for plane equation ax + by + cz + d = 0
        self.debug = True              # Set to True for detailed logs
        
        # For storing the latest result
        self.latest_binary_mask = None
        self.latest_header = None
        
        # Create a timer for publishing the binary mask at 0.5s intervals (2 Hz)
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        self.get_logger().info("GroundPlaneDetector initialized with 0.5s publishing rate")

    def image_callback(self, msg):
        """Callback to process the incoming depth image and create a binary mask."""
        try:
            # Convert ROS Image (depth) -> OpenCV format (32FC1)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Could not convert depth image: {e}")
            return
        
        height, width = depth_image.shape
        
        # Update the camera center if needed
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # We start with an all-black mask, since we want ground to be white
        binary_mask = np.zeros(depth_image.shape, dtype=np.uint8)
        
        # Filter out invalid or too-distant depth pixels for plane estimation
        valid_mask = (depth_image > 0) & (depth_image < self.max_depth)
        valid_count = np.count_nonzero(valid_mask)
        
        # Check if we have enough valid points to do any estimation
        if valid_count < self.min_points_ratio * height * width:
            self.get_logger().warn("Not enough valid depth points. Publishing blank mask.")
            self.latest_binary_mask = binary_mask
            self.latest_header = msg.header
            return
        
        # Indices of valid pixels
        v_indices, u_indices = np.indices(depth_image.shape)
        valid_v = v_indices[valid_mask]
        valid_u = u_indices[valid_mask]
        valid_z = depth_image[valid_mask]
        
        # Convert to 3D camera coordinates
        valid_x = (valid_u - self.cx) * valid_z / self.fx
        valid_y = (valid_v - self.cy) * valid_z / self.fy
        
        # RANSAC-like approach to estimate ground plane
        best_inliers = 0
        best_ground_height = None
        best_plane_model = None
        num_iterations = 50
        
        # We'll focus on the bottom portion of the image for ground-plane sampling
        bottom_mask = valid_v > (0.6 * height)  # e.g., lower 40% of the image
        bottom_indices = np.where(bottom_mask)[0]
        
        # If even the bottom portion is too sparse, we won't find a consistent plane
        if len(bottom_indices) < 10:
            self.get_logger().warn("Not enough points in the bottom region to estimate ground.")
            self.latest_binary_mask = binary_mask
            self.latest_header = msg.header
            return
        
        for _ in range(num_iterations):
            # Randomly sample up to 20 points from the bottom region
            sample_count = min(20, len(bottom_indices))
            sample_idxs = np.random.choice(bottom_indices, sample_count, replace=False)
            
            # Grab their y-coordinates
            sample_y = valid_y[sample_idxs]
            # Use median as the "candidate ground height"
            candidate_height = np.median(sample_y)
            
            # Count how many points in the entire valid set are within height_threshold
            inliers_mask = np.abs(valid_y - candidate_height) < self.height_threshold
            inliers = np.count_nonzero(inliers_mask)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_ground_height = candidate_height
                # Store a simple plane model (horizontal plane at this height)
                best_plane_model = (0, 1, 0, -candidate_height)  # y = candidate_height
        
        # If we failed to get a ground plane
        if best_ground_height is None:
            self.get_logger().warn("Failed to find ground plane. Publishing blank mask.")
            self.latest_binary_mask = binary_mask
            self.latest_header = msg.header
            return
        
        # Debug logs
        if self.debug:
            self.get_logger().info(f"Best ground height: {best_ground_height:.3f} (inliers: {best_inliers})")
        
        # Store the ground plane model
        self.ground_plane_model = best_plane_model
        
        # Project ground plane across the entire image, even behind objects
        # For every pixel in the image:
        for v in range(height):
            for u in range(width):
                # Calculate the ray from camera through this pixel
                z_depth = 1.0  # We'll use a unit vector for the ray direction
                x = (u - self.cx) * z_depth / self.fx
                y = (v - self.cy) * z_depth / self.fy
                
                # Normalize the ray direction
                ray_length = np.sqrt(x*x + y*y + z_depth*z_depth)
                x /= ray_length
                y /= ray_length
                z_depth /= ray_length
                
                # Check if ray intersects with ground plane
                # Plane: ax + by + cz + d = 0
                # Ray: camera_pos + t * ray_dir
                a, b, c, d = best_plane_model
                t = -d / (a*x + b*y + c*z_depth)
                
                # If intersection is in front of camera and within reasonable range
                if t > 0 and t < self.max_depth:
                    # Calculate 3D point of intersection
                    intersect_x = t * x
                    intersect_y = t * y
                    intersect_z = t * z_depth
                    
                    # If this point is near the ground height, mark it as ground
                    if abs(intersect_y - best_ground_height) < self.height_threshold:
                        binary_mask[v, u] = 255
        
        # Now handle the actual observed depth points to refine the mask
        # For observed points, we might want to override our plane projection
        for i in range(len(valid_v)):
            v, u, z = valid_v[i], valid_u[i], valid_z[i]
            y_coord = valid_y[i]
            
            # If this point is clearly not on the ground plane
            if abs(y_coord - best_ground_height) > self.height_threshold:
                # If it's above the ground (e.g., object), mark it as not ground
                if y_coord < best_ground_height:  # Remember: y is up-down in camera frame
                    binary_mask[v, u] = 0
        
        # Morphological operations to clean up noise
        if self.noise_removal_kernel > 1:
            kernel = np.ones((self.noise_removal_kernel, self.noise_removal_kernel), np.uint8)
            # Close small holes
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            # Remove small specks
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Store the final result for periodic publishing
        self.latest_binary_mask = binary_mask
        self.latest_header = msg.header
        
        if self.debug:
            white_pixels = np.count_nonzero(binary_mask == 255)
            self.get_logger().info(
                f"Detected ground coverage: {100.0 * white_pixels / (height * width):.2f}%"
            )

    def timer_callback(self):
        """Publish the latest binary mask at 2 Hz."""
        if self.latest_binary_mask is not None and self.latest_header is not None:
            out_msg = self.bridge.cv2_to_imgmsg(self.latest_binary_mask, encoding="mono8")
            out_msg.header = self.latest_header
            self.binary_pub.publish(out_msg)
            
            if self.debug:
                self.get_logger().info("Published ground binary mask")

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