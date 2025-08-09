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
        
        # Create publisher for the segmentation mask
        self.binary_pub = self.create_publisher(Image, 'ground_binary_image', 10)
        self.bridge = CvBridge()
        
        # Parameters for ground plane detection
        self.max_depth = 8.0             # Maximum depth in meters for valid points
        self.height_threshold = 0.10     # Allowed variation around the estimated ground height
        self.min_points_ratio = 0.05     # Min fraction of depth pixels needed for reliable detection
        self.sky_threshold = 0.1         # Depth values below this are considered invalid/sky
        
        # Camera intrinsics (adjust to your camera)
        self.fx = 525.0
        self.fy = 525.0
        # cx and cy will be set dynamically once we know image width, height
        
        # Parameters for morphological filtering
        self.noise_removal_kernel = 5    # Kernel size for morphological open/close
        self.debug = True                # Set to True for detailed logs
        
        # For storing the latest result
        self.latest_mask = None
        self.latest_header = None
        
        # Create a timer for publishing the binary mask at 0.5s intervals (2 Hz)
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        self.get_logger().info("GroundPlaneDetector initialized with 0.5s publishing rate")

    def image_callback(self, msg):
        """Callback to process the incoming depth image and create a segmentation mask."""
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
        
        # We'll use a three-class segmentation:
        # 0 = sky/background (black)
        # 127 = objects (gray)
        # 255 = ground (white)
        segmentation_mask = np.zeros(depth_image.shape, dtype=np.uint8)
        
        # First, identify sky vs valid measurements
        # Sky typically has invalid or extremely far measurements
        sky_mask = (depth_image <= self.sky_threshold) | (depth_image >= self.max_depth)
        valid_mask = ~sky_mask
        
        # Mark all valid points as objects initially (gray)
        segmentation_mask[valid_mask] = 127
        
        valid_count = np.count_nonzero(valid_mask)
        
        # Check if we have enough valid points to do any estimation
        if valid_count < self.min_points_ratio * height * width:
            self.get_logger().warn("Not enough valid depth points. Publishing only sky/object distinction.")
            self.latest_mask = segmentation_mask
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
        
        # RANSAC-like approach to estimate "ground height" via sampling from the lower portion
        best_inliers = 0
        best_ground_height = None
        num_iterations = 50
        
        # We'll focus on the bottom portion of the image for ground-plane sampling
        bottom_mask = valid_v > (0.6 * height)  # e.g., lower 40% of the image
        bottom_indices = np.where(bottom_mask)[0]
        
        # If even the bottom portion is too sparse, we won't find a consistent plane
        if len(bottom_indices) < 10:
            self.get_logger().warn("Not enough points in the bottom region to estimate ground.")
            self.latest_mask = segmentation_mask
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
        
        # If we failed to get a ground plane
        if best_ground_height is None:
            self.get_logger().warn("Failed to find ground plane. Publishing sky/object distinction only.")
            self.latest_mask = segmentation_mask
            self.latest_header = msg.header
            return
        
        # Debug logs
        if self.debug:
            self.get_logger().info(f"Best ground height: {best_ground_height:.3f} (inliers: {best_inliers})")
        
        # For all valid points, check if they lie near the ground height
        ground_inliers_mask = np.abs(valid_y - best_ground_height) < self.height_threshold
        # The (v, u) coordinates of ground pixels
        ground_v = valid_v[ground_inliers_mask]
        ground_u = valid_u[ground_inliers_mask]
        
        # Mark these ground pixels as white (255)
        segmentation_mask[ground_v, ground_u] = 255
        
        # Morphological operations to clean up noise
        if self.noise_removal_kernel > 1:
            kernel = np.ones((self.noise_removal_kernel, self.noise_removal_kernel), np.uint8)
            
            # Process ground plane (255) and objects (127) separately
            
            # First, extract ground mask
            ground_mask = (segmentation_mask == 255).astype(np.uint8) * 255
            # Clean up ground
            ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
            ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)
            
            # Extract object mask
            object_mask = (segmentation_mask == 127).astype(np.uint8) * 127
            # Clean up objects
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
            
            # Combine the masks (prioritizing ground over objects)
            segmentation_mask = np.zeros_like(segmentation_mask)
            segmentation_mask[object_mask == 127] = 127
            segmentation_mask[ground_mask == 255] = 255
        
        # Store the final result for periodic publishing
        self.latest_mask = segmentation_mask
        self.latest_header = msg.header
        
        if self.debug:
            ground_pixels = np.count_nonzero(segmentation_mask == 255)
            object_pixels = np.count_nonzero(segmentation_mask == 127)
            sky_pixels = np.count_nonzero(segmentation_mask == 0)
            self.get_logger().info(
                f"Segmentation results - Ground: {100.0 * ground_pixels / (height * width):.2f}%, "
                f"Objects: {100.0 * object_pixels / (height * width):.2f}%, "
                f"Sky: {100.0 * sky_pixels / (height * width):.2f}%"
            )

    def timer_callback(self):
        """Publish the latest segmentation mask at 2 Hz."""
        if self.latest_mask is not None and self.latest_header is not None:
            out_msg = self.bridge.cv2_to_imgmsg(self.latest_mask, encoding="mono8")
            out_msg.header = self.latest_header
            self.binary_pub.publish(out_msg)
            
            if self.debug:
                self.get_logger().info("Published ground segmentation mask")

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