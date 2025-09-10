#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from tf_transformations import quaternion_from_euler

class EKFSLAMViz(Node):
    def __init__(self):
        super().__init__('ekf_slam_viz')

        # Subscribers
        self.create_subscription(PoseWithCovarianceStamped, '/filtered_data', self.pose_callback, 10)
        self.create_subscription(Float32MultiArray, '/slam_state', self.state_callback, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/slam_path', 10)
        self.landmarks_pub = self.create_publisher(MarkerArray, '/slam_landmarks', 10)

        # Store robot trajectory
        self.path = Path()
        self.path.header.frame_id = "odom"

        self.get_logger().info("EKF SLAM Visualization Node started")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Append robot pose to Path and publish it."""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.path.poses.append(pose_stamped)
        self.path_pub.publish(self.path)
        self.get_logger().debug(f"Path updated, total poses: {len(self.path.poses)}")

    def state_callback(self, msg: Float32MultiArray):
        """Visualize landmarks from the state vector."""
        data = msg.data
        if len(data) < 3:
            self.get_logger().warn("Received /slam_state but contains no robot state")
            return

        # Extract landmarks (after first 3 values: x, y, theta)
        landmarks = [(data[i], data[i+1]) for i in range(3, len(data), 2)]

        self.get_logger().info(f"Received state with {len(landmarks)} landmarks")

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for idx, (x, y) in enumerate(landmarks):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = now
            marker.ns = "slam_landmarks"
            marker.id = idx
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.3

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        if marker_array.markers:
            self.landmarks_pub.publish(marker_array)
            self.get_logger().info(f"Published {len(marker_array.markers)} landmark markers")
        else:
            self.get_logger().warn("No landmarks to publish in this state")

def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAMViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
