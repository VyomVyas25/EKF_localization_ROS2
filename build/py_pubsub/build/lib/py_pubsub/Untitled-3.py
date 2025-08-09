#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import argparse
import rclpy
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from std_msgs.msg import ColorRGBA
import tf_transformations as tf  # For quaternion conversion
from tf2_ros import StaticTransformBroadcaster


def compute_line_intersection_2d(p, d, q, e):
    """
    Compute the intersection between two lines in 2D:
    Line1: p + t*d
    Line2: q + s*e
    Returns (t, s, intersection) if lines are not parallel, else None.
    """
    denom = d[0] * e[1] - d[1] * e[0]
    if abs(denom) < 1e-6:
        return None  # Lines are parallel or nearly so
    delta = q - p
    t = (delta[0] * e[1] - delta[1] * e[0]) / denom
    s = (delta[0] * d[1] - delta[1] * d[0]) / denom
    intersection = p + t * d
    return t, s, intersection


class WallDetectorRViz(Node):
    def __init__(self):
        super().__init__('wall_detector')
        self.marker_pub = self.create_publisher(MarkerArray, 'wall_markers', 10)
        
        # Publish a static transform for "odom" (here, an identity transform).
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.send_odom_transform()
        
        # RANSAC parameters
        self.ransac_iterations = 150000
        self.ransac_distance_threshold = 0.04
        self.ransac_min_inliers = 30
        
        # Wall filtering parameters
        self.verticality_threshold = 0.25
        self.min_wall_area = 0.5
        self.min_height = 0.6
        
        # Clustering parameters
        self.cluster_eps = 0.2
        self.cluster_min_points = 15
        
        # Visualization parameters
        self.colors = [
            [1.0, 0.0, 0.0, 0.7],
            [0.0, 1.0, 0.0, 0.7],
            [0.0, 0.0, 1.0, 0.7],
            [1.0, 1.0, 0.0, 0.7],
            [1.0, 0.0, 1.0, 0.7],
            [0.0, 1.0, 1.0, 0.7],
            [0.8, 0.4, 0.0, 0.7],
            [0.5, 0.0, 0.5, 0.7],
            [0.0, 0.5, 0.5, 0.7],
            [0.5, 0.5, 0.0, 0.7],
            [0.7, 0.3, 0.3, 0.7],
            [0.3, 0.7, 0.3, 0.7],
        ]
        
        # Each wall is (points, plane_model, [width, height])
        self.walls = []
        self.wall_markers = MarkerArray()
        self.full_pcd = None
        
        # Visualization frame:
        self.frame_id = "odom"
        
        # Additional parameters
        self.wall_thickness = 0.02
        self.exclude_similar_planes = True
        self.plane_similarity_threshold = 0.92
        self.distance_threshold_for_merge = 0.15
        
        self.get_logger().info("Wall detector initialized and ready for RViz visualization")
    
    def send_odom_transform(self):
        """Publish an identity static transform for the 'odom' frame."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "odom"  # Identity
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform([t])
        self.get_logger().info("Published static transform for the odom frame")
    
    def load_point_cloud(self, pcd_file):
        """Load a point cloud from a PCD file."""
        self.get_logger().info(f"Loading point cloud from {pcd_file}...")
        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            self.get_logger().info(f"Loaded {len(pcd.points)} points.")
            return pcd
        except Exception as e:
            self.get_logger().error(f"Error loading point cloud: {e}")
            return None
    
    def preprocess_point_cloud(self, pcd):
        """Preprocess the point cloud (outlier removal, downsampling, normal estimation)."""
        self.get_logger().info("Preprocessing point cloud...")
        self.get_logger().info("Removing outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.2)
        self.get_logger().info("Downsampling point cloud...")
        pcd = pcd.voxel_down_sample(voxel_size=0.025)
        if not pcd.has_normals():
            self.get_logger().info("Estimating normals...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=60))
            pcd.orient_normals_consistent_tangent_plane(k=40)
        self.get_logger().info(f"Preprocessed point cloud has {len(pcd.points)} points.")
        return pcd
    
    def segment_walls(self, pcd):
        """Segment walls from the point cloud using RANSAC and cluster each plane's points."""
        self.get_logger().info("Segmenting walls...")
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        gravity_direction = np.array([0, 0, 1])
        
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(points)
        remaining_pcd.normals = o3d.utility.Vector3dVector(normals)
        if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
            remaining_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        
        self.walls = []
        max_planes = 20
        min_remaining_points = 30
        processed_planes = []
        
        for i in range(max_planes):
            if len(remaining_pcd.points) < min_remaining_points:
                self.get_logger().info(f"Too few points remain: {len(remaining_pcd.points)}")
                break
            
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=self.ransac_distance_threshold,
                ransac_n=3,
                num_iterations=self.ransac_iterations
            )
            
            if len(inliers) < self.ransac_min_inliers:
                self.get_logger().info(f"Not enough inliers for plane {i+1}: {len(inliers)}")
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                continue
            
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            verticality = abs(np.dot(normal, gravity_direction))
            
            # Check for duplicates
            is_duplicate = False
            if self.exclude_similar_planes:
                for processed_normal, _, _ in processed_planes:
                    similarity = abs(np.dot(normal, processed_normal))
                    if similarity > self.plane_similarity_threshold:
                        inlier_cloud = remaining_pcd.select_by_index(inliers)
                        inlier_centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
                        for _, prev_plane_model, _ in processed_planes:
                            prev_a, prev_b, prev_c, prev_d = prev_plane_model
                            dist = abs(prev_a * inlier_centroid[0] + prev_b * inlier_centroid[1] +
                                       prev_c * inlier_centroid[2] + prev_d) / np.sqrt(prev_a**2 + prev_b**2 + prev_c**2)
                            if dist < self.distance_threshold_for_merge:
                                is_duplicate = True
                                break
                    if is_duplicate:
                        break
            
            if is_duplicate:
                self.get_logger().info(f"Skipping duplicate plane {i+1}")
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                continue
            
            if verticality < self.verticality_threshold:
                self.get_logger().info(f"Found wall {i+1} with {len(inliers)} points. Verticality: {verticality:.4f}")
                inlier_cloud = remaining_pcd.select_by_index(inliers)
                inlier_points = np.asarray(inlier_cloud.points)
                h, w = self.calculate_wall_dimensions(inlier_points, normal)
                area = h * w
                processed_planes.append((normal, plane_model, inlier_points))
                if area > self.min_wall_area and h > self.min_height:
                    self.get_logger().info(f" Valid wall segment: {len(inlier_points)} pts, area={area:.2f}m², height={h:.2f}m")
                    self.walls.append((inlier_points, plane_model, [w, h]))
                    # Now cluster
                    segments = self.cluster_wall_segments(inlier_points)
                    for j, seg in enumerate(segments):
                        if len(seg) < self.cluster_min_points:
                            continue
                        h2, w2 = self.calculate_wall_dimensions(seg, normal)
                        area2 = h2 * w2
                        if area2 > self.min_wall_area and h2 > self.min_height:
                            self.get_logger().info(f"  Additional wall segment {j+1}: {len(seg)} pts, area={area2:.2f}m², height={h2:.2f}m")
                            self.walls.append((seg, plane_model, [w2, h2]))
            else:
                self.get_logger().info(f"Plane {i+1} is not vertical. Verticality: {verticality:.4f}")
            
            # Remove the inliers from the remaining cloud
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            self.get_logger().info(f"Remaining points: {len(remaining_pcd.points)}")
        
        self.post_process_walls()  # Merge or refine
        self.get_logger().info(f"Detected {len(self.walls)} walls after post-processing.")
        return self.walls
    
    def post_process_walls(self):
        """Replace with plane-group merging that also checks XY bounding boxes for closeness."""
        if len(self.walls) <= 1:
            return
        self.get_logger().info("Post-processing walls: merging walls on the same plane that overlap in XY.")
        self.walls = self.merge_wall_groups()
    
    def merge_wall_groups(self):
        """
        Merge walls on the same plane but also ensure their XY bounding boxes overlap or are close.
        This helps avoid distant walls from merging (like your aqua wall).
        """
        groups = self.group_walls_by_plane()
        merged_walls = []
        for group in groups:
            # Merge only subsets of walls that have overlapping XY bounding boxes
            # We'll do a union‐find or iterative approach to combine bounding boxes that overlap
            subwalls = group['walls']  # each is (points, plane_model, [width, height])
            # We'll treat each wall as a "node"; if bounding boxes overlap, we union them.
            
            # Build adjacency list based on bounding box overlap
            n = len(subwalls)
            adjacency = [[] for _ in range(n)]
            
            # Precompute XY bounding boxes for each wall
            bboxes = []
            for i, (pts, plane_model, dims) in enumerate(subwalls):
                min_xy, max_xy = self.compute_xy_min_max(pts)
                bboxes.append((min_xy, max_xy))
            
            overlap_threshold = 0.2  # Allow up to 20 cm gap to consider "overlapping"
            
            # Check pairwise bounding‐box overlap
            for i in range(n):
                (min_xy_i, max_xy_i) = bboxes[i]
                for j in range(i+1, n):
                    (min_xy_j, max_xy_j) = bboxes[j]
                    if self.xy_bboxes_overlap(min_xy_i, max_xy_i, min_xy_j, max_xy_j, overlap_threshold):
                        adjacency[i].append(j)
                        adjacency[j].append(i)
            
            # Now union them by BFS or DS
            visited = [False]*n
            for i in range(n):
                if not visited[i]:
                    # BFS to gather connected walls
                    queue = [i]
                    visited[i] = True
                    combined_points_list = []
                    plane_model = subwalls[i][1]
                    while queue:
                        u = queue.pop(0)
                        combined_points_list.append(subwalls[u][0])  # The points
                        for v in adjacency[u]:
                            if not visited[v]:
                                visited[v] = True
                                queue.append(v)
                    # Merge all points in combined_points_list
                    all_points = np.vstack(combined_points_list)
                    # Use rep plane from the first in BFS
                    # We already have group normal in group, but let's just reuse plane_model for simplicity
                    normal = group['normal']
                    h, w = self.calculate_wall_dimensions(all_points, normal)
                    merged_walls.append((all_points, plane_model, [w, h]))
        self.get_logger().info(f"Merged walls into {len(merged_walls)} final wall(s).")
        return merged_walls
    
    def group_walls_by_plane(self):
        """
        Group walls with similar plane normal and plane offset.
        We'll then do XY bounding box checks within each plane group.
        """
        groups = []
        normal_thresh = 0.95
        d_thresh = 0.1
        for wall in self.walls:
            pts, plane_model, dims = wall
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            d = plane_model[3]
            added = False
            for group in groups:
                if abs(np.dot(normal, group['normal'])) > normal_thresh and abs(d - group['d']) < d_thresh:
                    group['walls'].append(wall)
                    # Update plane offset average
                    old_count = len(group['walls'])
                    group['d'] = (group['d']*(old_count-1) + d) / old_count
                    added = True
                    break
            if not added:
                groups.append({
                    'normal': normal,
                    'd': d,
                    'walls': [wall]
                })
        self.get_logger().info(f"Grouped walls into {len(groups)} plane group(s).")
        return groups
    
    def compute_xy_min_max(self, points):
        """Compute the min_xy, max_xy bounding box for these points in XY."""
        xy = points[:, :2]
        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        return min_xy, max_xy
    
    def xy_bboxes_overlap(self, min_xy1, max_xy1, min_xy2, max_xy2, threshold=0.2):
        """
        Check if two 2D bounding boxes overlap or are within 'threshold' distance on XY plane.
        We'll say they "overlap" if (box1_min.x - box2_max.x) < threshold, etc.
        """
        # If one box is entirely to the left of the other, or entirely above/below, no overlap
        # We'll expand each box by threshold in all directions to allow "close" merges
        # Expand bounding box by threshold
        expanded_min1 = min_xy1 - threshold
        expanded_max1 = max_xy1 + threshold
        expanded_min2 = min_xy2 - threshold
        expanded_max2 = max_xy2 + threshold
        
        # Overlap check:
        # Overlap if expanded_min1.x <= expanded_max2.x and expanded_max1.x >= expanded_min2.x
        # Similarly for y
        if (expanded_min1[0] <= expanded_max2[0] and expanded_max1[0] >= expanded_min2[0] and
            expanded_min1[1] <= expanded_max2[1] and expanded_max1[1] >= expanded_min2[1]):
            return True
        else:
            return False
    
    def calculate_wall_dimensions(self, points, plane_normal):
        """Compute wall height (Z extent) and width in XY along an axis perpendicular to plane_normal (in XY)."""
        if len(points) < 3:
            return 0, 0
        pts = np.asarray(points)
        z_vals = np.sort(pts[:, 2])
        lower_idx = max(0, int(0.02 * len(z_vals)))
        upper_idx = min(len(z_vals) - 1, int(0.98 * len(z_vals)))
        height = z_vals[upper_idx] - z_vals[lower_idx]
        
        xy_norm = plane_normal[:2]
        if np.linalg.norm(xy_norm) < 1e-6:
            x_axis = np.array([1, 0])
        else:
            xy_norm = xy_norm / np.linalg.norm(xy_norm)
            x_axis = np.array([-xy_norm[1], xy_norm[0]])
        xy_points = pts[:, :2]
        center_xy = np.mean(xy_points, axis=0)
        proj = np.dot(xy_points - center_xy, x_axis)
        proj = np.sort(proj)
        lower_idx = max(0, int(0.02 * len(proj)))
        upper_idx = min(len(proj) - 1, int(0.98 * len(proj)))
        width = proj[upper_idx] - proj[lower_idx]
        return height, width
    
    def cluster_wall_segments(self, points):
        """Cluster points on the same plane using XY coordinates (DBSCAN)."""
        if len(points) < self.cluster_min_points:
            return []
        try:
            pts_xy = points[:, :2]
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_points).fit(pts_xy)
            labels = clustering.labels_
            clusters = []
            for label in set(labels):
                if label == -1:
                    continue
                seg_pts = points[labels == label]
                if len(seg_pts) < self.cluster_min_points:
                    continue
                clusters.append(seg_pts)
            self.get_logger().info(f"Clustered into {len(clusters)} wall segments")
            return clusters
        except Exception as e:
            self.get_logger().error(f"Error in clustering: {e}")
            return []
    
    def create_wall_markers(self):
        """Create markers for the full point cloud and the merged walls."""
        self.wall_markers = MarkerArray()
        
        # --- Full Point Cloud Marker ---
        if self.full_pcd is not None:
            full_marker = Marker()
            full_marker.header.frame_id = self.frame_id
            full_marker.header.stamp = self.get_clock().now().to_msg()
            full_marker.ns = "full_point_cloud"
            full_marker.id = 9999
            full_marker.type = Marker.POINTS
            full_marker.action = Marker.ADD
            full_marker.scale.x = 0.02
            full_marker.scale.y = 0.02
            full_marker.color = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.5)
            vis_pcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
            self.get_logger().info(f"Visualizing point cloud with {len(vis_pcd.points)} points")
            for pt in np.asarray(vis_pcd.points):
                p = Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
                full_marker.points.append(p)
            full_marker.lifetime.sec = 0
            self.wall_markers.markers.append(full_marker)
            self.get_logger().info(f"Created full point cloud marker with {len(full_marker.points)} points")
        else:
            self.get_logger().warn("No point cloud data available for visualization")
        
        # --- Merged walls ---
        for i, (pts, plane_model, dims) in enumerate(self.walls):
            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            center = np.mean(pts, axis=0)
            w, h = dims
            # Derive x_axis for orientation
            xy_norm = normal[:2]
            if np.linalg.norm(xy_norm) < 1e-6:
                x_axis_3d = np.array([1, 0, 0])
            else:
                xy_norm = xy_norm / np.linalg.norm(xy_norm)
                x_axis_3d = np.array([-xy_norm[1], xy_norm[0], 0])
            
            color = self.colors[i % len(self.colors)]
            
            # CUBE marker
            cube_marker = Marker()
            cube_marker.header.frame_id = self.frame_id
            cube_marker.header.stamp = self.get_clock().now().to_msg()
            cube_marker.ns = "walls"
            cube_marker.id = i
            cube_marker.type = Marker.CUBE
            cube_marker.action = Marker.ADD
            cube_marker.pose.position.x = center[0]
            cube_marker.pose.position.y = center[1]
            cube_marker.pose.position.z = center[2]
            cube_marker.scale.x = w
            cube_marker.scale.y = self.wall_thickness
            cube_marker.scale.z = h
            # Compute orientation
            z_axis = np.array([0, 0, 1])
            y_axis = np.cross(z_axis, x_axis_3d)
            if np.linalg.norm(y_axis) < 1e-6:
                y_axis = np.cross(normal, x_axis_3d)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis_3d = np.cross(y_axis, z_axis)
            x_axis_3d = x_axis_3d / np.linalg.norm(x_axis_3d)
            rot_mat = np.column_stack((x_axis_3d, y_axis, z_axis))
            R = np.eye(4)
            R[:3, :3] = rot_mat
            quat = tf.quaternion_from_matrix(R)
            cube_marker.pose.orientation.x = quat[0]
            cube_marker.pose.orientation.y = quat[1]
            cube_marker.pose.orientation.z = quat[2]
            cube_marker.pose.orientation.w = quat[3]
            cube_marker.color.r = float(color[0])
            cube_marker.color.g = float(color[1])
            cube_marker.color.b = float(color[2])
            cube_marker.color.a = float(color[3])
            cube_marker.lifetime.sec = 0
            self.wall_markers.markers.append(cube_marker)
            
            # TEXT marker
            text_marker = Marker()
            text_marker.header.frame_id = self.frame_id
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "wall_text"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = center[0]
            text_marker.pose.position.y = center[1]
            text_marker.pose.position.z = center[2] + h/2 + 0.2
            text_marker.text = f"Wall {i+1}: {w:.2f}m x {h:.2f}m"
            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.lifetime.sec = 0
            self.wall_markers.markers.append(text_marker)
            
            # ARROW marker
            arrow_marker = Marker()
            arrow_marker.header.frame_id = self.frame_id
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = "wall_normals"
            arrow_marker.id = i
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.points = []
            arrow_start = Point(x=center[0], y=center[1], z=center[2])
            arrow_marker.points.append(arrow_start)
            arrow_length = min(w/2.0, 0.5)
            arrow_end = Point(
                x=center[0] + x_axis_3d[0] * arrow_length,
                y=center[1] + x_axis_3d[1] * arrow_length,
                z=center[2] + x_axis_3d[2] * arrow_length
            )
            arrow_marker.points.append(arrow_end)
            arrow_marker.scale.x = 0.02
            arrow_marker.scale.y = 0.05
            arrow_marker.scale.z = 0.05
            arrow_marker.color.r = float(color[0])
            arrow_marker.color.g = float(color[1])
            arrow_marker.color.b = float(color[2])
            arrow_marker.color.a = 1.0
            arrow_marker.lifetime.sec = 0
            self.wall_markers.markers.append(arrow_marker)
        
        return self.wall_markers
    
    def publish_markers(self):
        """Publish markers to RViz."""
        if not self.walls:
            self.get_logger().info("No walls to visualize.")
        markers = self.create_wall_markers()
        self.marker_pub.publish(markers)
        self.get_logger().info(f"Published {len(markers.markers)} markers for visualization")
    
    def process_point_cloud(self, pcd_file):
        """Complete pipeline: load, preprocess, segment walls, merge, visualize."""
        pcd = self.load_point_cloud(pcd_file)
        if pcd is None:
            return False
        pcd = self.preprocess_point_cloud(pcd)
        self.full_pcd = pcd
        self.segment_walls(pcd)
        if self.walls:
            self.publish_markers()
            return True
        else:
            self.get_logger().info("No walls detected.")
            self.publish_markers()  # Show the point cloud anyway
            return False


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="Wall Detection with RViz Visualization (ROS2 version) + Merging in XY.")
    parser.add_argument("input_pcd", help="Input point cloud file (.pcd)")
    parser.add_argument("--frame_id", default="odom", help="TF frame ID for visualization (default: odom)")
    parser.add_argument("--publish_rate", type=float, default=1.0, help="Rate to publish markers (Hz)")
    parsed_args = parser.parse_args()
    
    detector = WallDetectorRViz()
    detector.frame_id = parsed_args.frame_id
    
    success = detector.process_point_cloud(parsed_args.input_pcd)
    rate_hz = parsed_args.publish_rate
    
    try:
        while rclpy.ok():
            detector.publish_markers()
            rclpy.spin_once(detector, timeout_sec=1.0 / rate_hz)
    except KeyboardInterrupt:
        detector.get_logger().info("Shutting down wall detector...")
    
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()