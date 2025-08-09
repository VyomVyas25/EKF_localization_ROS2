#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from std_msgs.msg import Float32MultiArray  # Using Float32MultiArray for multiple coordinates

class MapReader(Node):
    def __init__(self):
        super().__init__('map_reader')
        # Create publisher for cylinder coordinates (all in one message)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/cylinder_coordinates', 10)
        # Load map and detect cylinders
        self.load_map()
        # Publisher timer (publishes at 1 Hz)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def load_map(self):
        try:
            self.map_image = cv2.imread('/home/vyom/ros2_ws/src/maps/map.pgm', cv2.IMREAD_COLOR)
            with open('/home/vyom/ros2_ws/src/maps/map.yaml', 'r') as file:
                self.map_info = yaml.safe_load(file)
            self.get_logger().info('Map and configuration loaded successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to load map or config: {e}')
            return

        # Convert image to grayscale
        self.map_image = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2GRAY)
        # Apply median blur to reduce noise
        gray_blurred = cv2.medianBlur(self.map_image, 5)
        # Detect circles (cylinders) using Hough Transform
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=11,
            minRadius=5,
            maxRadius=30
        )
        if circles is not None:
            # Keep the float values for higher precision
            self.cylinders = circles[0, :]
            self.get_logger().info(f"{len(self.cylinders)} cylinders detected.")
        else:
            self.cylinders = []
            self.get_logger().warn('No cylinders detected.')

    def timer_callback(self):
        if len(self.cylinders) == 0:
            self.get_logger().warn('No cylinders to process.')
            return

        # Create a flat list of cylinder coordinates: [x1, y1, x2, y2, ...]
        data_list = []
        resolution = self.map_info['resolution']
        origin = self.map_info['origin']  # [origin_x, origin_y, origin_theta]
        for cylinder in self.cylinders:
            x_pixel, y_pixel, _ = cylinder  # use center coordinates
            x_world = origin[0] + (x_pixel * resolution)
            # Note: Adjust conversion for y as needed; here using image height minus y_pixel.
            y_world = origin[1] + ((self.map_image.shape[0] - y_pixel) * resolution)
            data_list.extend([float(x_world), float(y_world)])
            #self.get_logger().info(f"Detected cylinder at ({x_world:.2f}, {y_world:.2f})")
        
        msg = Float32MultiArray()
        msg.data = data_list
        self.publisher_.publish(msg)
        #self.get_logger().info("Published all cylinder coordinates.")

def main(args=None):
    rclpy.init(args=args)
    node = MapReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()