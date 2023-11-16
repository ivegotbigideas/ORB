#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from cv_bridge import CvBridge, CvBridgeError

class Orb():
    def __init__(self):
        # subscribers
        self._camera_subscriber = rospy.Subscriber("/camera/image_raw", Image)
        self._lidar_subscriber = None
        self._robot_ground_truth_subscriber = None
        self._slam_map_subscriber = None
        self._slam_estimated_pose_subscriber = None

        # publishers
        self._robot_twist_publisher = None
        self._robot_pose_publisher = None

    def get_latest_camera_data(self, *callback_message):
        """
        This function should return the latest camera data
        """
        if callback_message:
            msg = callback_message
        else:
            msg = rospy.wait_for_message("/camera/image_raw", Image)
        
        bridge = CvBridge()
        try:
            cv2 = bridge.imgmsg_to_cv2(msg, "rgb8")
            return cv2
        except CvBridgeError as e:
            print(e)

    def get_latest_lidar_data():
        """
        This function should return the latest lidar scanner data
        """
        pass

    def get_ground_truth_robot_pose():
        """
        This function should tell us where the robot actually is
        """
        pass

    def move_robot():
        """
        This function should take a direction (f, b, cw, acw) as an input
        and then move the robot in that direction
        """
        pass

    def randomise_robot_pose():
        """
        This function should put the robot in a random pose
        """
        pass

    def get_latest_slam_map():
        """
        This function should return the latest occupancy grid
        """
        pass

    def get_slam_location():
        """
        This function should return the estimated pose of the robot
        """
        pass

    def terminate_robot():
        """
        This function should put the robot in the terminal state
        """
        pass

class Target():
    def get_ground_truth_target_pose():
        """
        Returns the true location of the target
        """
        pass

    def randomise_target_pose():
        """
        Randomise
        """
        pass

rospy.init_node("bot_api")
orb = Orb()
target = Target()

if __name__=="__main__":
    rospy.spin()

rospy.spin()