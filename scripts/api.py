#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.msg import ModelStates


class Orb:
    def __init__(self):
        # subscribers
        self._camera_subscriber = rospy.Subscriber("/camera/image_raw", Image)
        self._lidar_subscriber = rospy.Subscriber("/base_scan", LaserScan)
        self._robot_ground_truth_subscriber = rospy.Subscriber(
            "/gazebo/model_states", ModelStates
        )
        self._slam_map_subscriber = rospy.Subscriber("/map", ModelStates)
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

    def get_latest_lidar_data(self, *callback_message):
        """
        This function should return the latest lidar scanner data
        """
        if callback_message:
            msg = callback_message
        else:
            msg = rospy.wait_for_message("/base_scan", LaserScan)
        msg = msg[0]

        laser_data = {
            "header": {
                "seq": msg.header.seq,
                "stamp": {
                    "secs": msg.header.stamp.secs,
                    "nsecs": msg.header.stamp.nsecs,
                },
                "frame_id": msg.header.frame_id,
            },
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
            "angle_increment": msg.angle_increment,
            "time_increment": msg.time_increment,
            "scan_time": msg.scan_time,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "ranges": list(msg.ranges),
            "intensities": list(msg.intensities),
        }

        return laser_data

    def get_ground_truth_robot_pose(self, *callback_message):
        """
        This function should tell us where the robot actually is
        """
        if callback_message:
            msg = callback_message
        else:
            msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
        msg = msg[0]

        orb_name = "simple_bot"
        index = msg.name.index(orb_name)
        pose = msg.pose[index]
        pose = {
            "position": {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
            },
            "orientation": {
                "x": pose.orientation.x,
                "y": pose.orientation.y,
                "z": pose.orientation.z,
                "w": pose.orientation.w,
            },
        }
        return pose

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


class Target:
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

if __name__ == "__main__":
    rospy.spin()

rospy.spin()
