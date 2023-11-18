#!/usr/bin/env python3

import rospy
import tf
import random
import numpy as np
import time
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates, ModelState

# config
orb_name = "simple_bot"


class Orb:
    def __init__(self):
        # subscribers
        if False:  # debug
            self._camera_subscriber = rospy.Subscriber(
                "/camera/image_raw", Image, self.get_latest_camera_data
            )
            self._lidar_subscriber = rospy.Subscriber(
                "/base_scan", LaserScan, self.get_latest_lidar_data
            )
            self._robot_ground_truth_subscriber = rospy.Subscriber(
                "/gazebo/model_states", ModelStates, self.get_ground_truth_robot_pose
            )
            self._slam_map_subscriber = rospy.Subscriber(
                "/map", OccupancyGrid, self.get_latest_slam_map
            )
            self._slam_odom_fix_subscriber = rospy.Subscriber(
                "/odom_fix", Odometry, self.get_slam_location
            )

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
        try:  # prevents an issue when this function is called from randomise_robot_pose
            msg = msg[0]
        except:
            pass

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

    def randomise_robot_pose(self):
        """
        This function should put the robot in a random valid pose. This will be utilised for training the MDP for exploration.
        """
        rospy.wait_for_service("/gazebo/set_model_state")

        msg = ModelState()
        msg.model_name = orb_name
        msg.pose.position.x = random.uniform(-5, 5) # tweak these limits
        msg.pose.position.y = random.uniform(-5, 5) # teak these limits
        msg.pose.position.z = 1

        new_yaw = random.uniform(-np.pi, np.pi)
        new_quat = tf.transformations.quaternion_from_euler(0, 0, new_yaw)
        msg.pose.orientation.x = new_quat[0]
        msg.pose.orientation.y = new_quat[1]
        msg.pose.orientation.z = new_quat[2]
        msg.pose.orientation.w = new_quat[3]

        rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)(msg)

    def get_latest_slam_map(self, *callback_message):
        """
        This function should return the latest occupancy grid
        """
        if callback_message:
            msg = callback_message
        else:
            msg = rospy.wait_for_message("/map", OccupancyGrid)
        msg = msg[0]

        occ_grid = {
            "header": {
                "seq": msg.header.seq,
                "stamp": {
                    "secs": msg.header.stamp.secs,
                    "nsecs": msg.header.stamp.nsecs,
                },
                "frame_id": msg.header.frame_id,
            },
            "info": {
                "map_load_time": {
                    "secs": msg.info.map_load_time.secs,
                    "nsecs": msg.info.map_load_time.nsecs,
                },
                "resolution": msg.info.resolution,
                "width": msg.info.width,
                "height": msg.info.height,
                "origin": {
                    "position": {
                        "x": msg.info.origin.position.x,
                        "y": msg.info.origin.position.y,
                        "z": msg.info.origin.position.z,
                    },
                    "orientation": {
                        "x": msg.info.origin.orientation.x,
                        "y": msg.info.origin.orientation.y,
                        "z": msg.info.origin.orientation.z,
                        "w": msg.info.origin.orientation.w,
                    },
                },
            },
            "data": msg.data,  # this is the map data itself
        }
        return occ_grid

    def get_slam_location(self, *callback_message):
        """
        This function should return the estimated pose of the robot
        """
        if callback_message:
            msg = callback_message
        else:
            msg = rospy.wait_for_message("/odom_fix", Odometry)
        msg = msg[0].pose.pose

        estimated_pose = {
            "position": {"x": msg.position.x, "y": msg.position.y, "z": msg.position.z},
            "orientation": {
                "x": msg.orientation.x,
                "y": msg.orientation.y,
                "z": msg.orientation.z,
                "w": msg.orientation.w,
            },
        }
        return estimated_pose

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
rospy.spin()
