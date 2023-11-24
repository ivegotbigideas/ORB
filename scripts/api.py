#!/usr/bin/env python3

import rospy
import tf
import random
import numpy as np
import time
import os
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Quaternion
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates, ModelState

# config
orb_name = "simple_bot"
target_name = "pink_box"
debug = False
if os.path.exists(".debug"):
    debug = True


class Orb:
    def __init__(self):
        # state
        self.robot_state = "searching"  # searching, converging or terminated

        # subscribers
        if debug:  # debug
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
        self._robot_twist_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._robot_model_state_publisher = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=10
        )

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
            # print("TEST waiting for base_scan")
            msg = rospy.wait_for_message("/base_scan", LaserScan)
        try:  # fixes an error that occurs when this is called from elsewhere idk
            msg = msg[0]
        except:
            pass

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
        try:  # prevents an issue when this function is called from randomise_robot_pose. Idk tbh.
            msg = msg[0]
        except:
            pass

        return get_true_pose(msg, orb_name)

    def move_robot(self, dir):
        """
        This function should take a direction (f, b, cw, acw) as an input
        and then move the robot in that direction
        """
        msg = Twist()

        if dir == "f":
            msg.linear.x = 1.0
        elif dir == "b":
            msg.linear.x = -1.0
        elif dir == "cw":
            msg.angular.z = -1.0
        elif dir == "acw":
            msg.angular.z = 1.0
        elif dir == "stop":
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self._robot_twist_publisher.publish(msg)

    def randomise_robot_pose(self):
        """
        This function should put the robot in a random valid pose. This will be utilised for training the MDP for exploration.
        """
        randomise_pose(orb_name)

    def get_latest_slam_map(self, *callback_message):
        """
        This function should return the latest occupancy grid
        """
        if callback_message:
            msg = callback_message
        else:
            msg = rospy.wait_for_message("/map", OccupancyGrid)
        try:
            msg = msg[0]
        except:
            pass

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

    def set_robot_state(self, state):
        """
        This function should put the robot in the provided state
        """
        self.robot_state = state


class Target:
    def get_ground_truth_target_pose(self, *callback_message):
        """
        Returns the true location of the target
        """
        if callback_message:
            msg = callback_message
        else:
            msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
        try:  # prevents an issue when this function is called from randomise_target_pose. Idk tbh.
            msg = msg[0]
        except:
            pass
        return get_true_pose(msg, target_name)

    def randomise_target_pose(self):
        """
        Randomise target pose
        """
        randomise_pose(target_name)


def get_true_pose(model_states_msg, obj_name):
    """
    Gets the true pose of a specified object name
    """
    index = model_states_msg.name.index(obj_name)
    pose = model_states_msg.pose[index]
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


def randomise_pose(obj_name):
    msg = ModelState()
    msg.model_name = obj_name
    msg.pose.position.x = random.uniform(-4.2, 4.2)  # tweak these limits
    msg.pose.position.y = random.uniform(-9, 9)  # tweak these limits
    msg.pose.position.z = 0.1

    new_yaw = random.uniform(-np.pi, np.pi)
    new_quat = tf.transformations.quaternion_from_euler(0, 0, new_yaw)
    msg.pose.orientation.x = new_quat[0]
    msg.pose.orientation.y = new_quat[1]
    msg.pose.orientation.z = new_quat[2]
    msg.pose.orientation.w = new_quat[3]

    rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)(msg)


if __name__ == "__main__":
    orb = Orb()
    target = Target()
    rospy.init_node("bot_api")
    rospy.spin()
