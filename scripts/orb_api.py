#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion

class Orb():
    def get_latest_camera_data():
        """
        This function should return the latest camera data
        """
        pass

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

orb = Orb()
target = Target()

if __name__=="__main__":
    pass