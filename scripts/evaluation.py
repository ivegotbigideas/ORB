#!/usr/bin/env python3
'''import rospy
from geometry_msgs.msg import Pose, Point

def pose_difference(pose1, pose2):
    # Extract translation components from Pose messages
    translation1 = pose1.position
    translation2 = pose2.position

    # Calculate the difference in translation and rotation
    translation_difference = Point(
        translation2.x - translation1.x,
        translation2.y - translation1.y,
        translation2.z - translation1.z
    )
    )

    # Create a new Pose message to represent the difference
    pose_difference = Pose(translation=translation_difference, orientation=rotation_difference)

    return pose_difference

if __name__ == "__main__":
    pose1 = get_ground_truth_target_pose
    pose2 = get_robot_pose

    # Calculate the difference between the two poses
    pose_diff = pose_difference(pose1, pose2)

    # Print the result
    print("Pose Difference:")
    print("Translation: ", pose_diff.position)


Notes:
get_ground_truth_target_pose to get the target pose
get true pose to get robots pose.
How to send signal when robot has "found" target? Where is that code
Formula to do evaluation'''
