#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class PosePublisherNode:
    def __init__(self):
        rospy.init_node('estimated_pose_publisher')
        self.odom_subscriber = rospy.Subscriber("/odom_fix", Odometry, self.pose_callback)
        self.pose_publisher = rospy.Publisher("/robot_estimated_pose", PoseStamped, queue_size=10)

    def pose_callback(self, msg):
        pose = msg.pose.pose

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "map"  # or "odom"?
        pose_stamped.pose = pose

        self.pose_publisher.publish(pose_stamped)

try:
    node = PosePublisherNode()
    rospy.spin()
except rospy.ROSInterruptException:
    pass