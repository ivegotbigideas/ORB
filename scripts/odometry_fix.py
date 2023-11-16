#!/usr/bin/env python3
import rospy
import tf
from nav_msgs.msg import Odometry

def callback(msg):
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]
    yaw += 3.14159
    quaternion = tf.transformations.quaternion_from_euler(euler[0], euler[1], yaw)
    
    msg.pose.pose.orientation.x = quaternion[0]
    msg.pose.pose.orientation.y = quaternion[1]
    msg.pose.pose.orientation.z = quaternion[2]
    msg.pose.pose.orientation.w = quaternion[3]

    publisher.publish(msg)

rospy.init_node('odometry_fix_node.py')
rospy.Subscriber('/odom', Odometry, callback)
publisher = rospy.Publisher('/odom_fix', Odometry, queue_size=10)

rospy.spin()