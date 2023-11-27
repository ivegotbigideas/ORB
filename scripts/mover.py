#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist


def callback(new_msg):
    global twist_message
    twist_message = new_msg

def __init__(self):
    publisher()

def publisher():
    global twist_message
    twist_message = Twist()

    rospy.init_node("mover")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    rospy.Subscriber("cmd_vel_proxy", Twist, callback)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        try:
            pub.publish(twist_message)
            rate.sleep()
        except rospy.ROSInterruptException as e:
            pass


if __name__ == "__main__":
    publisher()
