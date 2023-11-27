#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist


def callback(new_msg):
    global twist_message
    twist_message = new_msg


def publisher():
    global twist_message
    twist_message = Twist()

    rospy.init_node("mover", anonymous=True)
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    rospy.Subscriber("cmd_vel_proxy", Twist, callback)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pub.publish(twist_message)
        rate.sleep()


if __name__ == "__main__":
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
