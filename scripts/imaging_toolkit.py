from api import Orb, Target
import rospy
from time import sleep

orb = Orb()
target = Target()
rospy.init_node("bot_api")
image_cv2_rgb = orb.get_latest_camera_data()
print(image_cv2_rgb)
rospy.spin()


