from api import Orb, Target
import rospy
import cv2

orb = Orb()
target = Target()
rospy.init_node("image_toolkit")

image_cv2_bgr = cv2.cvtColor(orb.get_latest_camera_data(), cv2.COLOR_RGB2BGR)
cv2.imwrite("unmasked.png", image_cv2_bgr)

image_cv2_hsv = cv2.cvtColor(image_cv2_bgr, cv2.COLOR_BGR2HSV)
lower_pink = (140, 100, 100)
upper_pink = (170, 255, 255)
mask = cv2.inRange(image_cv2_hsv, lower_pink, upper_pink)
image_cv2_masked = cv2.bitwise_and(image_cv2_hsv, image_cv2_hsv, mask=mask)
cv2.imwrite("masked.png", image_cv2_masked)


