from api import Orb, Target
import rospy
import cv2

orb = Orb()
target = Target()
rospy.init_node("target_location_detector")

def detect_target():
    image_cv2_bgr = cv2.cvtColor(orb.get_latest_camera_data(), cv2.COLOR_RGB2BGR)
    image_cv2_hsv = cv2.cvtColor(image_cv2_bgr, cv2.COLOR_BGR2HSV)
    lower_pink = (140, 100, 100)
    upper_pink = (170, 255, 255)
    mask = cv2.inRange(image_cv2_hsv, lower_pink, upper_pink)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(max_contour)
        return x, y, width, height
    else:
        return None, None, None, None

def find_center_of_target(x, y, w, h):
    middle_x = x + w // 2
    middle_y = y + h // 2
    return middle_x, middle_y

def move_robot(horizontal_coord):
    if horizontal_coord == None:
        orb.move_robot("cw")
    elif horizontal_coord in range(318, 323):
        orb.move_robot("f")
    elif horizontal_coord < 318:
        orb.move_robot("acw")
    elif horizontal_coord > 323:
        orb.move_robot("cw")

while True:
    try:
        x, y, width, height = detect_target()
        if x == None:
            move_robot(None)
        else:
            center = find_center_of_target(x, y, width, height)
            marker_dist = orb.get_latest_cam_depth_data()[center[0], center[1]]
    except Exception as e:
        print(e)

