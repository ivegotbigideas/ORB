from api import Orb, Target, debug
import rospy
import tf
import cv2
import numpy as np
from visualization_msgs.msg import Marker

orb = Orb()
target = Target()

rospy.init_node("target_location_detector")
marker_pub = rospy.Publisher("target_marker", Marker, queue_size=10)
listener = tf.TransformListener()

# camera_data
FL_X, FL_Y, PX, PY = orb.get_camera_info()

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
        cv2.rectangle(image_cv2_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
    else:
        return None, None, None, None
    
    if debug:
        cv2.imshow("Camera Feed with Target", image_cv2_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return x, y, width, height

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

def determine_marker_loc(center_x, center_y, depth):
    X = depth
    Y = (center_x - PX) * depth / FL_X
    Z = (center_y - PY) * depth / FL_Y
    
    X = -X
    
    listener.waitForTransform('map', 'camera_link', rospy.Time(), rospy.Duration(4.0))
    (trans, rot) = listener.lookupTransform('map', 'camera_link', rospy.Time(0))
    
    X = X + trans[0]
    Y = Y + trans[1]
    Z = Z + trans[2]

    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.pose.position.x = X
    marker.pose.position.y = Y
    marker.pose.position.z = Z
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.2 
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker_pub.publish(marker)


while True:
    try:
        x, y, width, height = detect_target() # check
        if x == None:
            pass
        else:
            center = find_center_of_target(x, y, width, height)
            depth = orb.get_latest_cam_depth_data()[center[1], center[0]]
            determine_marker_loc(center[0], center[1], depth)
    except Exception as e:
        print(e)

