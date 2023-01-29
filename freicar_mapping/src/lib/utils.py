#!/usr/bin/env python

from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, RegionOfInterest, Image
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
import rospy

import cv2
from cv_bridge import CvBridge


def get_static_caminfo() -> CameraInfo:
    """
    Helper function to get a dummy CameraInfo message, timestamped at the
    current time with the intrinsics from the Intel Realsense d435 camera, as
    recorded from /freicar_3/d435/color/camera_info .

    This is just intended for testing, later we should use the live
    CameraInfo messages (the important parts seem to be static, but i think
    it's nicer if we subscribe to it).
    """
    cam_info = CameraInfo()
    cam_info.height = 720
    cam_info.width = 1280
    cam_info.distortion_model = "plumb_bob"

    cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]

    cam_info.K = [920.4984130859375, 0.0, 628.117431640625, 0.0,
                  919.5174560546875, 357.0383605957031, 0.0, 0.0, 1.0]

    cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    cam_info.P = [920.4984130859375, 0.0, 628.117431640625, 0.0, 0.0,
                  919.5174560546875, 357.0383605957031, 0.0, 0.0, 0.0, 1.0, 0.0]

    cam_info.binning_x = 0
    cam_info.binning_y = 0

    roi = RegionOfInterest()
    roi.x_offset = 0
    roi.y_offset = 0
    roi.height = 0
    roi.width = 0

    cam_info.roi = roi

    cam_info.header.stamp = rospy.Time.now()
    cam_info.header.frame_id = "freicar_3/d435_color_optical_frame"
    cam_info.header.seq = 0

    return cam_info


def get_aruco_bbox(image_msg: Image, id_mapping: dict, debug=False) -> BoundingBoxArray:
    """
    Function to get BoundingBoxArray messages of detected Aruco markers in the image for
    testing purposes. id_mapping should map the cv2.aruco.DICT_5X5_50 IDs to the values used in
    the 'label' field of the BoundingBox elements.
    If debug==True, print detected bounding boxes and show them using cv2.imshow.
    """
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(image_msg.message, desired_encoding='bgr8')
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    corners, ids, rejectedImagePts = cv2.aruco.detectMarkers(image, aruco_dict)
    if debug:
        print(f'corners: {corners}\nids: {ids}\n rejectedImagePts: {rejectedImagePts}')
        vis = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow('vis', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    boxes = []
    for i, corner in enumerate(corners):
        id_ = ids[i][0]
        if id_ in id_mapping.keys():
            xs = [p[0] for p in corner[0]]
            ys = [p[1] for p in corner[0]]

            bbox = BoundingBox()
            bbox.header.stamp = rospy.Time(420)
            bbox.pose.position.x = min(xs)
            bbox.pose.position.y = min(ys)
            bbox.dimensions.x = max(xs) - min(xs)
            bbox.dimensions.y = max(ys) - min(ys)
            bbox.label = id_mapping[id_]

            boxes.append(bbox)

    array_header = Header()
    array_header.frame_id = image_msg.message.header.frame_id
    array_header.stamp = image_msg.message.header.stamp

    return BoundingBoxArray(array_header, boxes)


def bbox2str(bbox: BoundingBox):
    s = f"Bounding box with label {bbox.label} at pixel position: "
    s += f"x: {bbox.pose.position.x}+{bbox.dimensions.x} "
    s += f"y: {bbox.pose.position.y}+{bbox.dimensions.y}"
    return s


def pose2str(pose: PoseStamped):
    theta = euler_from_quaternion(pose.pose.orientation)[2]
    theta_deg = theta/3.14159*180
    s = f"Pose with position ({pose.pose.position.x:.3f}, "
    s += f"{pose.pose.position.y:.3f}, {pose.pose.position.z:.3f}) "
    s += f"and rotation around z: {theta:.3f} rad / {theta_deg:.1f} deg"
    return s
