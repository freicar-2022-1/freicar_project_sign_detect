#!/usr/bin/env python3

import rospy
from jsk_recognition_msgs.msg import BoundingBox

from lib.deprojection import get_relative_pose_from_bbox
from lib.utils import get_static_caminfo


def test_deprojection():
    cam_info_msg = get_static_caminfo()

    bbox = BoundingBox()
    bbox.header.stamp = rospy.Time(420)
    # this is a centered 10x20 bounding box. The deprojected position should be directly in
    # front of the camera.
    # TODO: more test cases, especially off-center. Best compare with real measurements.
    bbox.pose.position.x = 1280 // 2 - 5
    bbox.pose.position.y = 720 // 2 - 10
    bbox.dimensions.x = 10
    bbox.dimensions.y = 20
    depth = 1.5

    sign_pose = get_relative_pose_from_bbox(bbox, cam_info_msg, depth)

    print("----------------------------")
    print("Deprojection test results:")
    print("----------------------------")
    print("Camera info:\n\n", cam_info_msg)
    print("----------------------------")
    print("Bounding box:\n\n", bbox)
    print("----------------------------")
    print("Deprojected sign pose:\n\n", sign_pose)
    print("----------------------------")

    assert int(sign_pose.header.stamp.to_sec()) == 420
    assert sign_pose.header.frame_id == "freicar_3/d435_color_optical_frame"
    assert abs(1.5 - sign_pose.pose.position.x) < 1e-5
    assert abs(-0.01936326176 - sign_pose.pose.position.y) < 1e-5
    assert abs(-0.00483129406 - sign_pose.pose.position.z) < 1e-5


if __name__ == '__main__':
    rospy.init_node('deprojection_test', anonymous=True)
    test_deprojection()
