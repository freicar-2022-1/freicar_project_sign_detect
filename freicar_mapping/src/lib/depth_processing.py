#!/usr/bin/env python3
from sensor_msgs.msg import Image
from typing import List
import rospy

# TODO: declare the dependencies in the CMakeList file
from jsk_recognition_msgs.msg import BoundingBox
from pyrealsense2 import intrinsics, rs2_deproject_pixel_to_point

from cv_bridge import CvBridge
import numpy as np


def compute_median_distance(depth_image: Image, bbox: BoundingBox) -> float:
    """
    Computes the median distance of the bounding box in the depth image.
    -----------
    Parameters:
        depth_image: depth image message
        bbox (jsk_recognition_msgs.msg.BoundingBox): Bounding box message describing the
                position of the sign (street sign or Aruco marker) in the image plane. Assumes
                the bbox to be axis-aligned and rectangular, and thus only uses x,y of the
                message's pose and dimensions and assumes x as the horizontal and y as the
                vertical axis of the image, with (0,0) being the upper left coordinate.
    --------
    Returns:
        distance (float): The median distance of the pixels in the bounding box in meters.
    """
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')

    width = int(bbox.dimensions.x)
    height = int(bbox.dimensions.y)

    x = int(bbox.pose.position.x)
    y = int(bbox.pose.position.y)

    # addressing is flipped here because it is a matrix
    roi = image[y:y+height, x:x+width]

    # TODO: depth values appear to be mm, but are they really? In the docs of librealsense2
    # they mention something about retrieving some scale:
    # https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0#depth-image-formats
    distance_mm = np.median(roi)
    distance = distance_mm / 1000

    return distance


def compute_distance_scan(depth_image: Image, bbox: BoundingBox) -> List[float]:
    """
    Computes the median distance of each pixel column inside the bounding box.
    -----------
    Parameters:
        depth_image: depth image message
        bbox (jsk_recognition_msgs.msg.BoundingBox): Bounding box message describing the
                position of the sign (street sign or Aruco marker) in the image plane. Assumes
                the bbox to be axis-aligned and rectangular, and thus only uses x,y of the
                message's pose and dimensions and assumes x as the horizontal and y as the
                vertical axis of the image, with (0,0) being the upper left coordinate.
    --------
    Returns:
        distance_scan (List[float]): The median distances of each pixel column in the bounding
        box in meters. Length will be equal to the width of the bbox
    """
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')

    width = int(bbox.dimensions.x)
    height = int(bbox.dimensions.y)

    x = int(bbox.pose.position.x)
    y = int(bbox.pose.position.y)

    # addressing is flipped here because it is a matrix
    roi = depth_image[y:y+height, x:x+width]

    distance_mm = np.median(roi, axis=0)
    distance = distance_mm / 1000

    return distance


def compute_sign_orientation(
        depth_image: Image, bbox: BoundingBox, cam_intrinsics: intrinsics
) -> float:
    """
    Computes the angle around the Z-axis/yaw (in radians, counter-clockwise around the Z-axis
    with the X-axis (east?) being 0).
    See this link for ROS conventions that we (hope to) follow:
    https://www.ros.org/reps/rep-0103.html#axis-orientation
    -----------
    Parameters:
        depth_image: depth image message
        bbox (jsk_recognition_msgs.msg.BoundingBox): Bounding box message describing the
                position of the sign (street sign or Aruco marker) in the image plane. Assumes
                the bbox to be axis-aligned and rectangular, and thus only uses x,y of the
                message's pose and dimensions and assumes x as the horizontal and y as the
                vertical axis of the image, with (0,0) being the upper left coordinate.
        cam_intrinsics (pyrealsense2.intrinsics): Camera intrinsics in librealsense2 format
    --------
    Returns:
        angle (float): The orientation of the sign as the angle described above.
    """
    # get median distances for each pixel column in the bounding box
    distance_scan = compute_distance_scan(depth_image, bbox)

    center_y = bbox.pose.position.y + (bbox.dimensions.y / 2)
    leftmost_x = bbox.pose.position.x
    bbox_width = int(bbox.dimensions.x)

    assert len(distance_scan) == bbox_width

    center_px_row = [[leftmost_x + x_inc, center_y] for x_inc in range(bbox_width)]

    # Get 3d coordinates for each depth on the center row of the bounding box.
    # doing this as list comprehension in the hope that it's faster that a loop, but not sure.
    # If this is too slow we should look into proper parallelization
    deprojected_center_row = [
        rs2_deproject_pixel_to_point(cam_intrinsics, px, dist)
        for px, dist in zip(center_px_row, distance_scan)
    ]

    # get coordinates from realsense convention to ros convention (right-hand rule)
    rs2_x, rs2_y, rs2_z = np.array(deprojected_center_row).T
    x = rs2_z
    y = -rs2_x
    # don't need z, we work with x,y only
    # z = -rs2_y

    # fit a line y' = a x + b through the detected points on the sign
    a, b = np.polyfit(x, y, deg=1)

    # get the tangent vector originating from the left edge of the sign
    x1 = x[0]
    x2 = x[-1]
    y1 = a * x1  # +b cancels out below, so we drop it
    y2 = a * x2
    tangent = np.array([x2 - x1, y2 - y1])
    # now rotate by 90 degree clockwise (because we have the tangent vector originating from
    # the left side of the sign and want a normal vector pointing towards the camera)
    rot90 = np.array(
        [[0,  1],
         [-1, 0]]
    )
    normal = rot90 @ tangent

    # Finally get the angle between our tangent vector and the positive x-axis (in the camera
    # frame, this is the axis pointing straight forward. So the tangent's angles will always be
    # smaller than -pi/2 or larger than pi/2 if the sign is pointing towards us (which it
    # always will)
    theta = np.math.atan2(normal[1], normal[0])

    return theta
