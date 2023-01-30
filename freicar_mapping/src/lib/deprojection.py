#!/usr/bin/env python3
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from lib.depth_processing import compute_median_distance, compute_sign_orientation

from tf.transformations import quaternion_from_euler

# TODO: declare the dependencies in the CMakeList file
from jsk_recognition_msgs.msg import BoundingBox
from pyrealsense2 import intrinsics, distortion, rs2_deproject_pixel_to_point


def get_relative_pose_from_bbox(
        bbox: BoundingBox, cam_info: CameraInfo, depth_image: Image
) -> PoseStamped:
    """
    Computes the sign pose relative to the camera frame.
    -----------
    Parameters:
        bbox (jsk_recognition_msgs.msg.BoundingBox): Bounding box message describing the
                position of the sign (street sign or Aruco marker) in the image plane. Assumes
                the bbox to be axis-aligned and rectangular, and thus only uses x,y of the
                message's pose and dimensions and assumes x as the horizontal and y as the
                vertical axis of the image, with (0,0) being the upper left coordinate.
        cam_info (sensor_msgs.msg.CameraInfo): Camera intrinsics message
                published in the .../camera_info topic
        depth_image (sensor_msgs.msg.Image): Full depth image at the same timestamp as bbox.
    --------
    Returns:
        sign_pose (PoseStamped): The street sign's pose in the camera frame, timestamped at the
                same time as the Bounding Box message. Units and coordinate system should be
                compatible with ROS.
    """
    # this just uses the center point of the bounding box for deprojection,
    # if we want to use non-rectangular boxes we'll need something fancier.
    center_pixel = [
        bbox.pose.position.x + (bbox.dimensions.x // 2),
        bbox.pose.position.y + (bbox.dimensions.y // 2),
    ]

    # Transform camera information into pyrealsense2 format
    # See these links for explanation on what these are:
    # https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
    # https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0#intrinsic-camera-parameters
    # https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    pr2_intrinsics = intrinsics()
    pr2_intrinsics.width = cam_info.width
    pr2_intrinsics.height = cam_info.height
    pr2_intrinsics.ppx = cam_info.K[2]
    pr2_intrinsics.ppy = cam_info.K[5]
    pr2_intrinsics.fx = cam_info.K[0]
    pr2_intrinsics.fy = cam_info.K[4]
    pr2_intrinsics.coeffs = cam_info.D

    if cam_info.distortion_model == "plumb_bob":
        # the d435 camera messages state the model as plumb_bob, and this:
        # https://calib.io/blogs/knowledge-base/camera-models
        # says that Plumb Bob is equivalent to Brown-Conrady.
        pr2_intrinsics.model = distortion.brown_conrady
    else:
        raise NotImplementedError(
            f"CameraInfo message specified unknown distortion model: {cam_info.distortion_model}"
        )

    depth = compute_median_distance(depth_image, bbox)

    # call to librealsense2 to do the actual deprojection
    pose_x, pose_y, pose_z = rs2_deproject_pixel_to_point(
        pr2_intrinsics, center_pixel, depth
    )

    yaw = compute_sign_orientation(depth_image, bbox, pr2_intrinsics)

    # use a stamped pose message as an easy way to pass on the timestamp and for possibly
    # publishing as a debug message.
    sign_pose = PoseStamped()
    sign_pose.header.stamp = bbox.header.stamp
    sign_pose.header.frame_id = cam_info.header.frame_id

    # convert coordinate system to the default ROS way (right-handed). See:
    # https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0#point-coordinates
    # and
    # https://www.ros.org/reps/rep-0103.html#axis-orientation
    sign_pose.pose.position.x = pose_z
    sign_pose.pose.position.y = -pose_x
    sign_pose.pose.position.z = -pose_y

    qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
    sign_pose.pose.orientation.x = qx
    sign_pose.pose.orientation.y = qy
    sign_pose.pose.orientation.z = qz
    sign_pose.pose.orientation.w = qw

    return sign_pose
