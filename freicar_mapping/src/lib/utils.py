#!/usr/bin/env python

from sensor_msgs.msg import CameraInfo, RegionOfInterest
import rospy


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
