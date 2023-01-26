#!/usr/bin/env python3
import rospy
from jsk_recognition_msgs.msg import BoundingBoxArray

from lib import deprojection
from tf import TransformListener


class MappingNode:
    def __init__(self):
        self.bbox_subscriber = rospy.Subscriber(
            "bbox", BoundingBoxArray, self.bbox_callback
        )
        self.tl = TransformListener()

    def bbox_callback(self, bounding_boxes: BoundingBoxArray):
        """
        Callback function to handle incoming BoundingBoxArray messages from the
        Aruco detector or the street sign detector.
        """
        for bbox in bounding_boxes:
            sign_pose_cam = deprojection.get_relative_pose_from_bbox(bbox)
            # TODO: if we want to use some kind of kalman filter, what is our measurement
            # space? In case of range-bearing, we would need to calculate that instead of the
            # sign pose relative to the camera.
            # TODO: "world" frame doesn't exist in our recording, figure out how to correctly
            # transform something from the camera frame to global with our setup. Definitely
            # need to include the vive data in the rosbag. Also, how do we get the transform
            # from the vive sensor to the camera? We would need the calibration from the other
            # team i think.
            sign_pose_world = self.tl.transformPose("world", sign_pose_cam)
            # TODO: check that we actually pass the recognized sign type as label
            sign_type = bbox.label
            # TODO: pass the measurement on to the mapping function.


if __name__ == "__main__":
    rospy.init_node("mapping_node")
    rospy.loginfo("Starting mapping node...")
    mapping_node = MappingNode()
    rospy.loginfo("Mapping node started.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Received keyboard interrupt, shutting down...")
    rospy.loginfo("Mapping node finished.")
