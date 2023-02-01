#!/usr/bin/env python3
import rospy
import message_filters
import tf

from jsk_recognition_msgs.msg import BoundingBoxArray
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from lib.utils import get_aruco_bbox, sign_pose_2_marker_msg

from lib import deprojection
from lib.map import Map


class MappingNode:
    def __init__(self):
        color_subscriber = message_filters.Subscriber("/freicar_3/d435/color/image_raw", Image)
        depth_subscriber = message_filters.Subscriber("/freicar_3/d435/aligned_depth_to_color/image_raw", Image)
        caminfo_subscriber = message_filters.Subscriber("/freicar_3/d435/color/camera_info", CameraInfo)

        self.sensor_subscriber = message_filters.TimeSynchronizer(
            [color_subscriber, depth_subscriber, caminfo_subscriber],
            queue_size=10,
        )
        self.sensor_subscriber.registerCallback(self.sensor_callback)

        # TODO: queue size?
        # see
        # https://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.marker_publisher = rospy.Publisher('sign_markers', Marker, queue_size=50)
        self.bbimg_publisher = rospy.Publisher('aruco_debug', Image, queue_size=50)

        self.tl = tf.TransformListener()
        self.tf_exception_counter = 0

        self.map = Map()

    def sensor_callback(self, colorimg_msg, depthimg_msg, caminfo_msg):
        """
        Callback function to handle incoming BoundingBoxArray messages from the
        Aruco detector or the street sign detector.
        """
        id_mapping = {1: 0, 3: 1, 10: 2}
        bounding_boxes = get_aruco_bbox(colorimg_msg, id_mapping, marker_pub=self.bbimg_publisher)
        for bbox in bounding_boxes.boxes:
            sign_pose_cam = deprojection.get_relative_pose_from_bbox(bbox, caminfo_msg, depthimg_msg)
            # TODO: transform between freicar_3 and cam?
            sign_pose_cam.header.frame_id = "freicar_3"
            try:
                sign_pose_world = self.tl.transformPose("world", sign_pose_cam)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # if no transform can be found, skip (when testing this only occured at the
                # begining of bag playback
                self.tf_exception_counter += 1
                rospy.loginfo("Encountered exception while looking up transform"
                              + f"({self.tf_exception_counter} encountered so far)")
                continue

            self.map.add_observation(sign_pose_world, sign_type=bbox.label)
            # TODO: don't publish so often?
            self.map.publish_markers(self.marker_publisher, bbox.header.stamp)


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
