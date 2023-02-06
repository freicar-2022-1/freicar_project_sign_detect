#!/usr/bin/env python3
import rospy
import message_filters
import tf

from jsk_recognition_msgs.msg import BoundingBoxArray
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from lib.utils import get_aruco_bbox, sign_pose_2_marker_msg, bbox2str, check_bbox

from lib import deprojection
from lib.map import Map
import numpy
import sys
import argparse


class MappingNode:
    def __init__(self, use_aruco, use_yolo):
        depth_subscriber = message_filters.Subscriber("/freicar_3/d435/aligned_depth_to_color/image_raw", Image)
        caminfo_subscriber = message_filters.Subscriber("/freicar_3/d435/color/camera_info", CameraInfo)

        if use_yolo:
            yolo_subscriber = message_filters.Subscriber("/freicar_3/trafficsigndetect/prediction/raw", BoundingBoxArray)
            self.sensor_subscriber_yolo = message_filters.TimeSynchronizer(
                [yolo_subscriber, depth_subscriber, caminfo_subscriber],
                queue_size=100
            )
            self.sensor_subscriber_yolo.registerCallback(self.sensor_callback)

        if use_aruco:
            aruco_subscriber = message_filters.Subscriber("/aruco_bbox", BoundingBoxArray)
            self.sensor_subscriber_aruco = message_filters.TimeSynchronizer(
                [aruco_subscriber, depth_subscriber, caminfo_subscriber],
                queue_size=100
            )
            self.sensor_subscriber_aruco.registerCallback(self.sensor_callback)

        # TODO: queue size?
        # see
        # https://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.marker_publisher = rospy.Publisher('sign_markers', Marker, queue_size=50)
        self.bbimg_publisher = rospy.Publisher('aruco_debug', Image, queue_size=50)

        self.tl = tf.TransformListener()
        self.tf_exception_counter = 0

        self.map = Map()

    def sensor_callback(self, bounding_boxes, depthimg_msg, caminfo_msg):
        """
        Function to handle incoming BoundingBoxArray messages from either the
        Aruco detector or the street sign detector.
        """
        rospy.logdebug(f"received Bboxarray with {len(bounding_boxes.boxes)} boxes")
        for bbox in bounding_boxes.boxes:
            if not check_bbox(bbox, caminfo_msg):
                rospy.logwarn(f"Ignoring weird bbox: {bbox2str(bbox)}")
                continue

            try:
                sign_pose_cam = deprojection.get_relative_pose_from_bbox(bbox, caminfo_msg,
                                                                         depthimg_msg, is_cone=(bbox.label == 3))
            except SystemError as e:
                if "LinAlgError" in str(e):
                    rospy.logwarn(f"Encountered linalg error while trying to get orientation"
                                  + f"from sign: {bbox2str(bbox)}")
                    continue
                else:
                    raise
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
    # TODO: if we use a launch file, we maybe have to set the argparse differently
    # https://discourse.ros.org/t/getting-python-argparse-to-work-with-a-launch-file-or-python-node/10606
    parser = argparse.ArgumentParser(
        description="Mapping node argparse"
    )
    parser.add_argument('--aruco', action='store_true')
    parser.add_argument('--yolo', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    rospy.init_node("mapping_node", log_level=rospy.INFO)
    rospy.loginfo("Starting mapping node...")

    if args.yolo and args.aruco:
        rospy.loginfo(f'Using Aruco detector and YOLO object detection model for bounding boxes.')
        mapping_node = MappingNode(use_aruco=args.aruco, use_yolo=args.yolo)
    elif args.yolo:
        rospy.loginfo(f'Using just YOLO object detection model for bounding boxes.')
        mapping_node = MappingNode(use_aruco=args.aruco, use_yolo=args.yolo)
    elif args.aruco:
        rospy.loginfo(f'Using just Aruco detector for bounding boxes.')
        mapping_node = MappingNode(use_aruco=args.aruco, use_yolo=args.yolo)
    else:
        rospy.loginfo(f'Neither --yolo nor --aruco specified, using both.')
        mapping_node = MappingNode(use_aruco=True, use_yolo=True)

    rospy.loginfo("Mapping node started.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Received keyboard interrupt, shutting down...")
    rospy.loginfo("Mapping node finished.")
