#!/usr/bin/env python3
import rospy
import tf

from jsk_recognition_msgs.msg import BoundingBoxArray
from sensor_msgs.msg import Image, CameraInfo
from lib.utils import get_aruco_bbox




class ArucoNode:
    def __init__(self):
        color_subscriber = rospy.Subscriber("/freicar_3/d435/color/image_raw", Image, self.callback)
        

        # TODO: queue size?
        # see
        # https://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.aruco_publisher = rospy.Publisher('aruco_bbox', BoundingBoxArray, queue_size=50)
        self.debug_publisher = rospy.Publisher('aruco_debug', Image, queue_size=1)

       

    def callback(self, colorimg_msg):
        """
        Callback function to handle incoming BoundingBoxArray messages from the
        Aruco detector or the street sign detector.
        """
        id_mapping = {1: 0, 3: 1, 10: 2}
        bboxes = get_aruco_bbox(colorimg_msg, id_mapping, marker_pub=self.debug_publisher)
        self.aruco_publisher.publish(bboxes)
        
       

if __name__ == "__main__":
    rospy.init_node("aruco_node")
    rospy.loginfo("Starting aruco node...")
    aruco_node = ArucoNode()
    rospy.loginfo("Aruco node started.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Received keyboard interrupt, shutting down...")
    rospy.loginfo("Aruco node finished.")
