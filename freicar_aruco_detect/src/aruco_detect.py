#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from tf.transformations import *
import message_filters


class Nodo(object):
    def __init__(self):
        # Params
        self.image = None
        self.depth_image=None
        self.br = CvBridge()
        self.corners =None
        self.ids=None
        self.rejected=None
        
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)
        
        #aruco detector
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        # Publishers
        self.pub = rospy.Publisher("bbox_aruco", BoundingBoxArray, queue_size = 10)

        # Subscribers
        color_sub = message_filters.Subscriber("/freicar_3/d435/color/image_raw",Image)
        depth_sub = message_filters.Subscriber("/freicar_3/d435/aligned_depth_to_color/image_raw",Image)
        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], 10)
        ts.registerCallback(self.callback)

    def callback(self, color_image, depth_image):
        rospy.loginfo('Image received...')
        self.image = self.br.imgmsg_to_cv2(color_image)
        self.depth_image = self.br.imgmsg_to_cv2(depth_image)
        self.corners, self.ids, self.rejected = self.detector.detectMarkers(self.image)


    def start(self):
        rospy.loginfo("publishing corners")
        while not rospy.is_shutdown():
            rospy.loginfo('bbox')           
            if self.corners is not None and len(self.corners) != 0:
            	box_arr = BoundingBoxArray()
            	now = rospy.Time.now()
            	box_arr.header.stamp = now
            	box_arr.header.frame_id = "robot_camera"
            	rospy.loginfo(self.corners)
            	rospy.loginfo(self.ids)
            	for index,id in enumerate(self.ids[0]):
            		corners = self.corners[0][index]
            		box = BoundingBox()
            		box.label = id           		
            		box.header.stamp = now
            		box.header.frame_id = "robot_camera"  
            		x = corners[0][0]
            		y = corners[0][1]           		            		
            		box.pose.position.x = x
            		box.pose.position.y = y
            		width = ((corners[1][0] - corners[0][0]) + (corners[2][0] - corners[3][0])) / 2
            		height = ((corners[3][1] - corners[0][1]) + (corners[2][1] - corners[1][1])) / 2
            		box.dimensions.x = width
            		box.dimensions.y = height           		
            		box.dimensions.z = 0
            		box.value = id
            		depth_1 = self.depth_image[int(y + height), int(x)]
            		depth_2 = self.depth_image[int(y + height), int(x + width)] 
            		rospy.loginfo(depth_1)
            		rospy.loginfo(depth_2)
            		q = quaternion_about_axis((index % 100) * math.pi * 2 / 100.0, [0, 0, 1])
            		box.pose.orientation.x = q[0]
            		box.pose.orientation.y = q[1]
            		box.pose.orientation.z = q[2]
            		box.pose.orientation.w = q[3]
            		box_arr.boxes.append(box)
            	self.pub.publish(box_arr)
            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("aruco_detector", anonymous=True)
    my_node = Nodo()
    my_node.start()
