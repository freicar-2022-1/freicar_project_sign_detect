#!/usr/bin/env python3
import sys
import rospy

if __name__ == '__main__':
    rospy.init_node('mapping_node')
    rospy.loginfo("mapping_node started!")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    rospy.loginfo("finsihed!")
