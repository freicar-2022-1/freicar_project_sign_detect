from geometry_msgs.msg import PoseStamped
from lib.sign import Sign
from tf.transformations import euler_from_quaternion
import rospy
import numpy as np


class Map:
    def __init__(self):
        """
        This class represents the whole landmark map as a list of Signs and tries to associate
        and incorporate new measurements.
        """
        self.signs: List[Sign]
        self.signs = []

        # minimum probability for two signs to be associated.
        # TODO: tune this
        self.association_threshold = 0.15

    def add_observation(self, sign_pose: PoseStamped, sign_type: int):
        """
        Add the observed sign to the map. If it can be associated to an existing sign, the pose
        is updated, otherwise we add a new sign.
        -----------
        Parameters:
            sign_pose: a PoseStamped message with position and orientation of the measured sign
            sign_type: an integer encoding the sign type (standard way, 0
        --
        Sign type correspond to colors as follows:
        Label 0/Stop
        Label 1/Priority
        Label 2/Autonomous driving
        Label 3/Traffic cone
        --
        """
        if sign_pose.header.frame_id != "world":
            raise ValueError(f"Tried to add sign pose with frame {sign_pose.header.frame_id} "
                             + "to the map. Can only add poses with frame 'world'.")

        # get x,y,theta sign pose from the PoseStamped msg
        x = sign_pose.pose.position.x
        y = sign_pose.pose.position.y

        qx = sign_pose.pose.orientation.x
        qy = sign_pose.pose.orientation.y
        qz = sign_pose.pose.orientation.z
        qw = sign_pose.pose.orientation.w

        _, _, theta = euler_from_quaternion([qx, qy, qz, qw])

        # create a candidate Sign object for comparing to other signs and maybe adding
        # to the map
        observed_sign = Sign(np.array([x, y, theta]), sign_type)

        # array with association probabilities for each known sign
        p_assoc = np.zeros((len(self.signs),))
        for i, sign in enumerate(self.signs):
            p_assoc[i] = sign.probability_equals(observed_sign)

        if len(p_assoc) == 0 or np.max(p_assoc) <= self.association_threshold:
            # Didn't find any valid association, add a new sign to the map
            # TODO: how do we deal with random measurements? Currently everything will be added
            # Possible solution: Add them, but only visualize them after they've been observed
            # a few times
            self.signs.append(observed_sign)
            rospy.loginfo(f"Added new {str(observed_sign)} to the map")
        else:
            # Good association found, update the sign's pose
            most_probable_assoc = np.argmax(p_assoc)
            self.signs[most_probable_assoc].update(observed_sign)
            rospy.loginfo(f"Updated pose of known {str(self.signs[most_probable_assoc])} "
                          + f"with new observed {str(observed_sign)}")

    def publish_markers(self, pub: rospy.Publisher, timestamp: rospy.Time):
        """
        Publish rviz markers for all signs in the map.
        -----------
        Parameters:
            pub (rospy.Publisher): The marker publisher to use.
        """
        # TODO: is MarkerArray better?
        for i, sign in enumerate(self.signs):
            marker = sign.get_marker(i, timestamp)
            pub.publish(marker)
