import numpy as np
import rospy
from typing import Tuple
from lib.utils import sign_pose_2_marker_msg
from scipy.stats import norm
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


class Sign:
    def __init__(self, pose: np.ndarray, sign_type: int):
        """
        Class representing a street sign, with 2d position, rotation around the z axis and sign
        type.
        -----------
        Parameters:
            pose: Numpy array of shape (3,) containing the first measured x,y,theta of the new
                  sign
            sign_type: Integer encoding the type of sign as described below
        --
        Sign type correspond to colors as follows:
        Label 0/Stop
        Label 1/Priority
        Label 2/Autonomous driving
        Label 3/Traffic cone
        --
        """
        # pose is always a numpy array([x,y,theta])
        self.pose = pose

        # number of measurements received, for updating the average pose
        self.measurements = 1

        # try to be a little probabilistic about our type of sign by counting and taking the
        # most frequent one.
        self.sign_type_counts = [0, 0, 0, 0]
        self.sign_type_counts[sign_type] += 1

    def __str__(self) -> str:
        pose, sign_type = self.get_current_average()
        if sign_type == 0:
            type_str = "Stop sign"
        elif sign_type == 1:
            type_str = "Junction priority sign"
        elif sign_type == 2:
            type_str = "Autonomous driving sign"
        elif sign_type == 3:
            type_str = "Traffic cone"
        else:
            type_str = "Sign of unknown type"

        s = f"{type_str} at position (x={pose[0]}, y={pose[1]}) with angle "
        s += f"{pose[2]} rad ({pose[2]/np.pi*180} deg)"
        return s

    def probability_equals(self, other_sign: 'Sign') -> float:
        """
        Tries to estimate a likelihood for this sign and the given measured sign to be the same
        (for data association). The probability is influenced by the distance (theta is ignored
        for now, as it is really noisy) and the sign type. If the sign types match, the
        allowed distance is higher than if they don't match.
        -----------
        Parameters:
            other_sign: The measured Sign
        -----------
        Returns:
            p: a sort of likelihood that these two signs are the same
        """
        other_pose, sign_type = other_sign.get_current_average()
        dist = np.linalg.norm(self.pose[:2] - other_pose[:2])
        own_sign_type = self.get_current_average()[1]

        # TODO: tune the two stdevs below
        if sign_type == own_sign_type:
            # 30cm standard deviation if sign types match
            stdev = 0.3
        else:
            # 15cm standard deviation if they don't
            stdev = 0.15

        # evaluate a normal distribution to get a probability for the distance between
        # measured sign and this sign
        p = norm(loc=0, scale=stdev).pdf(dist)
        return p

    def get_current_average(self) -> Tuple[np.ndarray, int]:
        """
        Calculate the current average sign type & pose.
        -----------
        Returns:
            (pose, sign_type): The average pose and sign type over observations made so far.
        """
        return self.pose, np.argmax(self.sign_type_counts)

    def update(self, other_sign: 'Sign'):
        """
        Given a new measurement and sign type, update this sign's current average pose and sign
        type. Sign type is updated by counting the measured occurences of each type and then
        using the most common one.
        -----------
        Parameters:
            other_sign: the measured sign whose position will be incorporated into this sign.
        """
        pose, sign_type = other_sign.get_current_average()
        # have to update the mean angle in this special way, because regular averaging won't
        # work for angles
        sines = np.sin(self.pose[2]) * self.measurements + np.sin(pose[2])
        cosines = np.cos(self.pose[2]) * self.measurements + np.cos(pose[2])
        self.pose[2] = np.arctan2(sines, cosines)

        # update pose average
        self.pose[:2] = (self.measurements * self.pose[:2] + pose[:2]) / (self.measurements + 1)

        # increase type occurence count
        self.sign_type_counts[sign_type] += 1

        self.measurements += 1

    def get_marker(
            self, id_: int, timestamp: rospy.Time = None, frame: str = "world", ns: str = 'sign'
    ) -> Marker:
        """
        Get an arrow marker message for publishing & visualizing in rviz.
        -----------
        Parameters:
            id_: unique id of this sign
            timestamp: rospy time stamp. If None, use current time
            frame: the reference frame, should always be the default "world"
            ns: namespace to be used in the marker message
        """
        pose, sign_type = self.get_current_average()
        if not timestamp:
            timestamp = rospy.get_time()

        # convert this sign's pose to a PoseStamped message for later conversion into a marker
        # TODO: refactor to avoid the extra step
        sign_pose = PoseStamped()
        sign_pose.header.stamp = timestamp
        sign_pose.header.frame_id = frame

        sign_pose.pose.position.x = self.pose[0]
        sign_pose.pose.position.y = self.pose[1]
        sign_pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0, 0, self.pose[2])
        sign_pose.pose.orientation.x = qx
        sign_pose.pose.orientation.y = qy
        sign_pose.pose.orientation.z = qz
        sign_pose.pose.orientation.w = qw

        marker = sign_pose_2_marker_msg(sign_pose, sign_type, id_, ns)
        return marker
