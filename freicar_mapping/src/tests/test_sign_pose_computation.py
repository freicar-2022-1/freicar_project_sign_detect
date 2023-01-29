import rosbag
import rospy
from lib.utils import get_aruco_bbox, get_static_caminfo, pose2str, bbox2str
from lib.deprojection import get_relative_pose_from_bbox


def test_sign_pose_computation():
    rospy.init_node('test', anonymous=True)
    bagfile = '../data/test_images.bag'
    image_topic = '/freicar_3/d435/color/image_raw'
    depth_topic = '/freicar_3/d435/aligned_depth_to_color/image_raw'

    bag = rosbag.Bag(bagfile)
    img_messages = bag.read_messages(image_topic)
    depth_messages = bag.read_messages(depth_topic)

    img_msg = next(img_messages)
    depth_msg = next(depth_messages)

    # maps aruco dictionary ids to our labels
    id_mapping = {1: 0, 3: 1, 10: 2}
    bounding_boxes = get_aruco_bbox(img_msg, id_mapping, debug=False)

    for bbox in bounding_boxes.boxes:
        pose = get_relative_pose_from_bbox(bbox, get_static_caminfo(), depth_msg)
        print('--------------')
        print(bbox2str(bbox))
        print(pose2str(pose))


if __name__ == '__main__':
    test_sign_pose_computation()
