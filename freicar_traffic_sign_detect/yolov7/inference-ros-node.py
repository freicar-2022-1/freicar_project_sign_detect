"""
Code for calling the YOLO model is mostly taken from https://github.com/WongKinYiu/yolov7/blob/main/detect.py
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import numpy as np
import cv2
import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# CONFIGURATION BEGIN
WEIGHTS_PATH = 'runs/train/exp20/weights/best.pt'
AUGMENTED_INFERENCE = False
IMG_SIZE = 640
STRIDE = 32
CONF_THRES = 0.65
IOU_THRES = 0.45  # default
IN_TOPIC_CAM_IMAGES = "/freicar_3/d435/color/image_raw"
OUT_TOPIC_CAM_IMAGES_WITH_BOUNDING_BOXES = '/freicar_3/trafficsigndetect/prediction/image'
OUT_TOPIC_BOUNDING_BOXES = '/freicar_3/trafficsigndetect/prediction/raw'
# CONFIGURATION END

class MLInferenceNode:
    def __init__(self):
        self.cv_bridge = CvBridge()

        self.pred_image_publisher = rospy.Publisher(
            OUT_TOPIC_CAM_IMAGES_WITH_BOUNDING_BOXES, Image, queue_size=10)
        self.bbox_publisher = rospy.Publisher(
            OUT_TOPIC_BOUNDING_BOXES, BoundingBoxArray, queue_size=10)

        self.imgsz = IMG_SIZE

        # Initialize
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(
            WEIGHTS_PATH, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check img_size

        if False:
            self.model = TracedModel(self.model, self.device, opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.class_names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.class_colors = [[random.randint(0, 255)
                              for _ in range(3)] for _ in self.class_names]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

        self.bbox_subscriber = rospy.Subscriber(
            IN_TOPIC_CAM_IMAGES, Image, self.camera_image_received, queue_size=1, buff_size=6000000
        )

    def camera_image_received(self, msg: Image):
        """
        Callback function to handle incoming camera images.
        Passes them to the ML model for object detection.
        Publishes the results.
        """
        # if not hasattr(self, 'model'):
        #    # Model still initializing, cannot do inference
        #    return

        print("########")
        print("Got camera image")
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv2.imwrite('camera_image.jpg', cv2_img)
        except CvBridgeError as e:
            print(e)

        self.detect_bboxes(cv2_img, msg.header.stamp)

    def detect_bboxes(self, camera_img, cv2_img_timestamp):
        # img = np.swapaxes(camera_img, 0, 2)

        # dataset = LoadImages('camera_image.jpg',
        #                     img_size=self.imgsz, stride=self.stride)
        # dataset.__iter__()
        # assert dataset.__len__() == 1
        # path, img, im0s, vid_cap = dataset.__next__()

        img = letterbox(camera_img, self.imgsz, stride=STRIDE)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Init model
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        # Start inference
        t0 = time.time()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=AUGMENTED_INFERENCE)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=AUGMENTED_INFERENCE)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)
        t3 = time_synchronized()

        bboxes = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            detected_objects_str = ''
            im0 = camera_img

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Prepare detected results string
                for class_id in det[:, -1].unique():
                    n = (det[:, -1] == class_id).sum()  # detections per class
                    # add to string
                    detected_objects_str += f"{n} {self.class_names[int(class_id)]}, "

                # Write results
                for *xyxy, conf, class_id in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh

                    # Convert 0...1 bounding boxes to absolute pixel values
                    # Convert position wrt. bounding box center to position wrt. bounding box top left corner
                    width = xywh[2] * camera_img.shape[1]
                    height = xywh[3] * camera_img.shape[0]
                    top_left_x = xywh[0] * camera_img.shape[1] - width/2
                    top_left_y = xywh[1] * camera_img.shape[0] - height/2

                    bbox = BoundingBox()
                    bbox.label = int(class_id)
                    bbox.header.stamp = cv2_img_timestamp
                    bbox.pose.position.x = top_left_x
                    bbox.pose.position.y = top_left_y
                    bbox.dimensions.x = width
                    bbox.dimensions.y = height
                    bboxes.append(bbox)

                    # Add bbox to image
                    label = f'{self.class_names[int(class_id)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label,
                                 color=self.class_colors[int(class_id)], line_thickness=1)

            # Print time (inference + NMS)
            print(
                f"Detection done ({(1E3 * (t2 - t1)):.1f}ms inference, {(1E3 * (t3 - t2)):.1f}ms NMS)")

            if len(detected_objects_str):
                print(detected_objects_str)
            else:
                print("No objcets detected")

            # Stream results
            if False:
                cv2.imwrite('camera_image_pred.jpg', im0)

            # Publish image with predictions to RVIZ for debugging
            self.pred_image_publisher.publish(
                self.cv_bridge.cv2_to_imgmsg(im0))

            # Publish detected bboxes
            bbox_array = BoundingBoxArray(None, bboxes)
            bbox_array.header.stamp = cv2_img_timestamp
            self.bbox_publisher.publish(bbox_array)

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == "__main__":
    rospy.init_node("signdetect_node")
    rospy.loginfo("Starting traffic sign bounding box detection node...")
    mapping_node = MLInferenceNode()
    rospy.loginfo(
        "Sign detection node started, ready to process camera images")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Received keyboard interrupt, shutting down...")
    rospy.loginfo("Traffic sign bounding box detection node finished.")
