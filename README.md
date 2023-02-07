# Traffic sign detection with the YOLOv7 neural network, ARUCO marker detection, and mapping for ROS

For autonomous driving, it is key to detect traffic signs and to map them.

This project is part of the [FreiCar lab course](http://freicar.informatik.uni-freiburg.de/) at the [University of Freiburg](https://uni-freiburg.de/), Germany.

Chosen landmarks for mapping were the German stop sign, the this-junction-priority sign, the autonomous driving landmark sign and orange traffic cones. These have been distributed manually in a robot hall. Afterwards, one of the FreiCar cars was driven remote-controlled through the hall, recording the front camera image ([Intel RealSense D435](https://www.intelrealsense.com/depth-camera-d435/)) and ground-truth poses from an indoor localization system ([HTC Vive Tracker](https://www.vive.com/us/accessory/tracker3/)). Object detection and mapping were run offline on a powerful machine.

![Sensor-equipped RC car, a stop sign (class 0), a this-junction-priority-sign (class 1), an autonomous driving sign (clasc 2) and a traffic cone (class 3)](images/car-and-classes.jpg)

Traffic signs were printed out to paper and glued on wooden mounts. The size of the traffic signs is 10x10 centimeters. Below, an [ARUCO marker](https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/) encoding the sign type was printed.

Components have been implemented as ROS nodes. ROS ([Robot Operating System](https://www.ros.org/)) provides a framework for basic common robot tasks.

## Training the YOLOv7 model

For object detection of camera images with traffic signs, we obtained best results using the YOLOv7X neural network, pretrained on the COCO dataset (download [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt)).

See below for instructions on how to convert between the label formats.

We trained all layers on the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html), only including the stop sign (class 14) and this-junction-priority sign (class 11):
```bash
$ python train.py \
    --cfg cfg/training/yolov7x_freicar.yaml \
    --data data/freicar-gtsrb.yaml \
    --weights yolov7x.pt \
    --cache-images \
    --epochs 400 \
    --batch-size 8 \
    --img-size 640 640 \
    --multi-scale
```

Afterwards, we freezed the backbone of the network (see file `freicar_traffic_sign_detect/yolov7/cfg/training/yolov7x_freicar.yaml` for the network architecture) that counts 59 layers and trained the model on our own small traffic sign dataset, using the following command:

```bash
$ python train.py \
    --cfg cfg/training/yolov7x_freicar.yaml \
    --data data/freicar-1-2.yaml \
    --weights PathToBestWeightsOfTrainingOnGTSRB.pt \
    --cache-images \
    --epochs 400 \
    --batch-size 16 \
    --img-size 640 640 \
    --multi-scale \
    --freeze 59
```

The resulting best weights file (`runs/train/expXX/best.pt`) was used for the inference ROS node.

### Training results on the test split

Precision/confidence curve:
![Precision/confidence curve](images/P_curve.png)

Recall/confidence curve:
![Recall/confidence curve](images/R_curve.png)

F1/confidence curve:
![Recall/confidence curve](images/F1_curve.png)

Confusion matrix:
![Confusion matrix](images/confusion_matrix.png)

### Inference examples on the test split
![Inference example 1](images/test-split-predictions/pred-1.jpg)
![Inference example 2](images/test-split-predictions/pred-2.jpg)
![Inference example 3](images/test-split-predictions/pred-3.jpg)

### GTSRB to COCO label coverter

After downloading the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html) (see `gtsrb-to-coco/download_dataset.sh`), the label format needed to be converted into the COCO label format used by YOLOv7.

This is done by running `$ cd gtsrb-to-coco` and afterwards `$ python gtsrb-to-coco.py`.

Because stop signs and this-junction-priority signs are the only ones from the dataset that were used in our project, the script only converts samples of these two classes.

### Labelme to COCO label converter

Furthermore, we created our own dataset consisting of ~900 images taken in our robot hall. We annotated them manually using [labelme](https://github.com/wkentaro/labelme). If you are interested in our dataset, please contact us.

To convert the label format used by labelme to the COCO label format, run `$ cd labelme-to-coco` and afterwards `$ python labelme2coco.py`.

## ROS nodes 

The project consists of three [ROS](https://www.ros.org/) nodes.

### YOLOv7 traffic sign detector

The traffic sign detector is located in the `freicar_traffic_sign_detect` directory.

Using the [YOLOv7](https://github.com/WongKinYiu/yolov7) neural network by Wang et al. and images from the car front camera, bounding boxes of traffic signs are detected. They are classified and passed to the mapping algorithm.

YOLOv7's [original repository](https://github.com/WongKinYiu/yolov7) has been cloned and extended in order to integrate it into the ROS and FreiCar ecosystem.

### ARUCO marker traffic sign detector

The aruco detector is located in the `freicar_aruco_detect` directory.

Outputs a message with:
1) Bounding box: position and dimensions (in pixel space)
2) The sign type: Stop, Priority, Autonomous driving, according to the Id’s of Markers
3) Timestamp

### Mapping node

The mapping node is located in the `freicar_mapping` directory.

It receives bounding boxes of detected signs from either the Aruco detector, the YOLOv7 traffic sign detector, or both (see paragraph about running the nodes for parameters). The incoming bounding box messages are synchronized with the depth images from the D435 RGBD camera and Camera Info messages.

From the bounding boxes, we calculate the 2d sign pose and orientation in the camera frame (using a deprojection function provided by Intel librealsense2and) then transform it to world frame using the TF TransformListener.

The mapping node then tries to associate the measured signs to known signs using a simple greedy data association scheme. If a good association has been found, we update the sign's pose using a running average and keep track of the most frequently observed type for each sign using a majority vote. The mapping node also publishes marker messages for each sign to visualize in Rviz.


## Building and running the ROS nodes

Please follow these instructions for building and running the ROS nodes.

### Mapping algorithm and ARUCO marker detection

The commands have to be run inside the docker in the `~/freicar_ws` directory.
- To build the nodes:
    1. Make sure required Python modules are installed: basic ROS modules, [jsk_recognition_msgs](http://wiki.ros.org/jsk_recognition_msgs), [pyrealsense2](https://pypi.org/project/pyrealsense2/), cv2, numpy
    2. run `catkin build -cs` from workspace directory

- To run the nodes:
```
# Street sign detection node:
rosrun freicar_street_sign street_sign.py
# Mapping node:
rosrun freicar_mapping mapping_node.py [--yolo] [--aruco]
```
If the packages aren't found, check `echo $ROS_PACKAGE_PATH`
- To clean the build artifacts just for our nodes:
```
catkin clean freicar_street_sign
catkin clean freicar_mapping
```

### Traffic sign detection (YOLOv7)

#### Initial setup

1. `$ cd freicar_traffic_sign_detect`
2. `$ pip install virtualenv`
3. `$ virtualenv .venv`
4. `$ source .venv/bin/activate`
5. `$ pip install -r yolov7/requirements.txt`

#### Running the inference node

Please have a look at the beginning of the file `freicar_traffic_sign_detect/yolov7/inference-ros-node.py` and ensure the configuration (e.g. ROS topics, path to the model weights, confidence thresholds) is correct.

1. `$ cd freicar_traffic_sign_detect/yolov7`
2. `$ source .venv/bin/activate`
3. `$ python inference-ros-node.py`
