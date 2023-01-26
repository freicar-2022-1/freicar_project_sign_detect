# Freicar group 1 final project: Traffic sign detection and pose estimation

## Building and running
The commands have to be run inside the docker in the `~/freicar_ws` directory.
- To build the nodes:
```
catkin build -cs
```
- To run the nodes:
```
# Street sign detection node:
rosrun freicar_street_sign street_sign.py
# Mapping node:
rosrun freicar_mapping mapping_node.py
```
If the packages aren't found, check `echo $ROS_PACKAGE_PATH`
- To clean the build artifacts just for our nodes:
```
catkin clean freicar_street_sign
catkin clean freicar_mapping
```
