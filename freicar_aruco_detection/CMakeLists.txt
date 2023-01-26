cmake_minimum_required(VERSION 3.0.2)
project(freicar_aruco_detection)


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -ds
		OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
		OUTPUT_STRIP_TRAILING_WHITESPACE
		)
if (${LSB_RELEASE_ID_SHORT} STREQUAL "Ubuntu 20.04.1 LTS")
	message("Ubuntu 20.04 detected...")
	find_package(OpenCV REQUIRED)
else()
	message("Ubuntu < 20.04 detected....")
	find_package(OpenCV 3 REQUIRED)
endif()



#find_package (Eigen3 REQUIRED NO_MODULE)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  freicar_common
  freicar_sign_detect
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

catkin_package(CATKIN_DEPENDS
        roscpp
        rospy
        sensor_msgs
        std_msgs
        freicar_sign_detect
        freicar_common
        image_transport
  	    cv_bridge
#  INCLUDE_DIRS include
#  LIBRARIES freicar_sign_detect
#  DEPENDS system_lib
)
include_directories(
		include
		${catkin_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${EIGEN_INCLUDES}
)


add_executable(${PROJECT_NAME}_node src/freicar_aruco_detection_node.cpp)
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
		${catkin_LIBRARIES}
		${OpenCV_LIBS}
		${CMAKE_THREAD_LIBS_INIT}
		pthread
		Eigen3::Eigen
		)

