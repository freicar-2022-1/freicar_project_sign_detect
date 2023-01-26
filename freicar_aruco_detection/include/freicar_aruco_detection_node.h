#pragma once

#include <cv_bridge/cv_bridge.h>
#include <freicar_common/FreiCarSign.h>
#include <freicar_common/FreiCarSigns.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class ArucoDetect {
public:
    ArucoDetect(const std::unique_ptr<ros::NodeHandle>& nodehandle, std::string name);
    bool readDetectorParameters(std::string filename, cv::Ptr<cv::aruco::DetectorParameters>& params);
    void detect(cv::Mat img, ros::Time stamp);
    void setIntrinsics(float fx, float fy, float cx, float cy);
    void publishRos(std::vector<cv::Vec3d>& Tvec, std::vector<cv::Vec3d>& Rvec, std::vector<int>& ids, ros::Time stamp);

    cv::Ptr<cv::aruco::DetectorParameters> detectorParams;
    cv::Ptr<cv::aruco::Dictionary> m_Dictionary;

    cv::Mat camMatrix, distCoeffs;
    bool cam_initialized_;

private:
    ros::Publisher aruco_pub_;
};

class FreicarSignDetect {
public:
    FreicarSignDetect(const std::unique_ptr<ros::NodeHandle>& nodehandle, const std::string agent_name);
    void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
    //    void CamInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);

private:
    image_transport::ImageTransport image_transport_;
    std::unique_ptr<ArucoDetect> ar_detect_;
    std::unique_ptr<image_transport::Subscriber> image_sub_;
    std::unique_ptr<ros::Subscriber> cam_info_sub_;
    std::string agent_name_;
};
