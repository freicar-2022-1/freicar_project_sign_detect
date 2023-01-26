#include "freicar_aruco_detection_node.h"

using cv::Point2f;
using cv::Ptr;
using cv::Vec3d;
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
FreicarSignDetect::FreicarSignDetect(const std::unique_ptr<ros::NodeHandle>& nodehandle, const std::string agent_name)
        : image_transport_(*nodehandle) {
    agent_name_ = agent_name;
    image_sub_ = make_unique<image_transport::Subscriber>(image_transport_.subscribe(
            agent_name_ + "/d435/color/image_raw"
            , 1, &FreicarSignDetect::ImageCallback, this));
    ar_detect_ = make_unique<ArucoDetect>(nodehandle, agent_name);
    ar_detect_->setIntrinsics(725.3607788085938, 725.3607788085938, 596.6277465820312, 337.6268615722656);
}

// void FreicarSignDetect::CamInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg){
//    if (!ar_detect_->cam_initialized_){
//    ROS_INFO("got camera info topic...");
//    ar_detect_->setIntrinsics(msg->K[0], msg->K[4], msg->K[2], msg->K[5]);
//    }
//}

void FreicarSignDetect::ImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat img = cv_ptr->image;

    // get aruco markers....
    ar_detect_->detect(img, msg->header.stamp);
}

// ARUCO !!!!!!!!!!!!!!!!!!!!!!
ArucoDetect::ArucoDetect(const std::unique_ptr<ros::NodeHandle>& nodehandle, std::string name)
        : cam_initialized_(false) {
    detectorParams = cv::aruco::DetectorParameters::create();

    aruco_pub_ = nodehandle->advertise<freicar_common::FreiCarSigns>(name + "/traffic_signs", 10);

    std::string path = ros::package::getPath("freicar_sign_detect") + "/param/detector_params.yml";

    bool readOk = readDetectorParameters(path, detectorParams);
    if (!readOk) {
        std::cerr << "Invalid detector parameters file" << std::endl;
    }

    m_Dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    camMatrix = cv::Mat::eye(3, 3, CV_64F);
}

void ArucoDetect::setIntrinsics(float fx, float fy, float cx, float cy) {
    camMatrix.at<double>(0, 0) = fx;
    camMatrix.at<double>(0, 2) = cx;

    camMatrix.at<double>(1, 1) = fy;
    camMatrix.at<double>(1, 2) = cy;
    cam_initialized_ = true;
}

bool ArucoDetect::readDetectorParameters(std::string filename, Ptr<cv::aruco::DetectorParameters>& params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

void ArucoDetect::detect(cv::Mat img, ros::Time stamp) {
    std::vector<int> ids;
    std::vector<std::vector<Point2f>> corners, rejected;
    std::vector<Vec3d> rvecs, tvecs;

    bool estimatePose = true;
    const float markerLength = 0.117f;

    // detect markers and estimate pose
    cv::aruco::detectMarkers(img, m_Dictionary, corners, ids, detectorParams, rejected);
    if (estimatePose && ids.size() > 0)
        cv::aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);
    // draw results
    cv::Mat imageCopy;
    img.copyTo(imageCopy);
    if (ids.size() > 0) {
        cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

        if (estimatePose) {
            for (unsigned int i = 0; i < ids.size(); i++)
                cv::aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 0.5f);
        }
    }

    //    cv::imshow("marker", imageCopy);
    //    cv::waitKey(10);
    publishRos(tvecs, rvecs, ids, stamp);
    // std::cout << "done ..." << std::endl;
}

void ArucoDetect::publishRos(std::vector<Vec3d>& Tvec, std::vector<Vec3d>& Rvec, std::vector<int>& ids,
                             ros::Time stamp) {
    freicar_common::FreiCarSigns signs_msg;

    signs_msg.header.frame_id = "/zed_camera";
    signs_msg.header.stamp = stamp;
    signs_msg.header.seq = 0;

    // TODO : remove. just added to remove warnings
    Rvec.size();
    for (size_t i = 0; i < Tvec.size(); i++) {
        //        cv::Mat rot(3, 3, CV_64FC1);
        //        cv::Rodrigues(Rvec.at(i), rot);
        //        Eigen::Matrix3d eigen_rot;
        //        cv::cv2eigen(rot, eigen_rot);
        //        Eigen::Quaterniond q_rot(eigen_rot);

        const Vec3d& translation = Tvec.at(i);
        freicar_common::FreiCarSign sign_msg;
        sign_msg.x = translation[0];
        sign_msg.y = translation[1];
        sign_msg.z = translation[2];

        const int id = ids.at(i);
        sign_msg.type = id;

        //        pose_msg.orientation.w = q_rot.w();
        //        pose_msg.orientation.x = q_rot.x();
        //        pose_msg.orientation.y = q_rot.y();
        //        pose_msg.orientation.z = q_rot.z();

        signs_msg.signs.push_back(sign_msg);
        //        std::cout << "Translation : " << translation[0] << " / " << translation[1] << " / " << translation[2]
        //        << std::endl;
    }

    //    std::cout << "Sending out " << std::endl;
    aruco_pub_.publish(signs_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "freicar_sign_detect_node");
    // unique pointer is enough
    std::unique_ptr<ros::NodeHandle> n = make_unique<ros::NodeHandle>();

    std::string agent_name;
    if (!ros::param::get("~agent_name", agent_name)) {
        ROS_ERROR("ERROR: could not find parameter: agent_name. check the launch file.");
        std::exit(EXIT_FAILURE);
    }

    FreicarSignDetect detector(n, agent_name);

    ros::Rate loop_rate(100);
    std::cout << "Freicar sign detector started! ... " << std::endl;
    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}

