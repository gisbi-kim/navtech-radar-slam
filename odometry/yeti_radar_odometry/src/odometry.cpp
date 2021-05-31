#include <thread>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include "radar_utils.hpp"
#include "features.hpp"
#include "association.hpp"

typedef pcl::PointXYZI PointType;

ros::Publisher pubOdom;
ros::Publisher pubLaserCloudLocal, pubLaserCloudGlobal;
Eigen::MatrixXd currOdom;

// modified for MulRan dataset batch evaluation 
int main(int argc, char *argv[]) 
{
	ros::init(argc, argv, "yetiOdom");
	ros::NodeHandle nh;

    omp_set_num_threads(8);

	pubOdom = nh.advertise<nav_msgs::Odometry>("/yeti_odom", 100);
    currOdom = Eigen::MatrixXd::Identity(4, 4); // initial pose is I

	pubLaserCloudLocal = nh.advertise<sensor_msgs::PointCloud2>("/yeti_cloud_local", 100);
	pubLaserCloudGlobal = nh.advertise<sensor_msgs::PointCloud2>("/yeti_cloud_global", 100);

    std::string seq_dir;
	nh.param<std::string>("seq_dir", seq_dir, ""); // pose assignment every k frames 
    std::string radardir = seq_dir + "sensor_data/radar/";
    std::string datadir = radardir + "polar_oxford_form/";

    // sensor params 
    int min_range = 58;                 // min range of radar points (bin)
    float radar_resolution = 0.0432;    // resolution of radar bins in meters per bin
    float cart_resolution = 0.2592;     // meters per pixel
    int cart_pixel_width = 964;         // height and width of cartesian image in pixels
    bool interp = true;
    int keypoint_extraction = 1;        // 0: cen2018, 1: cen2019, 2: orb

    // cen2018 parameters
    float zq = 3.0;
    int sigma_gauss = 17;
    // cen2019 parameters
    int max_points = 10000;
    // ORB descriptor / matching parameters
    int patch_size = 21;                // width of patch in pixels in cartesian radar image
    float nndr = 0.80;                  // Nearest neighbor distance ratio
    // RANSAC
    double ransac_threshold = 0.35;
    double inlier_ratio = 0.90;
    int max_iterations = 100;
    // MDRANSAC
    int max_gn_iterations = 10;
    double md_threshold = pow(ransac_threshold, 2);
    // DOPPLER
    double beta = 0.049;

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);

    std::string method;
    if(keypoint_extraction == 0) method = "cen2018";
    if(keypoint_extraction == 1) method = "cen2019";
    if(keypoint_extraction == 2) method = "orb";

    // File for storing the results of estimation on each frame (and the accuracy)
    std::ofstream ofs;
    ofs.open( radardir + "yeti_odom_result_" + method + ".csv", std::ios::out );
    // ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2,xmd,ymd,yawmd,xdopp,ydopp,yawdopp\n";
    
    std::cout << "writing file as " << "yeti_odom_result.csv" << std::endl; 
    
    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    // BRUTEFORCE_HAMMING for ORB descriptors FLANNBASED for cen2019 descriptors
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    Eigen::MatrixXd targets, cart_targets1, cart_targets2;
    Eigen::MatrixXd cart_feature_cloud;
    std::vector<int64_t> t1, t2;

    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    for (uint i = 0; i < radar_files.size() - 1; ++i) 
    {
        if( i % 100 == 0)
            std::cout << i << "/" << radar_files.size() << std::endl;

        if (i > 0) {
            t1 = t2; desc1 = desc2.clone(); cart_targets1 = cart_targets2;
            kp1 = kp2; img2.copyTo(img1);
        }
        load_radar(datadir + "/" + radar_files[i], times, azimuths, valid, fft_data, CIR204); // use CIR204 for MulRan dataset

        if (keypoint_extraction == 0)
            cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        if (keypoint_extraction == 1)
            cen2019features(fft_data, max_points, min_range, targets); // targets: 3xN
        if (keypoint_extraction == 0 || keypoint_extraction == 1) {
            radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img2, CV_8UC1);  // NOLINT
            polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets2, t2);
            cart_feature_cloud = cart_targets2;
            convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
            // cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
                // radar_resolution, 0.3456, 722, desc2);
            detector->compute(img2, kp2, desc2);
        }
        if (keypoint_extraction == 2) {
            detector->detect(img2, kp2);
            detector->compute(img2, kp2, desc2);
            convert_from_bev(kp2, cart_resolution, cart_pixel_width, cart_targets2);
            getTimes(cart_targets2, azimuths, times, t2);
        }

        if (i == 0)
            continue;

        // Match keypoint descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(desc1, desc2, knn_matches, 2);

        // Filter matches using nearest neighbor distance ratio (Lowe, Szeliski)
        std::vector<cv::DMatch> good_matches;
        for (uint j = 0; j < knn_matches.size(); ++j) {
            if (!knn_matches[j].size())
                continue;
            if (knn_matches[j][0].distance < nndr * knn_matches[j][1].distance) {
                good_matches.push_back(knn_matches[j][0]);
            }
        }

        // Convert the good key point matches to Eigen matrices
        Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(2, good_matches.size());
        Eigen::MatrixXd p2 = p1;
        std::vector<int64_t> t1prime = t1, t2prime = t2;
        for (uint j = 0; j < good_matches.size(); ++j) {
            p1(0, j) = cart_targets1(0, good_matches[j].queryIdx);
            p1(1, j) = cart_targets1(1, good_matches[j].queryIdx);
            p2(0, j) = cart_targets2(0, good_matches[j].trainIdx);
            p2(1, j) = cart_targets2(1, good_matches[j].trainIdx);
            t1prime[j] = t1[good_matches[j].queryIdx];
            t2prime[j] = t2[good_matches[j].trainIdx];
        }
        t1prime.resize(good_matches.size());
        t2prime.resize(good_matches.size());

        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);
        double delta_t = (time2 - time1) / 1000000.0;

        // v1: Compute the transformation using RANSAC
        Ransac ransac(p2, p1, ransac_threshold, inlier_ratio, max_iterations);
        srand(i);
        ransac.computeModel();
        Eigen::MatrixXd T;  // T_1_2
        ransac.getTransform(T);

        // v2: Compute the transformation using motion-distorted RANSAC
        MotionDistortedRansac mdransac(p2, p1, t2prime, t1prime, md_threshold, inlier_ratio, max_iterations);
        mdransac.setMaxGNIterations(max_gn_iterations);
        mdransac.correctForDoppler(false);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd;
        mdransac.getTransform(delta_t, Tmd);
        Tmd = Tmd.inverse();
        // Eigen::VectorXd wbar;
        // mdransac.getMotion(wbar);

        // v3: MDRANSAC + Doppler
        mdransac.correctForDoppler(true);
        mdransac.setDopplerParameter(beta);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd2 = Eigen::MatrixXd::Zero(4, 4);
        mdransac.getTransform(delta_t, Tmd2);
        Tmd2 = Tmd2.inverse();

        // not use for this example 
        // Retrieve the ground truth to calculate accuracy
        // std::vector<float> gtvec;
        // if (!get_groundtruth_odometry(gt, time1, time2, gtvec)) {
        //     std::cout << "ground truth odometry for " << time1 << " " << time2 << " not found... exiting." << std::endl;
        //     return 0;
        // }
        // float yaw = -1 * asin(T(0, 1));
        // float yaw2 = -1 * asin(Tmd(0, 1));
        // float yaw3 = -1 * asin(Tmd2(0, 1));
        float yaw = asin(-1 * T(0, 1));
        float yaw2 = asin(-1 * Tmd(0, 1));
        float yaw3 = asin(-1 * Tmd2(0, 1));

        // Write estimated and ground truth transform to the csv file
        ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ",";
        // ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        ofs << 0.0 << "," << 0.0 << "," << 0.0 << ","; // dummy gt 
        ofs << time1 << "," << time2 << "," << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << ",";
        ofs << Tmd2(0, 3) << "," << Tmd2(1, 3) << "," << yaw3 << "\n";

        // curuent state
        currOdom = currOdom * Tmd;
        Eigen::Matrix3d currOdomRot = currOdom.block(0,0,3,3);
        Eigen::Vector3d currOdomEuler = currOdomRot.eulerAngles(0,1,2);
        // Eigen::Vector3d currEulerVec = currOdomRot.eulerAngles(2, 1, 0);
        // float currYaw = asin(-1 * currOdom(0, 1));
        // float currYaw = currEulerVec(3);
        float currYaw = float(currOdomEuler(2));
        // std::cout << "currOdomEuler: " << currOdomEuler(0) << ", " << currOdomEuler(1) << ", " << currOdomEuler(2) << std::endl;

        // double currOdomTimeSec = double(time2) / 100000000.0;
        // std::cout << " currOdomTimeSec " << currOdomTimeSec << std::endl;
        auto currOdomROSTime = ros::Time::now();

        // pub
        nav_msgs::Odometry odom;
        odom.header.frame_id = "/camera_init";
        odom.child_frame_id = "/yeti_odom"; 
        // odom.header.stamp = ros::Time().fromSec( currOdomTimeSec );
        odom.header.stamp = currOdomROSTime;
        odom.pose.pose.position.x = currOdom(0, 3);
        odom.pose.pose.position.y = currOdom(1, 3);
        odom.pose.pose.position.z = currOdom(2, 3);
        odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, currYaw);
        pubOdom.publish(odom); // last pose 

        float contant_z_nonzero = 1.0; // for scan context (in this naive case, we can say we will use binary scan context).
        pcl::PointCloud<PointType>::Ptr laserCloudLocal(new pcl::PointCloud<PointType>());
        for (uint pt_idx = 0; pt_idx < cart_feature_cloud.cols(); ++pt_idx) {
            // cart_feature_cloud: 3xN (i.e., [x,y,z]' x N points)
            PointType feature_point;
            feature_point.x = cart_feature_cloud(0, pt_idx);
            feature_point.y = cart_feature_cloud(1, pt_idx);
            feature_point.z = cart_feature_cloud(2, pt_idx) + contant_z_nonzero;
            laserCloudLocal->push_back(feature_point);
        }
        sensor_msgs::PointCloud2 laserCloudLocalMsg;
        pcl::toROSMsg(*laserCloudLocal, laserCloudLocalMsg);
        laserCloudLocalMsg.header.frame_id = "/camera_init";
        // laserCloudLocalMsg.header.stamp = ros::Time().fromSec (currOdomTimeSec );
        laserCloudLocalMsg.header.stamp = currOdomROSTime;
        pubLaserCloudLocal.publish(laserCloudLocalMsg);

        pcl::PointCloud<PointType>::Ptr laserCloudGlobal(new pcl::PointCloud<PointType>());
        for (uint pt_idx = 0; pt_idx < cart_feature_cloud.cols(); ++pt_idx) {
            // cart_feature_cloud: 3xN (i.e., [x,y,z]' x N points)
            PointType feature_point_global;
            auto local_x = cart_feature_cloud(0, pt_idx);
            auto local_y = cart_feature_cloud(1, pt_idx);
            auto local_z = cart_feature_cloud(2, pt_idx) + contant_z_nonzero;
            feature_point_global.x = currOdom(0,0)*local_x + currOdom(0,1)*local_y + currOdom(0,2)*local_z + currOdom(0,3);
            feature_point_global.y = currOdom(1,0)*local_x + currOdom(1,1)*local_y + currOdom(1,2)*local_z + currOdom(1,3);
            feature_point_global.z = currOdom(2,0)*local_x + currOdom(2,1)*local_y + currOdom(2,2)*local_z + currOdom(2,3);
            laserCloudGlobal->push_back(feature_point_global);
        }
        sensor_msgs::PointCloud2 laserCloudGlobalMsg;
        pcl::toROSMsg(*laserCloudGlobal, laserCloudGlobalMsg);
        laserCloudGlobalMsg.header.frame_id = "/camera_init";
        // laserCloudGlobalMsg.header.stamp = ros::Time().fromSec (currOdomTimeSec );
        laserCloudGlobalMsg.header.stamp = currOdomROSTime;
        pubLaserCloudGlobal.publish(laserCloudGlobalMsg);

        // vis (optional)
        if(0) {
            cv::Mat img_matches;
            cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("good", img_matches);
            cv::waitKey(20);
        }

        // make sure under 10hz, to privde enough time for following PGO module
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    ros::spin();

    return 0;
}
