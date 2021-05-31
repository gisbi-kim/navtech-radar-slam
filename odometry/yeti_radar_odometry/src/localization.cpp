#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "association.hpp"

void removeDoppler(Eigen::MatrixXd &p, Eigen::Vector3d vbar, double beta) {
    for (uint j = 0; j < p.cols(); ++j) {
        double phi = atan2f(p(1, j), p(0, j));
        double delta_r = beta * (vbar(0) * cos(phi) + vbar(1) * sin(phi));
        p(0, j) += delta_r * cos(phi);
        p(1, j) += delta_r * sin(phi);
    }
}

void removeMotionDistortion(Eigen::MatrixXd &p, std::vector<int64_t> tprime, Eigen::VectorXd wbar, int64_t t_ref) {
    for (uint j = 0; j < p.cols(); ++j) {
        double delta_t = (tprime[j] - t_ref) / 1.0e6;
        Eigen::MatrixXd T = se3ToSE3(wbar * delta_t);
        Eigen::Vector4d pbar = {p(0, j), p(1, j), 0, 1};
        pbar = T * pbar;
        p(0, j) = pbar(0);
        p(1, j) = pbar(1);
    }
}

// Rigid RANSAC
Eigen::MatrixXd computeAndGetTransform(Eigen::MatrixXd p2, Eigen::MatrixXd p1, double ransac_threshold,
    double inlier_ratio, int max_iterations){
    Ransac ransac(p2, p1, ransac_threshold, inlier_ratio, max_iterations);
    ransac.computeModel();
    Eigen::MatrixXd T;
    ransac.getTransform(T);
    return T;
}

// Assuming rotations are SO(2), this function returns the angle (yaw/heading) corresponding to the rotation
double getRotation(Eigen::MatrixXd T) {
    Eigen::MatrixXd Cmin = T.block(0, 0, 2, 2);
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(3, 3);
    C.block(0, 0, 2, 2) = Cmin;
    double eps = 1e-15;
    int i = 2, j = 1, k = 0;
    double c_y = sqrt(pow(C(i, i), 2) + pow(C(j, i), 2));
    double phi = 0;
    if (c_y > eps) {
        phi = atan2f(C(k, j), C(k, k));
    } else {
        phi = atan2f(-C(j, k), C(j, j));
    }
    return phi;
}

int main(int argc, const char *argv[]) {
    std::string root = "/media/backup2/2020_11_26";
    validateArgs(argc, argv, root);
    std::string datadir = root + "/radar";
    std::string gt = root + "/applanix/radar_loc_gt.csv";

    bool interp = true;
    float zq = 3.0;
    int sigma_gauss = 17;
    int patch_size = 21;
    float nndr = 0.99;
    double ransac_threshold = 0.75;
    double inlier_ratio = 0.90;
    int max_iterations = 1000;
    double beta = 0.049;

    float cart_resolution = 0.2384;
    int cart_pixel_width = 1048;
    int min_range = 42;
    float radar_resolution = 0.0596;

    float cart_res2 = 0.3576;
    int cart_width2 = 700;

    // File for storing the results of estimation on each frame (and the accuracy)
    std::ofstream ofs;
    ofs.open("localization_accuracy.csv", std::ios::out);

    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    Eigen::MatrixXd targets, cart_targets1, cart_targets2;
    std::vector<int64_t> t1, t2;

    std::vector<int64_t> times1, times2;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    std::ifstream ifs(gt);
    std::string line;
    std::getline(ifs, line);  // Clear out the header

    int i = 0;
    while (std::getline(ifs, line)) {
        std::cout << i << std::endl;
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        int64_t time1 = std::stoll(parts[0]) / 1000;
        int64_t time2 = std::stoll(parts[1]) / 1000;
        std::vector<double> gtvec;
        for (uint j = 2; j < parts.size(); ++j) {
            gtvec.push_back(std::stod(parts[j]));
        }

        std::string radar_file1 = parts[0] + ".png";

        load_radar(datadir + "/" + radar_file1, times1, azimuths, valid, fft_data, CIR204);
        cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp,
            img1, CV_8UC1, CIR204);
        polar_to_cartesian_points(azimuths, times1, targets, radar_resolution, cart_targets1, t1);
        convert_to_bev(cart_targets1, cart_resolution, cart_pixel_width, patch_size, kp1, t1);
        // detector->compute(img1, kp1, desc1);
        cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets1,
            radar_resolution, cart_res2, cart_width2, desc1);

        std::string radar_file2 = parts[1] + ".png";

        load_radar(datadir + "/" + radar_file2, times2, azimuths, valid, fft_data, CIR204);
        cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img2,
            CV_8UC1, CIR204);
        polar_to_cartesian_points(azimuths, times2, targets, radar_resolution, cart_targets2, t2);
        convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
        // detector->compute(img2, kp2, desc2);
        cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
            radar_resolution, cart_res2, cart_width2, desc2);

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

        // Compute the transformation using RANSAC
        srand(i);
        Eigen::MatrixXd T1 = computeAndGetTransform(p2, p1, ransac_threshold, inlier_ratio, max_iterations);

        Eigen::MatrixXd p1temp = p1, p2temp = p2;

        // Remove Doppler effects (first)
        Eigen::Vector3d vbar1 = {gtvec[3], gtvec[4], 0};
        Eigen::Vector3d vbar2 = {gtvec[6], gtvec[7], 0};
        // double v1 = gtvec[3];
        // double v2 = gtvec[5];
        removeDoppler(p1, vbar1, beta);
        removeDoppler(p2, vbar2, beta);
        srand(i);
        Eigen::MatrixXd T2 = computeAndGetTransform(p2, p1, ransac_threshold, inlier_ratio, max_iterations);

        // Remove motion distortion from the pointclouds (second)
        Eigen::VectorXd wbar1 = Eigen::VectorXd::Zero(6);
        wbar1(0) = gtvec[3];
        wbar1(1) = gtvec[4];
        wbar1(5) = gtvec[5];
        removeMotionDistortion(p1, t1prime, wbar1, times1[times1.size() - 1]);
        Eigen::VectorXd wbar2 = Eigen::VectorXd::Zero(6);
        wbar2(0) = gtvec[6];
        wbar2(1) = gtvec[7];
        wbar2(5) = gtvec[8];
        removeMotionDistortion(p2, t2prime, wbar2, times2[times2.size() - 1]);
        srand(i);
        Eigen::MatrixXd T3 = computeAndGetTransform(p2, p1, ransac_threshold, inlier_ratio, max_iterations);

        // Remove motion distortion (first)
        p1 = p1temp;
        p2 = p2temp;
        removeMotionDistortion(p1, t1prime, wbar1, times1[times1.size() - 1]);
        removeMotionDistortion(p2, t2prime, wbar2, times2[times2.size() - 1]);
        srand(i);
        Eigen::MatrixXd T4 = computeAndGetTransform(p2, p1, ransac_threshold, inlier_ratio, max_iterations);

        // Remove Doppler effects (second)
        removeDoppler(p1, vbar1, beta);
        removeDoppler(p2, vbar2, beta);
        srand(i);
        Eigen::MatrixXd T5 = computeAndGetTransform(p2, p1, ransac_threshold, inlier_ratio, max_iterations);

        // Retrieve the ground truth to calculate accuracy
        float yaw = getRotation(T1);
        float yaw2 = getRotation(T2);
        float yaw3 = getRotation(T3);
        float yaw4 = getRotation(T4);
        float yaw5 = getRotation(T5);

        // Write estimated and ground truth transform to the csv file
        ofs << T1(0, 2) << "," << T1(1, 2) << "," << yaw << ",";
        ofs << T2(0, 2) << "," << T2(1, 2) << "," << yaw2 << ",";
        ofs << T3(0, 2) << "," << T3(1, 2) << "," << yaw3 << ",";
        ofs << T4(0, 2) << "," << T4(1, 2) << "," << yaw4 << ",";
        ofs << T5(0, 2) << "," << T5(1, 2) << "," << yaw5 << ",";
        ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[2] << "," << time1 << "," << time2 << ",";
        ofs << gtvec[3] << "," << gtvec[4] << "," << gtvec[5] << "," << gtvec[6] << "\n";

        i++;

        std::cout << "GT: " << gtvec[0] << "," << gtvec[1] << "," << gtvec[2] << std::endl;
        std::cout << "RIGID: " << T1(0, 2) << "," << T1(1, 2) << "," << yaw << std::endl;
        std::cout << "DOPP: " << T2(0, 2) << "," << T2(1, 2) << "," << yaw2 << std::endl;
        std::cout << "DOPP + MD: " << T3(0, 2) << "," << T3(1, 2) << "," << yaw3 << std::endl;
        std::cout << "MD: " << T4(0, 2) << "," << T4(1, 2) << "," << yaw4 << std::endl;
        std::cout << "MD + DOPP: " << T5(0, 2) << "," << T5(1, 2) << "," << yaw5 << std::endl;

        // cv::Mat img_matches;
        // cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
        //          cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cv::imshow("good", img_matches);
        // cv::waitKey(0);
    }

    return 0;
}
