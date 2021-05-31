#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include "matplotlibcpp.h"  // NOLINT
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "association.hpp"
namespace plt = matplotlibcpp;

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

int get_closest_lidar(int64_t query_time, std::vector<std::string> lidar_files) {
    double min_delta = 0.25;
    int closest_lidar = -1;
    for (uint j = 0; j < lidar_files.size(); ++j) {
        std::vector<std::string> parts;
        boost::split(parts, lidar_files[j], boost::is_any_of("."));
        int64_t t = std::stoll(parts[0]);
        double delta = fabs((t - query_time) / 1.0e9);
        if (delta < min_delta) {
            min_delta = delta;
            closest_lidar = j;
        }
    }
    assert(closest_lidar != -1);
    return closest_lidar;
}

bool get_groundtruth_data(std::string gtfile, std::string sensor_file, std::vector<double> &gt) {
    std::vector<std::string> farts;
    boost::split(farts, sensor_file, boost::is_any_of("."));
    std::string ftime = farts[0];   // Unique timestamp identifier for the sensor_file to search for
    std::ifstream ifs(gtfile);
    std::string line;
    gt.clear();
    bool gtfound = false;
    std::getline(ifs, line);  // clear out the csv file header before searching
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        if (parts[0] == ftime) {
            for (uint i = 1; i < parts.size(); ++i) {
                gt.push_back(std::stod(parts[i]));
            }
            gtfound = true;
            break;
        }
    }
    return gtfound;
}

// Unwarping the cartesian image instead of target points is tricky.
// We need to do the opposite and "warp" the ideal output to find the pixel locations in the original, distorted image.
void undistort_radar_image(cv::Mat &input, cv::Mat &output, Eigen::VectorXd wbar, float cart_resolution,
    int cart_pixel_width, std::vector<int64_t> radar_times, std::vector<double> azimuths, double beta,
    bool remove_doppler, bool remove_motion) {
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    std::vector<double> times(radar_times.size());
    for (uint i = 0; i < radar_times.size(); ++i) {
        times[i] = double(radar_times[i] / 1.0e6);
    }
    double t_ref = times[times.size() - 1];

    std::vector<Eigen::Matrix4d> transforms(times.size());
    for (uint i = 0; i < times.size(); ++i) {
        double delta_t = times[i] - t_ref;
        transforms[i] = get_inverse_tf(se3ToSE3(wbar * delta_t));
    }

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < map_y.cols; ++j) {
        for (int i = 0; i < map_y.rows; ++i) {
            map_y.at<float>(i, j) = -1 * cart_min_range + j * cart_resolution;
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < map_x.rows; ++i) {
        for (int j = 0; j < map_x.cols; ++j) {
            map_x.at<float>(i, j) = cart_min_range - i * cart_resolution;
        }
    }

    cv::Mat orig_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat orig_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    double v = sqrt(pow(wbar(0), 2) + pow(wbar(1), 2));
    float r_min = v / 4;  // Prevents undefined warping behavior near sensor

#pragma omp parallel for collapse(2)
    for (int i = 0; i < orig_x.rows; ++i) {
        for (int j = 0; j < orig_y.cols; ++j) {
            // Undistorted cartesian coordinates
            float x_u = map_x.at<float>(i, j);
            float y_u = map_y.at<float>(i, j);
            float r_u = sqrt(pow(x_u, 2) + pow(y_u, 2));
            if (r_u < r_min)
                continue;
            double psi = atan2f(y_u, x_u);
            psi = wrapto2pi(psi);

            double idx = get_azimuth_index(azimuths, psi);

            Eigen::Vector4d xbar = {x_u, y_u, 0, 1};
            if (remove_motion)
                xbar = transforms[int(round(idx))] * xbar;

            float x_d = xbar(0);
            float y_d = xbar(1);

            if (remove_doppler) {
                double delta_r = beta * (wbar(0) * cos(psi) + wbar(1) * sin(psi));
                x_d -= delta_r * cos(psi);
                y_d -= delta_r * sin(psi);
            }

            // Convert into BEV pixel coordinates
            float u_bev = (cart_min_range + y_d) / cart_resolution;
            float v_bev = (cart_min_range - x_d) / cart_resolution;

            if (0 <= u_bev && u_bev < cart_pixel_width && 0 <= v_bev && v_bev <= cart_pixel_width) {
                orig_x.at<float>(i, j) = u_bev;
                orig_y.at<float>(i, j) = v_bev;
            }
        }
    }

    cv::remap(input, output, orig_x, orig_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

int main(int argc, const char *argv[]) {
    std::string root = "/media/backup2/2020_11_26";
    validateArgs(argc, argv, root);
    std::string radar_gt_file = root + "/applanix/radar_poses.csv";
    std::string lidar_gt_file = root + "/applanix/lidar_poses.csv";
    std::vector<std::string> lidar_files;
    get_file_names(root + "/lidar/", lidar_files, "bin");

    float cart_resolution = 0.2384;
    int cart_pixel_width = 586;
    int min_range = 42;
    float radar_resolution = 0.0596;
    float zq = 2.5;
    int sigma_gauss = 17;
    float beta = 0.049;

    std::vector<uint> green = {0, 255, 0};
    std::vector<uint> red = {0, 0, 255};
    std::vector<uint> blue = {255, 0, 0};

    std::ifstream ifs(radar_gt_file);
    std::string line;
    std::getline(ifs, line);  // Clear out the header

    int i = 0;
    while (std::getline(ifs, line)) {
        i++;
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        std::vector<double> radar_gt;
        for (uint j = 1; j < parts.size(); ++j) {
            radar_gt.push_back(std::stod(parts[j]));
        }
        double v = sqrt(pow(radar_gt[4], 2) + pow(radar_gt[5], 2));
        if (v < 14.0)
            continue;

        std::string radar_file = parts[0] + ".png";
        int64_t radar_rostime = std::stoll(parts[0]);
        std::cout << "radar ROS time: " << radar_rostime << std::endl;

        Eigen::Matrix4d T_enu_radar = getTransformFromGT(radar_gt);
        Eigen::Matrix3d C_enu_radar = T_enu_radar.block(0, 0, 3, 3);
        Eigen::Vector3d vbar_enu = {radar_gt[4], radar_gt[5], radar_gt[6]};
        Eigen::Vector3d vbar_radar = C_enu_radar.transpose() * vbar_enu;
        vbar_radar(2) = 0;
        Eigen::VectorXd wbar_radar = Eigen::VectorXd::Zero(6, 1);
        wbar_radar(0) = vbar_radar(0);
        wbar_radar(1) = vbar_radar(1);
        wbar_radar(5) = radar_gt[10];

        std::cout << "radar wbar: " << std::endl << wbar_radar.transpose() << std::endl;

        std::vector<int64_t> radar_times;
        std::vector<double> azimuths;
        std::vector<bool> valid;
        cv::Mat f1;
        load_radar(root + "/radar/" + radar_file, radar_times, azimuths, valid, f1, CIR204);
        // Convert radar FFT data into cartesian image
        cv::Mat cart_img;
        radar_polar_to_cartesian(azimuths, f1, radar_resolution, cart_resolution, cart_pixel_width, true, cart_img,
            CV_32F, CIR204);

        // Get LIDAR pointcloud with timestamp closest to this radar scan
        int closest_lidar = get_closest_lidar(radar_rostime, lidar_files);
        std::cout << "closest lidar file: " << lidar_files[closest_lidar] << std::endl;

        std::vector<float> lidar_times;
        Eigen::MatrixXd intensities;
        Eigen::MatrixXd pc;
        load_velodyne3(root + "/lidar/" + lidar_files[closest_lidar], pc, intensities, lidar_times);
        std::vector<double> lidar_gt;
        assert(get_groundtruth_data(lidar_gt_file, lidar_files[closest_lidar], lidar_gt));
        Eigen::Matrix4d T_enu_lidar = getTransformFromGT(lidar_gt);

        // Filter the pointcloud
        uint k = 0;
        for (uint j = 0; j < pc.cols(); ++j) {
            // Remove points close to the ground to make the visualization cleaner.
            if (pc(2, j) < -1.4)
                continue;
            // Remove points close to the sensor to remove self-detections
            if (sqrt(pow(pc(0, j), 2) + pow(pc(1, j), 2)) < 2.0)
                continue;
            pc.block(0, k, 4, 1) = pc.block(0, j, 4, 1);
            lidar_times[k] = lidar_times[j];
            k++;
        }
        pc.conservativeResize(4, k);
        lidar_times.resize(k);

        // Remove motion distortion from lidar data
        removeMotionDistortion(pc, lidar_times, T_enu_lidar, lidar_gt, 0);

        // Transform lidar data into the radar frame
        Eigen::MatrixXd T_radar_lidar = get_inverse_tf(T_enu_radar) * T_enu_lidar;
        pc = T_radar_lidar * pc;
        Eigen::MatrixXd pc2 = Eigen::MatrixXd::Ones(3, pc.cols());
        pc2.block(0, 0, 2, pc.cols()) = pc.block(0, 0, 2, pc.cols());

        // Extract radar extract features
        Eigen::MatrixXd targets, cart_targets;
        cen2018features(f1, zq, sigma_gauss, min_range, targets);
        std::vector<int64_t> t1;
        polar_to_cartesian_points(azimuths, radar_times, targets, radar_resolution, cart_targets, t1);

        cv::Mat vis;
        draw_points(cart_img, pc2, cart_resolution, cart_pixel_width, vis, red);
        draw_points(vis, cart_targets, cart_resolution, cart_pixel_width, green);

        Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(2, 4);
        double size = 2.5;
        cross << -size, size, 0, 0, 0, 0, -size, size;
        std::vector<cv::Point2f> cross_points;
        convert_to_bev(cross, cart_resolution, cart_pixel_width, cross_points);
        cv::line(vis, cross_points[0], cross_points[1], cv::Scalar(255, 255, 255));
        cv::line(vis, cross_points[2], cross_points[3], cv::Scalar(255, 255, 255));

        cv::imshow("radar (green) distorted", vis);

        // Remove Doppler distortion only
        removeDoppler(cart_targets, vbar_radar, beta);

        cv::Mat undistort;
        undistort_radar_image(cart_img, undistort, wbar_radar, cart_resolution, cart_pixel_width,
            radar_times, azimuths, beta, true, false);
        cv::Mat vis2;
        draw_points(undistort, pc2, cart_resolution, cart_pixel_width, vis2, red);
        draw_points(vis2, cart_targets, cart_resolution, cart_pixel_width, green);

        cv::line(vis2, cross_points[0], cross_points[1], cv::Scalar(255, 255, 255));
        cv::line(vis2, cross_points[2], cross_points[3], cv::Scalar(255, 255, 255));

        cv::imshow("radar (green) Doppler distortion removed", vis2);

        // Remove both Doppler and motion distortion from the radar targets
        removeMotionDistortion(cart_targets, t1, wbar_radar, radar_times[radar_times.size() - 1]);

        cv::Mat undistort2;
        undistort_radar_image(cart_img, undistort2, wbar_radar, cart_resolution, cart_pixel_width,
            radar_times, azimuths, beta, true, true);
        cv::Mat vis3;
        draw_points(undistort2, pc2, cart_resolution, cart_pixel_width, vis3, red);
        draw_points(vis3, cart_targets, cart_resolution, cart_pixel_width, green);

        cv::line(vis3, cross_points[0], cross_points[1], cv::Scalar(255, 255, 255));
        cv::line(vis3, cross_points[2], cross_points[3], cv::Scalar(255, 255, 255));

        cv::imshow("radar (green) Doppler and Motion distortion removed", vis3);

        cv::waitKey(0);
    }
}
