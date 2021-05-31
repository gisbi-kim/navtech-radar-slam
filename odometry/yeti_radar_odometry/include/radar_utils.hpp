#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <boost/algorithm/string.hpp>

#define CTS350 0
#define CIR204 1

/*!
   \brief Retrieves a vector of the (radar) file names in ascending order of time stamp
   \param datadir (absolute) path to the directory that contains (radar) files
   \param radar_files [out] A vector to be filled with a string for each file name
   \param extension Optional argument to specify the desired file extension. Files without this extension are rejected
*/
void get_file_names(std::string datadir, std::vector<std::string> &radar_files, std::string extension = "");

/*!
   \brief Decode a single Oxford Radar RobotCar Dataset radar example
   \param path path to the radar image png file
   \param timestamps [out] Timestamp for each azimuth in int64 (UNIX time)
   \param azimuths [out] Rotation for each polar radar azimuth (radians)
   \param valid [out] Mask of whether azimuth data is an original sensor reasing or interpolated from adjacent azimuths
   \param fft_data [out] Radar power readings along each azimuth
*/
void load_radar(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data, int navtech_version = CTS350);

void load_velodyne(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    Eigen::MatrixXd &pc);

void load_velodyne2(std::string path, Eigen::MatrixXd &pc);

void load_velodyne3(std::string path, Eigen::MatrixXd &pc, Eigen::MatrixXd & intensities, std::vector<float> &times);

double get_azimuth_index(std::vector<double> &azimuths, double azimuth);

/*!
   \brief Decode a single Oxford Radar RobotCar Dataset radar example
   \param azimuths Rotation for each polar radar azimuth (radians)
   \param fft_data Radar power readings along each azimuth
   \param radar_resolution Resolution of the polar radar data (metres per pixel)
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width Width and height of the returned square cartesian output (pixels).
   \param interpolate_crossover If true, interpolates between the end and start azimuth of the scan.
   \param cart_img [out] Cartesian radar power readings
*/
void radar_polar_to_cartesian(std::vector<double> &azimuths, cv::Mat &fft_data, float radar_resolution,
    float cart_resolution, int cart_pixel_width, bool interpolate_crossover, cv::Mat &cart_img,
    int output_type = CV_32F, int navtech_version = CTS350);

/*!
   \brief Converts points from polar coordinates to cartesian coordinates
   \param azimuths The actual azimuth of each row in the fft data reported by the Navtech sensor
   \param polar_points Matrix of point locations (azimuth_bin, range_bin) x N
   \param radar_resolution Resolution of the polar radar data (metres per pixel)
   \param cart_points [out] Matrix of points in cartesian space (x, y) x N in metric
*/
void polar_to_cartesian_points(std::vector<double> azimuths, Eigen::MatrixXd polar_points, float radar_resolution,
    Eigen::MatrixXd &cart_points);

void polar_to_cartesian_points(std::vector<double> azimuths, std::vector<int64_t> times, Eigen::MatrixXd polar_points,
    float radar_resolution, Eigen::MatrixXd &cart_points, std::vector<int64_t> &point_times);

/*!
   \brief Converts points from metric cartesian coordinates to pixel coordinates in the BEV image
   \param cart_points Vector of points in metric cartesian space (x, y)
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width: Width and height of the returned square cartesian output (pixels)
   \param bev_points [out] Vector of pixel locations in the BEV cartesian image (u, v)
*/

void convert_to_bev(Eigen::MatrixXd &cart_points, float cart_resolution, int cart_pixel_width,
    std::vector<cv::Point2f> &bev_points);

void convert_to_bev(Eigen::MatrixXd &cart_points, float cart_resolution, int cart_pixel_width, int patch_size,
    std::vector<cv::KeyPoint> &bev_points, std::vector<int64_t> &point_times);

/*!
   \brief Converts points from pixel coordinates in the BEV image to metric cartesian coordinates
   \param bev_points Vector of pixel locations in the BEV cartesian image (u, v)
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width: Width and height of the returned square cartesian output (pixels)
   \param cart_points [out] Vector of points in metric cartesian space (x, y)
*/
void convert_from_bev(std::vector<cv::KeyPoint> bev_points, float cart_resolution, int cart_pixel_width,
    Eigen::MatrixXd &cart_points);

/*!
   \brief Draws a red dot for each feature on the top-down cartesian view of the radar image
   \param cart_img Cartesian radar power readings
   \param cart_targets Matrix of points in cartesian space (x, y) < N in metric
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width Width and height of the square cartesian image.
   \param vis [out] Output image with the features drawn onto it
*/
void draw_points(cv::Mat cart_img, Eigen::MatrixXd cart_targets, float cart_resolution, int cart_pixel_width,
    cv::Mat &vis, std::vector<uint> color = {0, 0, 255});

void draw_points(cv::Mat &vis, Eigen::MatrixXd cart_targets, float cart_resolution, int cart_pixel_width,
    std::vector<uint> color = {0, 0, 255});

/*!
   \brief Retrieves the ground truth odometry between radar timestamps t1 and t2
   \param gtfile (absolute) file location of the radar_odometry.csv file
   \param t1
   \param t2
   \param gt [out] Vector of floats for the ground truth transform between radar timestamp t1 and t2 (x, y, z, r, p, y)
*/
bool get_groundtruth_odometry(std::string gtfile, int64 t1, int64 t2, std::vector<float> &gt);

bool get_groundtruth_odometry2(std::string gtfile, int64_t t, std::vector<double> &gt);

void draw_matches(cv::Mat &img, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2,
    std::vector<cv::DMatch> matches, int radius = 4);

void getTimes(Eigen::MatrixXd cart_targets, std::vector<double> azimuths, std::vector<int64_t> times,
    std::vector<int64_t> &tout);

/*!
   \brief Load arguments from the command line and check their validity.
*/
int validateArgs(const int argc, const char *argv[], std::string &root, std::string &seq, std::string &app);
int validateArgs(const int argc, const char *argv[], std::string &root);
