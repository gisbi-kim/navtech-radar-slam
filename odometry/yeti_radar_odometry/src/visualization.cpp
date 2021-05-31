#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"
#include "features.hpp"

int main() {
    std::string datadir = "/media/keenan/autorontossd1/2020_11_10/radar";
    float cart_resolution = 0.2592;
    int cart_pixel_width = 772;
    bool interpolate_crossover = true;

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files, "png");
    std::cout << radar_files[0] << std::endl;

    float radar_resolution = 0.0432;
    std::vector<int64_t> timestamps;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    load_radar(datadir + "/" + radar_files[0], timestamps, azimuths, valid, fft_data);

    Eigen::MatrixXd targets;
    int min_range = 58;

    // int max_points = 10000;
    // cen2019features(fft_data, max_points, min_range, targets);

    float zq = 2.5;
    int sigma_gauss = 17;
    cen2018features(fft_data, zq, sigma_gauss, min_range, targets);

    cv::Mat cart_img;
    radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
        interpolate_crossover, cart_img);

    Eigen::MatrixXd cart_targets;
    polar_to_cartesian_points(azimuths, targets, radar_resolution, cart_targets);
    cv::Mat vis;
    // draw_points(cart_img, cart_targets, cart_resolution, cart_pixel_width, vis);
    cv::imshow("cart", cart_img);
    cv::waitKey(0);

    return 0;
}
