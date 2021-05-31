#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"

int main(int argc, const char *argv[]) {
    std::string root;
    validateArgs(argc, argv, root);
    std::cout << root << std::endl;
    float cart_resolution = 0.2240;
    int cart_pixel_width = 640;
    bool interpolate_crossover = true;
    float radar_resolution = 0.0560;
    int navtech_version = CIR204;
    std::vector<std::string> radar_files;
    std::string radar_folder = root + "radar/";
    std::string cart_folder = radar_folder + "cart/";
    std::string mask_folder = radar_folder + "mask/";
    omp_set_num_threads(8);
    cv::setNumThreads(8);
    get_file_names(radar_folder, radar_files, "png");
    for (uint idx = 0; idx < radar_files.size(); ++idx) {
        std::cout << idx << " / " << radar_files.size() - 1 << std::endl;
        std::vector<int64_t> timestamps;
        std::vector<double> azimuths;
        std::vector<bool> valid;
        cv::Mat polar;
        load_radar(radar_folder + radar_files[idx], timestamps, azimuths, valid, polar, navtech_version);
        cv::Mat cart;
        radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_resolution, cart_pixel_width,
            interpolate_crossover, cart, CV_32F, navtech_version);
        // mask
        cv::Mat polar_mask = cv::Mat::zeros(polar.rows, polar.cols, CV_32F);
        #pragma omp parallel for
        for (int i = 0; i < polar.rows; ++i) {
            float mean = 0;
            for (int j = 0; j < polar.cols; ++j) {
                mean += polar.at<float>(i, j);
            }
            mean /= polar.cols;
            for (int j = 0; j < polar.cols; ++j) {
                polar_mask.at<float>(i, j) = polar.at<float>(i, j) > (mean * 3.0);
            }
        }
        cv::Mat mask;
        radar_polar_to_cartesian(azimuths, polar_mask, radar_resolution, cart_resolution, cart_pixel_width,
            interpolate_crossover, mask, CV_32F, navtech_version);

        cv::Mat cart_out, mask_out;
        cart.convertTo(cart_out, CV_8UC1, 255.0);
        mask.convertTo(mask_out, CV_8UC1, 255.0);
        cv::imwrite(cart_folder + radar_files[idx], cart_out);
        cv::imwrite(mask_folder + radar_files[idx], mask_out);
    }
    return 0;
}
