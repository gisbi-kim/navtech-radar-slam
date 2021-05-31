#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "matplotlibcpp.h"  // NOLINT
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "association.hpp"
#include <nanoflann.hpp>
namespace plt = matplotlibcpp;
using namespace nanoflann;  // NOLINT

template <typename T>
struct PointCloud {
    struct Point {
        T  x, y, z;
    };
    std::vector<Point>  pts;
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
    if (dim == 0) return pts[idx].x;
    else if (dim == 1) return pts[idx].y;
    else
        return pts[idx].z;
    }
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

template <typename T>
bool contains(std::vector<T> v, T x) {
    for (uint i = 0; i < v.size(); ++i) {
        if (v[i] == x)
            return true;
    }
    return false;
}

void get_lines(std::vector<double> vertices_x, std::vector<double> vertices_y, Eigen::MatrixXd &lines) {
    lines = Eigen::MatrixXd::Zero(7, vertices_x.size());
    for (uint i = 0; i < vertices_x.size() - 1; ++i) {
        double x1 = vertices_x[i];
        double y1 = vertices_y[i];
        double x2 = vertices_x[i + 1];
        double y2 = vertices_y[i + 1];
        double theta = atan2(y2 - y1, x2 - x1);
        bool flag = 0;
        double m = 0, b = 0;
        if ((0 <= theta && theta < M_PI / 4) || (3 * M_PI / 4 <= theta && theta < 5 * M_PI / 4) ||
            (7 * M_PI / 4 <= theta && theta < 2 * M_PI)) {
            m = (y2 - y1) / (x2 - x1);  // y = m*x + b
            b = y1 - m * x1;
        } else {
            m = (x2 - x1) / (y2 - y1);  // x = m*y + b
            b = x1 - m * y1;
            flag = 1;
        }
        lines(0, i) = flag;
        lines(1, i) = m;
        lines(2, i) = b;
        lines(3, i) = x1;
        lines(4, i) = y1;
        lines(5, i) = x2;
        lines(6, i) = y2;
    }
}

typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> my_kd_tree_t;

int main(int argc, char *argv[]) {
    // Generate fake data using a known motion distortion
    double v = 20.0;
    double omega = 90.0 * M_PI / 180;  // rad/s
    if (argc > 1)
        v = atof(argv[1]);
    if (argc > 2)
        omega = atof(argv[2]);
    std::cout << v << " " << omega << std::endl;
    std::vector<double> square_x = {25, -25, -25, 25, 25};
    std::vector<double> square_y = {25, 25, -25, -25, 25};
    std::vector<double> cross_x = {25, 25, -25, -25, -75, -75, -25, -25, 25, 25, 75, 75, 25};
    std::vector<double> cross_y = {25, 75,  75,  25, 25, -25, -25, -75, -75, -25, -25, 25, 25};
    plt::plot(square_x, square_y, "k");

    Eigen::MatrixXd lines;
    get_lines(square_x, square_y, lines);

    std::vector<double> x1, y1, x2, y2;
    std::vector<int64_t> t1, t2;
    std::vector<double> a1, a2;
    double delta_t = 0.000625;
    double time = 0.0;
    Eigen::MatrixXd desc1 = Eigen::MatrixXd::Zero(2, 400);
    Eigen::MatrixXd desc2 = Eigen::MatrixXd::Zero(2, 400);
    std::vector<double> x_pos_vec, y_pos_vec, theta_pos_vec;

    // Simulate the generation of two clouds, motion-distorted:
    for (int scan = 0; scan < 2; scan++) {
        for (int i = 0; i < 400; ++i) {
            // Get sensor position
            double theta_pos = omega * time;
            theta_pos = wrapto2pi(theta_pos);
            double x_pos = 0, y_pos = 0;
            if (omega == 0) {
                x_pos = v * time;
                y_pos = 0;
            } else {
                x_pos = (v / omega) * sin(theta_pos);
                y_pos = (v / omega) * (1 - cos(theta_pos));
            }
            x_pos_vec.push_back(x_pos);
            y_pos_vec.push_back(y_pos);
            theta_pos_vec.push_back(theta_pos);

            double theta_rad = i * 0.9 * M_PI / 180.0;
            double theta = theta_pos + theta_rad;
            theta = wrapto2pi(theta);
            double m = 0;
            double b = 0;
            bool flag = 0;

            if (scan == 0) {
                a1.push_back(theta_rad);
                t1.push_back(time * 1000000);
            } else {
                a2.push_back(theta_rad);
                t2.push_back(time * 1000000);
            }

            if ((0 <= theta && theta < M_PI / 4) || (3 * M_PI / 4 <= theta && theta < 5 * M_PI / 4) ||
                (7 * M_PI / 4 <= theta && theta < 2 * M_PI)) {
                m = tan(theta);  // y = m*x + b
                b = y_pos - m * x_pos;
            } else {
                m = cos(theta) / sin(theta);  // x = m*y + b
                b = x_pos - m * y_pos;
                flag = 1;
            }

            double dmin = 1000000;
            double x_true = 0, y_true = 0;
            for (int j = 0; j < lines.cols(); ++j) {
                double m2 = lines(1, j);
                double b2 = lines(2, j);
                double x_int = 0, y_int = 0;
                if (!flag && lines(0, j) == 0) {
                    x_int = (b2 - b) / (m - m2);
                    y_int = m * x_int + b;
                } else if (!flag && lines(0, j) == 1) {
                    y_int = (m * b2 + b) / (1 - m * m2);
                    x_int = m2 * y_int + b2;
                } else if (flag && lines(0, j) == 0) {
                    y_int = (m2 * b + b2) / (1 - m * m2);
                    x_int = m * y_int + b;
                } else {
                    y_int = (b2 - b) / (m - m2);
                    x_int = m * y_int + b;
                }
                // double theta_test = atan2(y_int - y_pos, x_int - x_pos);
                if ((0 <= theta && theta < M_PI && (y_int - y_pos) < 0) ||
                    (M_PI <= theta && theta < 2 * M_PI && (y_int - y_pos) > 0)) {
                    continue;
                }
                if (( ((0 <= theta && theta < M_PI / 2) || (3 * M_PI / 2 <= theta && theta < 2 * M_PI)) &&
                    (x_int - x_pos) < 0) ||
                    (M_PI / 2 <= theta && theta < 3 * M_PI / 2 && (x_int - x_pos) > 0))
                    continue;
                std::vector<double> x_range = {lines(3, j), lines(5, j)};
                std::sort(x_range.begin(), x_range.end());
                std::vector<double> y_range = {lines(4, j), lines(6, j)};
                std::sort(y_range.begin(), y_range.end());
                if (x_int < x_range[0] || x_int > x_range[1] || y_int < y_range[0] || y_int > y_range[1]) {
                    continue;
                }
                double d = pow(x_pos - x_int, 2) + pow(y_pos - y_int, 2);
                if (d < dmin) {
                    dmin = d;
                    x_true = x_int;
                    y_true = y_int;
                }
            }
            double r = sqrt(pow(x_pos - x_true, 2) + pow(y_pos - y_true, 2));
            if (scan == 0) {
                desc1(0, i) = x_true;
                desc1(1, i) = y_true;
                x1.push_back(r * cos(theta_rad));
                y1.push_back(r * sin(theta_rad));
            } else {
                desc2(0, i) = x_true;
                desc2(1, i) = y_true;
                x2.push_back(r * cos(theta_rad));
                y2.push_back(r * sin(theta_rad));
            }
            time += delta_t;
        }
    }

    std::map<std::string, std::string> kw;
    kw.insert(std::pair<std::string, std::string>("c", "r"));
    plt::scatter(x1, y1, 25.0, kw);
    plt::scatter(x_pos_vec, y_pos_vec, 25.0);
    std::map<std::string, std::string> kw2;
    kw2.insert(std::pair<std::string, std::string>("c", "b"));
    plt::scatter(x2, y2, 25.0, kw2);

    // Perform NN matching using the descriptors from each cloud:
    PointCloud<double> cloud2;
    cloud2.pts.resize(desc2.cols());
    for (uint i = 0; i < desc2.cols(); ++i) {
        cloud2.pts[i].x = desc2(0, i);
        cloud2.pts[i].y = desc2(1, i);
        cloud2.pts[i].z = 0;
    }
    my_kd_tree_t index2(2, cloud2, KDTreeSingleIndexAdaptorParams(1));
    index2.buildIndex();
    std::vector<int> matches;
    size_t num_results = 1;
    std::vector<size_t>   ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);

    int size = 0;
    for (uint i = 0; i < desc1.cols(); ++i) {
        double query_pt[3] = {desc1(0, i), desc1(1, i), 0};
        index2.knnSearch(&query_pt[0], 1, &ret_index[0], &out_dist_sqr[0]);
        int idx = int(ret_index[0]);
        if (!contains(matches, idx)) {
            matches.push_back(idx);
            size++;
        } else {
            matches.push_back(-1);
        }
    }

    // Create the p1 and p2 matrices based on the matches:
    Eigen::MatrixXd p1, p2;
    p1 = Eigen::MatrixXd::Zero(2, size);
    p2 = p1;
    int j = 0;
    std::vector<int64_t> t1prime = t1, t2prime = t2;
    for (uint i = 0; i < desc1.cols(); ++i) {
        if (matches[i] == -1)
            continue;
        p1(0, j) = x1[i];
        p1(1, j) = y1[i];
        p2(0, j) = x2[matches[i]];
        p2(1, j) = y2[matches[i]];
        t1prime[j] = t1[i];
        t2prime[j] = t2[matches[i]];
        j++;
    }
    t1prime.resize(size);
    t2prime.resize(size);
    // run the rigid RANSAC algo for comparison
    Ransac ransac(p2, p1, 0.35, 0.90, 100);
    ransac.computeModel();
    Eigen::MatrixXd T;
    ransac.getTransform(T);
    Eigen::MatrixXd T2 = Eigen::MatrixXd::Identity(4, 4);
    T2.block(0, 0, 2, 2) = T.block(0, 0, 2, 2);
    T2.block(0, 3, 2, 1) = T.block(0, 2, 2, 1);
    std::cout << "(RIGID) T: " << std::endl << T2 << std::endl;
    Eigen::VectorXd xi = SE3tose3(T2) * 4;
    std::cout << "(RIGID) wbar: " << xi << std::endl;
    Eigen::MatrixXd p2prime = Eigen::MatrixXd::Ones(3, p2.cols());
    p2prime.block(0, 0, 2, p2.cols()) = p2;
    p2prime = T * p2prime;
    std::vector<double> x3, y3;
    for (uint i = 0; i < p2.cols(); ++i) {
        x3.push_back(p2prime(0, i));
        y3.push_back(p2prime(1, i));
    }
    std::map<std::string, std::string> kw3;
    kw3.insert(std::pair<std::string, std::string>("c", "g"));
    plt::scatter(x3, y3, 25.0, kw3);
    plt::show();

    // run the motion-distorted RANSAC to extract the motion parameters:
    MotionDistortedRansac mdransac(p2, p1, t2prime, t1prime, 0.35, 0.90, 100);
    mdransac.computeModel();
    Eigen::VectorXd w;
    mdransac.getMotion(w);
    Eigen::MatrixXd Tmd;
    Tmd = se3ToSE3(w / 4);
    std::cout << "(MD) T: " << std::endl << Tmd.inverse() << std::endl;
    std::cout << "(MD) wbar: " << std::endl << w * -1 << std::endl;

    // Test removing the motion distortion
    std::vector<double> x4, y4;
    for (uint i = 1; i < t1prime.size(); ++i) {
        double delta_t = (t1prime[i] - t1prime[0])/1000000.0;
        Eigen::MatrixXd T = se3ToSE3(w * delta_t);
        Eigen::Vector4d p1bar = {p1(0, i), p1(1, i), 0, 1};
        p1bar = T.inverse() * p1bar;
        x4.push_back(p1bar(0));
        y4.push_back(p1bar(1));
    }

    plt::scatter(x4, y4, 25.0);
    plt::show();

    return 0;
}
