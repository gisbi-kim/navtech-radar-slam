#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <features.hpp>

void cfar1d(cv::Mat fft_data, int window_size, float scale, int guard_cells, int min_range, Eigen::MatrixXd &targets) {
    assert(fft_data.depth() == CV_32F);
    assert(fft_data.channels() == 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    int kernel_size = window_size + guard_cells * 2 + 1;
    cv::Mat kernel = cv::Mat::ones(1, kernel_size, CV_32F) * -1 * scale / window_size;
    kernel.at<float>(0, kernel_size / 2) = 1;
    for (int i = 0; i < guard_cells; i++) {
        kernel.at<float>(0, window_size / 2 + i) = 0;
    }
    for (int i = 0; i < guard_cells; i++) {
        kernel.at<float>(0, kernel_size / 2 + 1 + i) = 0;
    }
    cv::Mat output;
    cv::filter2D(fft_data, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);
    // Find filter responses > 0
    std::vector<cv::Point2f> t;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = min_range; j < output.cols; j++) {
            if (output.at<float>(i, j) > 0) {
                t.push_back(cv::Point(i, j));
            }
        }
    }
    targets = Eigen::MatrixXd::Ones(3, t.size());
    for (uint i = 0; i < t.size(); ++i) {
        targets(0, i) = t[i].x;
        targets(1, i) = t[i].y;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    std::cout << "feature extraction: " << e.count() << std::endl;
}

// Runtime: 0.035s
double cen2018features(cv::Mat fft_data, float zq, int sigma_gauss, int min_range, Eigen::MatrixXd &targets) {
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> sigma_q(fft_data.rows, 0);
    // Estimate the bias and subtract it from the signal
    cv::Mat q = fft_data.clone();
    for (int i = 0; i < fft_data.rows; ++i) {
        float mean = 0;
        for (int j = 0; j < fft_data.cols; ++j) {
            mean += fft_data.at<float>(i, j);
        }
        mean /= fft_data.cols;
        for (int j = 0; j < fft_data.cols; ++j) {
            q.at<float>(i, j) = fft_data.at<float>(i, j) - mean;
        }
    }

    // Create 1D Gaussian Filter (0.09)
    assert(sigma_gauss % 2 == 1);
    int fsize = sigma_gauss * 3;
    int mu = fsize / 2;
    float sig_sqr = sigma_gauss * sigma_gauss;
    cv::Mat filter = cv::Mat::zeros(1, fsize, CV_32F);
    float s = 0;
    for (int i = 0; i < fsize; ++i) {
        filter.at<float>(0, i) = exp(-0.5 * (i - mu) * (i - mu) / sig_sqr);
        s += filter.at<float>(0, i);
    }
    filter /= s;
    cv::Mat p;
    cv::filter2D(q, p, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

    // Estimate variance of noise at each azimuth (0.004)
    for (int i = 0; i < fft_data.rows; ++i) {
        int nonzero = 0;
        for (int j = 0; j < fft_data.cols; ++j) {
            float n = q.at<float>(i, j);
            if (n < 0) {
                sigma_q[i] += 2 * (n * n);
                nonzero++;
            }
        }
        if (nonzero)
            sigma_q[i] = sqrt(sigma_q[i] / nonzero);
        else
            sigma_q[i] = 0.034;
    }

    // Extract peak centers from each azimuth
    std::vector<std::vector<cv::Point2f>> t(fft_data.rows);
#pragma omp parallel for
    for (int i = 0; i < fft_data.rows; ++i) {
        std::vector<int> peak_points;
        float thres = zq * sigma_q[i];
        for (int j = min_range; j < fft_data.cols; ++j) {
            float nqp = exp(-0.5 * pow((q.at<float>(i, j) - p.at<float>(i, j)) / sigma_q[i], 2));
            float npp = exp(-0.5 * pow(p.at<float>(i, j) / sigma_q[i], 2));
            float b = nqp - npp;
            float y = q.at<float>(i, j) * (1 - nqp) + p.at<float>(i, j) * b;
            if (y > thres) {
                peak_points.push_back(j);
            } else if (peak_points.size() > 0) {
                t[i].push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
                peak_points.clear();
            }
        }
        if (peak_points.size() > 0)
            t[i].push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
    }

    int size = 0;
    for (uint i = 0; i < t.size(); ++i) {
        size += t[i].size();
    }
    targets = Eigen::MatrixXd::Ones(3, size);
    int k = 0;
    for (uint i = 0; i < t.size(); ++i) {
        for (uint j = 0; j < t[i].size(); ++j) {
            targets(0, k) = t[i][j].x;
            targets(1, k) = t[i][j].y;
            k++;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    return e.count();
}

struct Point {
    float i;
    int a;
    int r;
    Point(float i_, int a_, int r_) {i = i_; a = a_; r = r_;}
};

struct greater_than_pt {
    inline bool operator() (const Point& p1, const Point& p2) {
        return p1.i > p2.i;
    }
};

static void findRangeBoundaries(cv::Mat &s, int a, int r, int &rlow, int &rhigh) {
    rlow = r;
    rhigh = r;
    if (r > 0) {
        for (int i = r - 1; i >= 0; i--) {
            if (s.at<float>(a, i) < 0)
                rlow = i;
            else
                break;
        }
    }
    if (r < s.rows - 1) {
        for (int i = r + 1; i < s.cols; i++) {
            if (s.at<float>(a, i) < 0)
                rhigh = i;
            else
                break;
        }
    }
}

static bool checkAdjacentMarked(cv::Mat &R, int a, int start, int end) {
    int below = a - 1;
    int above = a + 1;
    if (below < 0)
        below = R.rows - 1;
    if (above >= R.rows)
        above = 0;
    for (int r = start; r <= end; r++) {
        if (R.at<float>(below, r) || R.at<float>(above, r))
            return true;
    }
    return false;
}

static void getMaxInRegion(cv::Mat &h, int a, int start, int end, int &max_r) {
    int max = -1000;
    for (int r = start; r <= end; r++) {
        if (h.at<float>(a, r) > max) {
            max = h.at<float>(a, r);
            max_r = r;
        }
    }
}

// Runtime: 0.050s
double cen2019features(cv::Mat fft_data, int max_points, int min_range, Eigen::MatrixXd &targets) {
    auto t1 = std::chrono::high_resolution_clock::now();
    // Calculate gradient along each azimuth using the Prewitt operator
    cv::Mat prewitt = cv::Mat::zeros(1, 3, CV_32F);
    prewitt.at<float>(0, 0) = -1;
    prewitt.at<float>(0, 2) = 1;
    cv::Mat g;
    cv::filter2D(fft_data, g, -1, prewitt, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);
    g = cv::abs(g);
    double maxg = 1, ming = 1;
    cv::minMaxIdx(g, &ming, &maxg);
    g /= maxg;

    // Subtract the mean from the radar data and scale it by 1 - gradient magnitude
    float mean = cv::mean(fft_data)[0];
    cv::Mat s = fft_data - mean;
    cv::Mat h = s.mul(1 - g);
    float mean_h = cv::mean(h)[0];

    // Get indices in descending order of intensity
    std::vector<Point> vec;
    for (int i = 0; i < fft_data.rows; ++i) {
        for (int j = 0; j < fft_data.cols; ++j) {
            if (h.at<float>(i, j) > mean_h)
                vec.push_back(Point(h.at<float>(i, j), i, j));
        }
    }
    std::sort(vec.begin(), vec.end(), greater_than_pt());

    // Create a matrix, R, of "marked" regions consisting of continuous regions of an azimuth that may contain a target
    int false_count = fft_data.rows * fft_data.cols;
    uint j = 0;
    int l = 0;
    cv::Mat R = cv::Mat::zeros(fft_data.rows, fft_data.cols, CV_32F);
    while (l < max_points && j < vec.size() && false_count > 0) {
        if (!R.at<float>(vec[j].a, vec[j].r)) {
            int rlow = vec[j].r;
            int rhigh = vec[j].r;
            findRangeBoundaries(s, vec[j].a, vec[j].r, rlow, rhigh);
            bool already_marked = false;
            for (int i = rlow; i <= rhigh; i++) {
                if (R.at<float>(vec[j].a, i)) {
                    already_marked = true;
                    continue;
                }
                R.at<float>(vec[j].a, i) = 1;
                false_count--;
            }
            if (!already_marked)
                l++;
        }
        j++;
    }

    std::vector<std::vector<cv::Point2f>> t(fft_data.rows);

#pragma omp parallel for
    for (int i = 0; i < fft_data.rows; i++) {
        // Find the continuous marked regions in each azimuth
        int start = 0;
        int end = 0;
        bool counting = false;
        for (int j = min_range; j < fft_data.cols; j++) {
            if (R.at<float>(i, j)) {
                if (!counting) {
                    start = j;
                    end = j;
                    counting = true;
                } else {
                    end = j;
                }
            } else if (counting) {
                // Check whether adjacent azimuths contain a marked pixel in this range region
                if (checkAdjacentMarked(R, i, start, end)) {
                    int max_r = start;
                    getMaxInRegion(h, i, start, end, max_r);
                    t[i].push_back(cv::Point(i, max_r));
                }
                counting = false;
            }
        }
    }

    int size = 0;
    for (uint i = 0; i < t.size(); ++i) {
        size += t[i].size();
    }
    targets = Eigen::MatrixXd::Ones(3, size);
    int k = 0;
    for (uint i = 0; i < t.size(); ++i) {
        for (uint j = 0; j < t[i].size(); ++j) {
            targets(0, k) = t[i][j].x;
            targets(1, k) = t[i][j].y;
            k++;
        }
    }

    // std::cout << "feature size is " << size << std::endl;

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    return e.count();
}

// Note: the dimensions of polar_points and cart_targets may not align
// Runtime: 100 ms
double cen2019descriptors(std::vector<double> azimuths, cv::Size polar_dims, Eigen::MatrixXd polar_points,
    Eigen::MatrixXd cart_targets, float radar_resolution, float cart_resolution, int cart_pixel_width,
    cv::Mat &descriptors, int navtech_version) {

    auto t1 = std::chrono::high_resolution_clock::now();
    // Create binary grid based on polar feature locations
    cv::Mat polar_binary = cv::Mat::zeros(polar_dims.height, polar_dims.width, CV_32F);
#pragma omp parallel for
    for (uint i = 0; i < polar_points.cols(); ++i) {
        polar_binary.at<float>(polar_points(0, i), polar_points(1, i)) = 1.0;
    }
    // Convert it to cartesian
    cv::Mat cart_binary;
    radar_polar_to_cartesian(azimuths, polar_binary, radar_resolution, cart_resolution, cart_pixel_width, true,
        cart_binary, CV_32F, navtech_version);

    // int M = polar_dims.height;
    int M = 384;
    float azimuth_step = (2 * M_PI) / float(M);
    int N = 128;
    float range_step = (cart_pixel_width / 2.0) / float(N);
    float max_range_sq = pow(N * range_step, 2);

    cv::Mat d1 = cv::Mat::zeros(cart_targets.cols(), M, CV_32F);
    cv::Mat d2 = cv::Mat::zeros(cart_targets.cols(), N, CV_32F);

    std::vector<cv::Point2f> bev_points;
    convert_to_bev(cart_targets, cart_resolution, cart_pixel_width, bev_points);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < cart_binary.rows; ++i) {
        for (int j = 0; j < cart_binary.cols; ++j) {
            if (cart_binary.at<float>(i, j) > 0) {
                for (uint k = 0; k < bev_points.size(); ++k) {
                    float range = pow(i - bev_points[k].y, 2) + pow(j - bev_points[k].x, 2);
                    if (range > max_range_sq)
                        continue;
                    range = sqrt(range);
                    float azimuth = atan2f(bev_points[k].y - i, j - bev_points[k].x);
                    if (azimuth < 0)
                        azimuth += 2 * M_PI;
                    int azimuth_bin = azimuth / azimuth_step;
                    int range_bin = range / range_step;
#pragma omp atomic
                    d1.at<float>(k, azimuth_bin)++;
#pragma omp atomic
                    d2.at<float>(k, range_bin)++;
                }
            }
        }
    }

    // Calculate the FFT for each azimuth, normalize the magnitude
#pragma omp parallel for
    for (uint i = 0; i < cart_targets.cols(); ++i) {
        cv::Mat row = cv::Mat::zeros(1, M, CV_32F);
        for (int j = 0; j < M; ++j) {
            row.at<float>(0, j) = d1.at<float>(i, j);
        }
        cv::Mat planes[] = {cv::Mat_<float>(row), cv::Mat::zeros(row.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
        cv::dft(complexI, complexI);            // this way the result may fit in the source matrix
        cv::split(complexI, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);
        cv::Mat magI = planes[0];
        cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
        for (int j = 0; j < M; ++j) {
            d1.at<float>(i, j) = magI.at<float>(0, j);
        }
    }

// #pragma omp parallel for
//     // Reorder with the densest column first
//     for (uint i = 0; i < cart_targets.cols(); ++i) {
//         float max = 0;
//         int max_col = 0;
//         for (int j = 0; j < M; ++j) {
//             if (d1.at<float>(i, j) > max) {
//                 max = d1.at<float>(i, j);
//                 max_col = j;
//             }
//         }
//         cv::Mat row = cv::Mat::zeros(1, M, CV_32F);
//         int k = 0;
//         for (int j = max_col; j < M; ++j) {
//             row.at<float>(0, k) = d1.at<float>(i, j);
//             k++;
//         }
//         for (int j = 0; j < max_col; ++j) {
//             row.at<float>(0, k) = d1.at<float>(i, j);
//             k++;
//         }
//         cv::normalize(row, row, 0, 1, cv::NORM_MINMAX);
//         for (int j = 0; j < M; ++j) {
//             d1.at<float>(i, j) = row.at<float>(0, j);
//         }
//     }

    // Normalize the counts for each range bin
#pragma omp parallel for
    for (uint i = 0; i < cart_targets.cols(); ++i) {
        cv::normalize(d2.row(i), d2.row(i), 0, 1, cv::NORM_MINMAX);
    }
    // cv::hconcat(d1, d2, descriptors);
    descriptors = d2;

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    return e.count();
}
