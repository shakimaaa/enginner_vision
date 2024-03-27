#include "pnp_solver.hpp"

namespace rm_auto_box
{
    PnPSolver::PnPSolver(
        const std::array<double, 9> &camera_matrix, const std::vector<double> &dist_coeffs)
        : camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
          dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
    {
        // Unit: m
        constexpr double box_half_y = ENERGY_LEAF_WIDTH / 2.0 / 1000.0;
        constexpr double box_half_z = ENERGY_LEAF_HEIGHT / 2.0 / 1000.0;
        box_points_.emplace_back(cv::Point3f(0, box_half_y, -box_half_z));//BOTTOM_RIGHT
        box_points_.emplace_back(cv::Point3f(0, box_half_y, box_half_z));//TOP_RIGHT
        box_points_.emplace_back(cv::Point3f(0, -box_half_y, box_half_z));//TOP_LEFT
        box_points_.emplace_back(cv::Point3f(0, -box_half_y, -box_half_z));//BOTTOM_LEFT
    }
    float PnPSolver::calculateDistanceToCenter(const cv::Point2f &image_point)
    {
        float cx = camera_matrix_.at<double>(0, 2);
        float cy = camera_matrix_.at<double>(1, 2);
        return cv::norm(image_point - cv::Point2f(cx, cy));
    }
    bool PnPSolver::solvePnP_(const Box &box, cv::Mat &rvec, cv::Mat &tvec)
    {
        std::vector<cv::Point2f> image_box_points;

        // Fill in image points
        image_box_points.emplace_back(box.kpt[POINT_0]);
        image_box_points.emplace_back(box.kpt[POINT_1]);
        image_box_points.emplace_back(box.kpt[POINT_2]);
        image_box_points.emplace_back(box.kpt[POINT_3]);
        return cv::solvePnP(box_points_, image_box_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    }
}