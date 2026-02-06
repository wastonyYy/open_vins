/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_GRIDER_GRID_H
#define OV_CORE_GRIDER_GRID_H

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utils/opencv_lambda_body.h"

namespace ov_core {

/**
 * @brief Extracts FAST features in a grid pattern.
 *
 * As compared to just extracting fast features over the entire image,
 * we want to have as uniform of extractions as possible over the image plane.
 * Thus we split the image into a bunch of small grids, and extract points in each.
 * We then pick enough top points in each grid so that we have the total number of desired points.
 */
class Grider_GRID {

public:
  /**
   * @brief Compare keypoints based on their response value.
   * @param first First keypoint
   * @param second Second keypoint
   *
   * We want to have the keypoints with the highest values!
   * See: https://stackoverflow.com/a/10910921
   */
  static bool compare_response(cv::KeyPoint first, cv::KeyPoint second) { return first.response > second.response; }

  /**
   * @brief This function will perform grid extraction using FAST.
   * @param img Image we will do FAST extraction on
   * @param mask Region of the image we do not want to extract features in (255 = do not detect features)
   * @param valid_locs Valid 2d grid locations we will extract in (instead of the whole image)
   * @param pts vector of extracted points we will return
   * @param num_features max number of features we want to extract
   * @param grid_x size of grid in the x-direction / u-direction
   * @param grid_y size of grid in the y-direction / v-direction
   * @param threshold FAST threshold paramter (10 is a good value normally)
   * @param nonmaxSuppression if FAST should perform non-max suppression (true normally)
   *
   * Given a specified grid size, this will try to extract fast features from each grid.
   * It will then return the best from each grid in the return vector.
   */
  #if 0
  // static void perform_griding(const cv::Mat &img, const cv::Mat &mask, const std::vector<std::pair<int, int>> &valid_locs,
  //                             std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold,
  //                             bool nonmaxSuppression) {

  //   // Return if there is nothing to extract
  //   if (valid_locs.empty())
  //     return;

  //   // We want to have equally distributed features
  //   // NOTE: If we have more grids than number of total points, we calc the biggest grid we can do
  //   // NOTE: Thus if we extract 1 point per grid we have
  //   // NOTE:    -> 1 = num_features / (grid_x * grid_y)
  //   // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
  //   // NOTE:    -> grid_y = sqrt(num_features / ratio)
  //   if (num_features < grid_x * grid_y) {
  //     double ratio = (double)grid_x / (double)grid_y;
  //     grid_y = std::ceil(std::sqrt(num_features / ratio));
  //     grid_x = std::ceil(grid_y * ratio);
  //   }
  //   int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  //   assert(grid_x > 0);
  //   assert(grid_y > 0);
  //   assert(num_features_grid > 0);

  //   // Calculate the size our extraction boxes should be
  //   int size_x = img.cols / grid_x;
  //   int size_y = img.rows / grid_y;

  //   // Make sure our sizes are not zero
  //   assert(size_x > 0);
  //   assert(size_y > 0);

  //   // Parallelize our 2d grid extraction!!
  //   std::vector<std::vector<cv::KeyPoint>> collection(valid_locs.size());
  //   parallel_for_(cv::Range(0, (int)valid_locs.size()), LambdaBody([&](const cv::Range &range) {
  //                   for (int r = range.start; r < range.end; r++) {

  //                     // Calculate what cell xy value we are in
  //                     auto grid = valid_locs.at(r);
  //                     int x = grid.first * size_x;
  //                     int y = grid.second * size_y;

  //                     // Skip if we are out of bounds
  //                     if (x + size_x > img.cols || y + size_y > img.rows)
  //                       continue;

  //                     // Calculate where we should be extracting from
  //                     cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);

  //                     // Extract FAST features for this part of the image
  //                     std::vector<cv::KeyPoint> pts_new;
  //                     cv::FAST(img(img_roi), pts_new, threshold, nonmaxSuppression);

  //                     // Now lets get the top number from this
  //                     std::sort(pts_new.begin(), pts_new.end(), Grider_FAST::compare_response);

  //                     // Append the "best" ones to our vector
  //                     // Note that we need to "correct" the point u,v since we extracted it in a ROI
  //                     // So we should append the location of that ROI in the image
  //                     for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++) {

  //                       // Create keypoint
  //                       cv::KeyPoint pt_cor = pts_new.at(i);
  //                       pt_cor.pt.x += (float)x;
  //                       pt_cor.pt.y += (float)y;

  //                       // Reject if out of bounds (shouldn't be possible...)
  //                       if ((int)pt_cor.pt.x < 0 || (int)pt_cor.pt.x > img.cols || (int)pt_cor.pt.y < 0 || (int)pt_cor.pt.y > img.rows)
  //                         continue;

  //                       // Check if it is in the mask region
  //                       // NOTE: mask has max value of 255 (white) if it should be removed
  //                       if (mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127)
  //                         continue;
  //                       collection.at(r).push_back(pt_cor);
  //                     }
  //                   }
  //                 }));

  //   // Combine all the collections into our single vector
  //   for (size_t r = 0; r < collection.size(); r++) {
  //     pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
  //   }

  //   // Return if no points
  //   if (pts.empty())
  //     return;

  //   // Sub-pixel refinement parameters
  //   cv::Size win_size = cv::Size(5, 5);
  //   cv::Size zero_zone = cv::Size(-1, -1);
  //   cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001);

  //   // Get vector of points
  //   std::vector<cv::Point2f> pts_refined;
  //   for (size_t i = 0; i < pts.size(); i++) {
  //     pts_refined.push_back(pts.at(i).pt);
  //   }

  //   // Finally get sub-pixel for all extracted features
  //   cv::cornerSubPix(img, pts_refined, win_size, zero_zone, term_crit);

  //   // Save the refined points!
  //   for (size_t i = 0; i < pts.size(); i++) {
  //     pts.at(i).pt = pts_refined.at(i);
  //   }
  // }
  #endif
// [辅助函数] 计算局部方差，用于剔除热成像的低对比度噪声
// 建议放在 Grider_GRID 类内部或作为 static 函数
static double calcNeighborHistVar(const cv::Point2f& pt, const cv::Mat& img) {
    int r = 2; // 5x5 窗口
    int x = std::round(pt.x);
    int y = std::round(pt.y);
    
    // 边界检查
    if (x - r < 0 || x + r >= img.cols || y - r < 0 || y + r >= img.rows) return 0.0;
    
    // 提取 ROI 并计算均值和标准差
    cv::Mat patch = img(cv::Rect(x - r, y - r, 2 * r + 1, 2 * r + 1));
    cv::Scalar mean, stddev;
    cv::meanStdDev(patch, mean, stddev);
    
    // 返回方差 (stddev^2)
    return stddev[0] * stddev[0];
}
#if 1
static void perform_griding(const cv::Mat &img, const cv::Mat &mask, const std::vector<std::pair<int, int>> &valid_locs,
                            std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold,
                            bool nonmaxSuppression) {

    // === [TIMER START] ===
    double t_start = (double)cv::getTickCount();

    if (valid_locs.empty()) return;

    if (num_features < grid_x * grid_y) {
      double ratio = (double)grid_x / (double)grid_y;
      grid_y = std::ceil(std::sqrt(num_features / ratio));
      grid_x = std::ceil(grid_y * ratio);
    }
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    num_features_grid = std::max(1, num_features_grid);

    int size_x = img.cols / grid_x;
    int size_y = img.rows / grid_y;

    std::vector<std::vector<cv::KeyPoint>> collection(valid_locs.size());

    // 用于统计总的原始候选点数量，评估 GFTT 的初筛能力
    std::atomic<int> total_raw_candidates(0);
    std::atomic<double> total_variance_sum(0.0);

    parallel_for_(cv::Range(0, (int)valid_locs.size()), LambdaBody([&](const cv::Range &range) {
        for (int r = range.start; r < range.end; r++) {

            auto grid = valid_locs.at(r);
            int x = grid.first * size_x;
            int y = grid.second * size_y;

            if (x + size_x > img.cols || y + size_y > img.rows) continue;

            cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);
            std::vector<cv::Point2f> pts_raw;
            int maxCorners = num_features_grid * 3 + 5; 
            double qualityLevel = 0.01; 
            double minDistance = 5.0;   
            
            cv::goodFeaturesToTrack(img(img_roi), pts_raw, maxCorners, qualityLevel, minDistance, cv::Mat(), 3);

            std::vector<cv::KeyPoint> pts_valid;
            double var_threshold = 15.0; 
            
            for (auto& pt : pts_raw) {
                cv::Point2f pt_global = pt;
                pt_global.x += x;
                pt_global.y += y;

                if (mask.at<uint8_t>((int)pt_global.y, (int)pt_global.x) > 127) continue;

                double var = calcNeighborHistVar(pt_global, img);
                // 统计所有合格点的方差总和
                if (var > var_threshold) {
                    cv::KeyPoint kp;
                    kp.pt = pt_global;
                    kp.response = (float)var; 
                    pts_valid.push_back(kp);
                    
                    // 原子操作累加，会有轻微性能损耗但为了统计值得
                    // 如果太慢可以去掉
                    // total_variance_sum = total_variance_sum + var; 
                }
            }
            collection.at(r) = pts_valid;
            total_raw_candidates += pts_valid.size();
        }
    }));

    std::vector<int> need_supplement_ids;
    std::vector<int> have_surplus_ids;
    
    for (size_t r = 0; r < collection.size(); r++) {
        if (collection[r].size() < (size_t)num_features_grid) {
            need_supplement_ids.push_back(r);
        } else {
            have_surplus_ids.push_back(r);
        }
    }

    // Step A
    for (int id : need_supplement_ids) {
        pts.insert(pts.end(), collection[id].begin(), collection[id].end());
    }

    // Step B
    std::vector<cv::KeyPoint> surplus_pool;
    for (int id : have_surplus_ids) {
        auto& p_vec = collection[id];
        pts.insert(pts.end(), p_vec.begin(), p_vec.begin() + num_features_grid);
        if (p_vec.size() > (size_t)num_features_grid) {
            surplus_pool.insert(surplus_pool.end(), p_vec.begin() + num_features_grid, p_vec.end());
        }
    }

    // Step C
    int current_count = pts.size();
    int deficit = num_features - current_count;
    int supplemented_count = 0; // 记录实际补充了多少点

    if (deficit > 0 && !surplus_pool.empty()) {
        int take_count = std::min(deficit, (int)surplus_pool.size());
        // 按照响应值(方差)排序，优先补充高质量点
        std::sort(surplus_pool.begin(), surplus_pool.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b){
             return a.response > b.response;
        });
        pts.insert(pts.end(), surplus_pool.begin(), surplus_pool.begin() + take_count);
        supplemented_count = take_count;
    }

    if (pts.empty()) return;

    // 亚像素
    cv::Size win_size = cv::Size(5, 5);
    cv::Size zero_zone = cv::Size(-1, -1);
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001);

    std::vector<cv::Point2f> pts_refined;
    for (const auto& kp : pts) pts_refined.push_back(kp.pt);
    if (!img.empty()) cv::cornerSubPix(img, pts_refined, win_size, zero_zone, term_crit);
    for (size_t i = 0; i < pts.size(); i++) pts.at(i).pt = pts_refined.at(i);

    // === [METRICS REPORT] ===
    double t_cost = ((double)cv::getTickCount() - t_start) / cv::getTickFrequency() * 1000.0;
    
    // 计算指标
    double fill_rate = (double)pts.size() / (double)num_features * 100.0; // 填充率
    double poor_grid_ratio = (double)need_supplement_ids.size() / (double)valid_locs.size() * 100.0; // 贫穷网格占比
    
    // 打印量化报告 (建议使用醒目的颜色)
    // 限制打印频率：你可以加一个 static int counter 或者是每帧都打
    // 这里每帧打印，调试完记得注释掉
    PRINT_ERROR(CYAN  "[GRID-STATS] Time: %.2f ms | Target: %d | Got: %zu (%.1f%%)\n" RESET, 
        t_cost, num_features, pts.size(), fill_rate);
        
    if (fill_rate < 80.0) {
        PRINT_ERROR(RED   "   [WARNING] Low Fill Rate! (Try lowering var_threshold or qualityLevel)\n" RESET);
    }

    PRINT_ERROR(YELLOW "   [UNIFORMITY] Poor Grids: %zu/%zu (%.1f%%) | Supplemented: %d\n" RESET,
        need_supplement_ids.size(), valid_locs.size(), poor_grid_ratio, supplemented_count);
        
    // 如果贫穷网格太多，说明特征太集中
    if (poor_grid_ratio > 50.0) {
        PRINT_ERROR(RED   "   [WARNING] High Concentration! Features are bunched up.\n" RESET);
    }
}
#endif
};

} // namespace ov_core

#endif /* OV_CORE_GRIDER_GRID_H */
