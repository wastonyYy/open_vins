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

#include "Landmark.h"

using namespace ov_type;

Eigen::Matrix<double, 3, 1> Landmark::get_xyz(bool getfej) const {
  // 点是直接以三维坐标（X, Y, Z）来表示。
  // 输出方式：直接返回 value() 或 fej()，就是三维坐标本身
  // CASE: Global 3d feature representation
  // CASE: Anchored 3D feature representation
  if (_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D ||
      _feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
    return (getfej) ? fej() : value();
  }
  // 点是用一个 方向（两个角）+ 逆深度 表示的。
  // p_invFinG = (theta, phi, rho)，分别是方位角、俯仰角、逆深度。
  // 转换方式：把极坐标 + 逆深度转为 3D 坐标。
  // CASE: Global inverse depth feature representation
  // CASE: Anchored full inverse depth feature representation
  if (_feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH ||
      _feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {
    Eigen::Matrix<double, 3, 1> p_invFinG = (getfej) ? fej() : value();
    Eigen::Matrix<double, 3, 1> p_FinG;
    p_FinG << (1 / p_invFinG(2)) * std::cos(p_invFinG(0)) * std::sin(p_invFinG(1)),
        (1 / p_invFinG(2)) * std::sin(p_invFinG(0)) * std::sin(p_invFinG(1)), (1 / p_invFinG(2)) * std::cos(p_invFinG(1));
    return p_FinG;
  }
  // MSCKF 使用的逆深度模型，表示为 (α, β, ρ)
  // CASE: Anchored MSCKF inverse depth feature representation
  if (_feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {
    Eigen::Matrix<double, 3, 1> p_FinA;
    Eigen::Matrix<double, 3, 1> p_invFinA = value();
    p_FinA << (1 / p_invFinA(2)) * p_invFinA(0), (1 / p_invFinA(2)) * p_invFinA(1), 1 / p_invFinA(2);
    return p_FinA;
  }

  // 方向是固定已知的（单位向量 uv_norm_zero），只优化深度。
  // CASE: Estimate single depth of the feature using the initial bearing
  if (_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
    // if(getfej) return 1.0/fej()(0)*uv_norm_zero_fej;
    return 1.0 / value()(0) * uv_norm_zero;
  }

  // Failure
  assert(false);
  return Eigen::Vector3d::Zero();
}

void Landmark::set_from_xyz(Eigen::Matrix<double, 3, 1> p_FinG, bool isfej) {

  // CASE: Global 3d feature representation
  // CASE: Anchored 3d feature representation
  if (_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D ||
      _feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
    if (isfej)
      set_fej(p_FinG);
    else
      set_value(p_FinG);
    return;
  }

  // CASE: Global inverse depth feature representation
  // CASE: Anchored inverse depth feature representation
  if (_feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH ||
      _feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {

    // Feature inverse representation
    // NOTE: This is not the MSCKF inverse form, but the standard form
    // NOTE: Thus we go from p_FinG and convert it to this form
    double g_rho = 1 / p_FinG.norm();
    double g_phi = std::acos(g_rho * p_FinG(2));
    // double g_theta = std::asin(g_rho*p_FinG(1)/std::sin(g_phi));
    double g_theta = std::atan2(p_FinG(1), p_FinG(0));
    Eigen::Matrix<double, 3, 1> p_invFinG;
    p_invFinG(0) = g_theta;
    p_invFinG(1) = g_phi;
    p_invFinG(2) = g_rho;

    // Set our feature value
    if (isfej)
      set_fej(p_invFinG);
    else
      set_value(p_invFinG);
    return;
  }

  // CASE: MSCKF anchored inverse depth representation
  if (_feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {

    // MSCKF representation
    Eigen::Matrix<double, 3, 1> p_invFinA_MSCKF;
    p_invFinA_MSCKF(0) = p_FinG(0) / p_FinG(2);
    p_invFinA_MSCKF(1) = p_FinG(1) / p_FinG(2);
    p_invFinA_MSCKF(2) = 1 / p_FinG(2);

    // Set our feature value
    if (isfej)
      set_fej(p_invFinA_MSCKF);
    else
      set_value(p_invFinA_MSCKF);
    return;
  }

  // CASE: Estimate single depth of the feature using the initial bearing
  if (_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

    // Get the inverse depth
    Eigen::VectorXd temp;
    temp.resize(1, 1);
    temp(0) = 1.0 / p_FinG(2);

    // Set our bearing vector
    if (!isfej)
      uv_norm_zero = 1.0 / p_FinG(2) * p_FinG;
    else
      uv_norm_zero_fej = 1.0 / p_FinG(2) * p_FinG;

    // Set our feature value
    if (isfej)
      set_fej(temp);
    else
      set_value(temp);
    return;
  }

  // Failure
  assert(false);
}
