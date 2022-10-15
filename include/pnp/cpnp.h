#pragma once

#include <Eigen/Core>
#include <vector>

namespace pnpsolver {

/**
 * @brief  Estimate the camera pose from 2D-3D correspondences
 * 
 * The CPnP algorithm is taken from "CPnP: Consistent Pose Estimator for Perspective-n-Point 
 * Problem with Bias Elimination" https://arxiv.org/pdf/2209.05824
 * 
 * @param points_2d 2D image points
 * @param points_3d 3D world points
 * @param params    camera params [fx, fy, cx, cy]
 * @param qvec      Quaternion (qw, qx, qy, qz) from world to camera
 * @param tvec      Translation (tx, ty, tz) from world to camera
 * @param qvec_GN   Refined quaternion (qw, qx, qy, qz) from world to camera
 * @param tvec_GN   Refined translation (tx, ty, tz) from world to camera
 * @return true 
 * @return false 
 */
bool CPnP(const std::vector<Eigen::Vector2d>& points_2d,
          const std::vector<Eigen::Vector3d>& points_3d,
          const std::vector<double>& params,
          Eigen::Vector4d& qvec,
          Eigen::Vector3d& tvec,
          Eigen::Vector4d& qvec_GN,
          Eigen::Vector3d& tvec_GN);
}  // namespace pnpsolver