#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>

namespace pnpsolver {

/**
 * @brief Skew-symmetric matrix from a given 3x1 vector
 *
 * This is based on equation 6 in [Indirect Kalman Filter for 3D Attitude Estimation](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf):
 * \f{align*}{
 *  \lfloor\mathbf{v}\times\rfloor =
 *  \begin{bmatrix}
 *  0 & -v_3 & v_2 \\ v_3 & 0 & -v_1 \\ -v_2 & v_1 & 0
 *  \end{bmatrix}
 * @f}
 *
 * @param[in] w 3x1 vector to be made a skew-symmetric
 * @return 3x3 skew-symmetric matrix
 */
inline Eigen::Matrix<double, 3, 3> Skew(const Eigen::Matrix<double, 3, 1>& w) {
  Eigen::Matrix<double, 3, 3> w_x;
  w_x << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  return w_x;
}

/**
 * @brief Returns vector portion of skew-symmetric
 *
 * See skew_x() for details.
 *
 * @param[in] w_x skew-symmetric matrix
 * @return 3x1 vector portion of skew
 */
inline Eigen::Matrix<double, 3, 1> Vee(const Eigen::Matrix<double, 3, 3>& w_x) {
  Eigen::Matrix<double, 3, 1> w;
  w << w_x(2, 1), w_x(0, 2), w_x(1, 0);
  return w;
}

/**
 * @brief SO(3) matrix exponential
 *
 * SO(3) matrix exponential mapping from the vector to SO(3) lie group.
 * This formula ends up being the [Rodrigues formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).
 * This definition was taken from "Lie Groups for 2D and 3D Transformations" by Ethan Eade equation 15.
 * http://ethaneade.com/lie.pdf
 *
 * \f{align*}{
 * \exp\colon\mathfrak{so}(3)&\to SO(3) \\
 * \exp(\mathbf{v}) &=
 * \mathbf{I}
 * +\frac{\sin{\theta}}{\theta}\lfloor\mathbf{v}\times\rfloor
 * +\frac{1-\cos{\theta}}{\theta^2}\lfloor\mathbf{v}\times\rfloor^2 \\
 * \mathrm{where}&\quad \theta^2 = \mathbf{v}^\top\mathbf{v}
 * @f}
 *
 * @param[in] w 3x1 vector we will take the exponential of
 * @return SO(3) rotation matrix
 */
inline Eigen::Matrix<double, 3, 3> ExpSO3(const Eigen::Matrix<double, 3, 1>& w) {
  // get theta
  Eigen::Matrix<double, 3, 3> w_x = Skew(w);
  double theta = w.norm();
  // Handle small angle values
  double A, B;
  if (theta < 1e-12) {
    A = 1;
    B = 0.5;
  } else {
    A = sin(theta) / theta;
    B = (1 - cos(theta)) / (theta * theta);
  }
  // compute so(3) rotation
  Eigen::Matrix<double, 3, 3> R;
  if (theta == 0) {
    R = Eigen::MatrixXd::Identity(3, 3);
  } else {
    R = Eigen::MatrixXd::Identity(3, 3) + A * w_x + B * w_x * w_x;
  }
  return R;
}

/**
 * @brief SO(3) matrix logarithm
 *
 * This definition was taken from "Lie Groups for 2D and 3D Transformations" by Ethan Eade equation 17 & 18.
 * http://ethaneade.com/lie.pdf
 * \f{align*}{
 * \theta &= \textrm{arccos}(0.5(\textrm{trace}(\mathbf{R})-1)) \\
 * \lfloor\mathbf{v}\times\rfloor &= \frac{\theta}{2\sin{\theta}}(\mathbf{R}-\mathbf{R}^\top)
 * @f}
 *
 * @param[in] R 3x3 SO(3) rotation matrix
 * @return 3x1 in the se(3) space [omegax, omegay, omegaz]
 */
inline Eigen::Matrix<double, 3, 1> LogSO3(const Eigen::Matrix<double, 3, 3>& R) {
  // magnitude of the skew elements (handle edge case where we sometimes have a>1...)
  double a = 0.5 * (R.trace() - 1);
  double theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));
  // Handle small angle values
  double D;
  if (theta < 1e-12) {
    D = 0.5;
  } else {
    D = theta / (2 * sin(theta));
  }
  // calculate the skew symetric matrix
  Eigen::Matrix<double, 3, 3> w_x = D * (R - R.transpose());
  // check if we are near the identity
  if (R != Eigen::MatrixXd::Identity(3, 3)) {
    Eigen::Vector3d vec;
    vec << w_x(2, 1), w_x(0, 2), w_x(1, 0);
    return vec;
  } else {
    return Eigen::Vector3d::Zero();
  }
}

/**
 * @brief SE(3) matrix exponential function
 *
 * Equation is from Ethan Eade's reference: http://ethaneade.com/lie.pdf
 * \f{align*}{
 * \exp([\boldsymbol\omega,\mathbf u])&=\begin{bmatrix} \mathbf R & \mathbf V \mathbf u \\ \mathbf 0 & 1 \end{bmatrix} \\[1em]
 * \mathbf R &= \mathbf I + A \lfloor \boldsymbol\omega \times\rfloor + B \lfloor \boldsymbol\omega \times\rfloor^2 \\
 * \mathbf V &= \mathbf I + B \lfloor \boldsymbol\omega \times\rfloor + C \lfloor \boldsymbol\omega \times\rfloor^2
 * \f}
 * where we have the following definitions
 * \f{align*}{
 * \theta &= \sqrt{\boldsymbol\omega^\top\boldsymbol\omega} \\
 * A &= \sin\theta/\theta \\
 * B &= (1-\cos\theta)/\theta^2 \\
 * C &= (1-A)/\theta^2
 * \f}
 *
 * @param vec 6x1 in the se(3) space [omega, u]
 * @return 4x4 SE(3) matrix
 */
inline Eigen::Matrix4d ExpSE3(const Eigen::Matrix<double, 6, 1>& vec) {
  // Precompute our values
  Eigen::Vector3d w = vec.head(3);
  Eigen::Vector3d u = vec.tail(3);
  double theta = sqrt(w.dot(w));
  Eigen::Matrix3d wskew;
  wskew << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  // Handle small angle values
  double A, B, C;
  if (theta < 1e-12) {
    A = 1;
    B = 0.5;
    C = 1.0 / 6.0;
  } else {
    A = sin(theta) / theta;
    B = (1 - cos(theta)) / (theta * theta);
    C = (1 - A) / (theta * theta);
  }

  // Matrices we need V and Identity
  Eigen::Matrix3d I_33 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d V = I_33 + B * wskew + C * wskew * wskew;

  // Get the final matrix to return
  Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
  mat.block(0, 0, 3, 3) = I_33 + A * wskew + B * wskew * wskew;
  mat.block(0, 3, 3, 1) = V * u;
  mat(3, 3) = 1;
  return mat;
}

/**
 * @brief SO(3) matrix logarithm
 *
 * This definition was taken from "Lie Groups for 2D and 3D Transformations" by Ethan Eade equation 17 & 18.
 * http://ethaneade.com/lie.pdf
 * \f{align*}{
 * \theta &= \textrm{arccos}(0.5(\textrm{trace}(\mathbf{R})-1)) \\
 * \lfloor\mathbf{v}\times\rfloor &= \frac{\theta}{2\sin{\theta}}(\mathbf{R}-\mathbf{R}^\top)
 * @f}
 *
 * @param[in] R 3x3 SO(3) rotation matrix
 * @return 3x1 in the se(3) space [omegax, omegay, omegaz]
 */
inline Eigen::Matrix<double, 3, 1> LogSE3(const Eigen::Matrix<double, 3, 3>& R) {
  // magnitude of the skew elements (handle edge case where we sometimes have a>1...)
  double a = 0.5 * (R.trace() - 1);
  double theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));
  // Handle small angle values
  double D;
  if (theta < 1e-12) {
    D = 0.5;
  } else {
    D = theta / (2 * sin(theta));
  }
  // calculate the skew symetric matrix
  Eigen::Matrix<double, 3, 3> w_x = D * (R - R.transpose());
  // check if we are near the identity
  if (R != Eigen::MatrixXd::Identity(3, 3)) {
    Eigen::Vector3d vec;
    vec << w_x(2, 1), w_x(0, 2), w_x(1, 0);
    return vec;
  } else {
    return Eigen::Vector3d::Zero();
  }
}

/**
 * @brief Hat operator for R^6 -> Lie Algebra se(3)
 *
 * \f{align*}{
 * \boldsymbol\Omega^{\wedge} = \begin{bmatrix} \lfloor \boldsymbol\omega \times\rfloor & \mathbf u \\ \mathbf 0 & 0 \end{bmatrix}
 * \f}
 *
 * @param vec 6x1 in the se(3) space [omega, u]
 * @return Lie algebra se(3) 4x4 matrix
 */
inline Eigen::Matrix4d HatSE3(const Eigen::Matrix<double, 6, 1>& vec) {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
  mat.block(0, 0, 3, 3) = Skew(vec.head(3));
  mat.block(0, 3, 3, 1) = vec.tail(3);
  return mat;
}

/**
 * @brief SE(3) matrix analytical inverse
 *
 * It seems that using the .inverse() function is not a good way.
 * This should be used in all cases we need the inverse instead of numerical inverse.
 * https://github.com/rpng/open_vins/issues/12
 * \f{align*}{
 * \mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^\top & -\mathbf{R}^\top\mathbf{p} \\ \mathbf{0} & 1 \end{bmatrix}
 * \f}
 *
 * @param[in] T SE(3) matrix
 * @return inversed SE(3) matrix
 */
inline Eigen::Matrix4d InvSE3(const Eigen::Matrix4d& T) {
  Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
  Tinv.block(0, 0, 3, 3) = T.block(0, 0, 3, 3).transpose();
  Tinv.block(0, 3, 3, 1) = -Tinv.block(0, 0, 3, 3) * T.block(0, 3, 3, 1);
  return Tinv;
}

/**
 * @brief Computes left Jacobian of SO(3)
 *
 * The left Jacobian of SO(3) is defined equation (7.77b) in [State Estimation for
 * Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) by Timothy D. Barfoot. Specifically it is the following (with
 * \f$\theta=|\boldsymbol\theta|\f$ and \f$\mathbf a=\boldsymbol\theta/|\boldsymbol\theta|\f$): \f{align*}{ J_l(\boldsymbol\theta) =
 * \frac{\sin\theta}{\theta}\mathbf I + \Big(1-\frac{\sin\theta}{\theta}\Big)\mathbf a \mathbf a^\top + \frac{1-\cos\theta}{\theta}\lfloor
 * \mathbf a \times\rfloor \f}
 *
 * @param w axis-angle
 * @return The left Jacobian of SO(3)
 */
inline Eigen::Matrix<double, 3, 3> LeftJacobianSO3(const Eigen::Matrix<double, 3, 1>& w) {
  double theta = w.norm();
  if (theta < 1e-12) {
    return Eigen::MatrixXd::Identity(3, 3);
  } else {
    Eigen::Matrix<double, 3, 1> a = w / theta;
    Eigen::Matrix<double, 3, 3> J = sin(theta) / theta * Eigen::MatrixXd::Identity(3, 3) +
                                    (1 - sin(theta) / theta) * a * a.transpose() +
                                    ((1 - cos(theta)) / theta) * Skew(a);
    return J;
  }
}

inline Eigen::Matrix<double, 3, 3> RightJacobianSO3(const Eigen::Matrix<double, 3, 1>& w) {
  return LeftJacobianSO3(-w);
}

}  // namespace pnpsolver