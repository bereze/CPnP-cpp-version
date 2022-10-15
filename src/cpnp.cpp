#include "pnp/cpnp.h"

#include <Eigen/Eigenvalues>
#include <iostream>

#include "pnp/ops.h"

namespace pnpsolver {

/**
 * @brief Generalized Eigenvalues Solver, taken from "Eigenvalue and Generalized Eigenvalue Problems: Tutorial"
 * https://arxiv.org/pdf/1903.11240
 * 
 * det(A - lambda * B) = 0
 * 
 * @param A 
 * @param B 
 * @param eigen_vector 
 * @param eigen_values 
 * @return true 
 * @return false 
 */
static bool GeneralizedEigenSolver(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                   Eigen::MatrixXd& eigen_vector, Eigen::MatrixXd& eigen_values) {
  int N = B.rows();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
  Eigen::VectorXd lambda_B = es.eigenvalues();
  Eigen::MatrixXd phi_B = es.eigenvectors();

  Eigen::MatrixXd lambda_B_sqrt = 1e-4 * Eigen::MatrixXd::Identity(N, N);
  for (size_t i = 0; i < N; i++) {
    lambda_B_sqrt(i, i) += sqrt(lambda_B(i));
  }

  Eigen::MatrixXd phi_B_hat = phi_B * lambda_B_sqrt.inverse();
  Eigen::MatrixXd A_hat = phi_B_hat.transpose() * A * phi_B_hat;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_hat(A_hat);
  eigen_values = es_hat.eigenvalues();
  eigen_vector = phi_B_hat * es_hat.eigenvectors();

  return true;
}

bool CPnP(const std::vector<Eigen::Vector2d>& points_2d,
          const std::vector<Eigen::Vector3d>& points_3d,
          const std::vector<double>& params,
          Eigen::Vector4d& qvec,
          Eigen::Vector3d& tvec,
          Eigen::Vector4d& qvec_GN,
          Eigen::Vector3d& tvec_GN) {
  assert(points_2d.size() == points_3d.size());
  const int N = points_2d.size();

  Eigen::Vector3d bar_p3d = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < N; ++i) {
    bar_p3d += points_3d[i];
  }
  bar_p3d /= N;

  /// Step1: Calculate decentralized pixel coordinates
  Eigen::MatrixXd obs = Eigen::MatrixXd(2 * N, 1);
  for (size_t i = 0; i < points_2d.size(); ++i) {
    obs(2 * i) = points_2d[i](0) - params[2];
    obs(2 * i + 1) = points_2d[i](1) - params[3];
  }

  /// Step2: Estimate the variance of projection noises
  Eigen::MatrixXd Ones_2n = Eigen::MatrixXd::Ones(2 * N, 1);
  Eigen::Matrix2d W = Eigen::Matrix2d::Zero();
  W(0, 0) = params[0];
  W(1, 1) = params[1];

  // A, G
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * N, 11);
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(2 * N, 11);
  for (size_t i = 0; i < N; i++) {
    Eigen::Vector3d p3d_de = points_3d[i] - bar_p3d;
    G.block<1, 3>(2 * i, 0) = -p3d_de.transpose();
    G.block<1, 3>(2 * i + 1, 0) = -p3d_de.transpose();

    A.block<1, 3>(2 * i, 0) = -obs(2 * i) * p3d_de.transpose();
    A.block<1, 3>(2 * i, 3) = params[0] * points_3d[i].transpose();
    A(2 * i, 6) = params[0];

    A.block<1, 3>(2 * i + 1, 0) = -obs(2 * i + 1) * p3d_de.transpose();
    A.block<1, 3>(2 * i + 1, 7) = params[1] * points_3d[i].transpose();
    A(2 * i + 1, 10) = params[1];
  }

  Eigen::MatrixXd Phi = Eigen::MatrixXd(12, 12);
  Phi.block<11, 11>(0, 0) = A.transpose() * A;
  Phi.block<11, 1>(0, 11) = A.transpose() * obs;
  Phi.block<1, 11>(11, 0) = obs.transpose() * A;
  Phi.block<1, 1>(11, 11) = obs.transpose() * obs;
  Phi /= (2 * N);

  Eigen::MatrixXd Delta = Eigen::MatrixXd(12, 12);
  Delta.block<11, 11>(0, 0) = G.transpose() * G;
  Delta.block<11, 1>(0, 11) = G.transpose() * Ones_2n;
  Delta.block<1, 11>(11, 0) = Ones_2n.transpose() * G;
  Delta(11, 11) = 2 * N;
  Delta /= (2 * N);

  // |Phi - sigma2 * Delta| = 0
  // Eigen::MatrixXd eigen_vector;
  // Eigen::MatrixXd eigen_values;
  // generalizedEigenSolver(Phi, Delta, eigen_vector, eigen_values);

  // double sigma2_est = 1e9;
  // for (size_t i = 0; i < eigen_values.rows(); i++) {
  //   if (sigma2_est > abs(eigen_values(i))) {
  //     sigma2_est = abs(eigen_values(i));
  //   }
  // }

  double sigma2_est = 1e12;
  Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(Phi, Delta);
  Eigen::VectorXcd eigenvalues = ges.eigenvalues();
  for (int i = 0; i < eigenvalues.rows(); ++i) {
    if (sigma2_est > abs(eigenvalues(i).real())) {
      sigma2_est = abs(eigenvalues(i).real());
    }
  }

  /// Step3: Calculate the bias-eliminated solution
  Eigen::VectorXd est_bias_eli = (A.transpose() * A - sigma2_est * G.transpose() * G).inverse() *
                                 (A.transpose() * obs - sigma2_est * G.transpose() * Ones_2n);
  /// Step4: Recover R and t
  Eigen::Matrix3d R_bias_eli;
  R_bias_eli.block<1, 3>(0, 0) = est_bias_eli.segment<3>(3).transpose();
  R_bias_eli.block<1, 3>(1, 0) = est_bias_eli.segment<3>(7).transpose();
  R_bias_eli.block<1, 3>(2, 0) = est_bias_eli.segment<3>(0).transpose();

  Eigen::Vector3d t_bias_eli;
  t_bias_eli << est_bias_eli(6), est_bias_eli(10),
      1 - bar_p3d.transpose() * est_bias_eli.segment<3>(0);

  double normalize_factor = pow(R_bias_eli.determinant(), 1.0 / 3.0);
  R_bias_eli /= normalize_factor;
  t_bias_eli /= normalize_factor;

  /// Step5: Project the rotation matrix into SO(3) using SVD
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_bias_eli, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd UVt = svd.matrixU() * svd.matrixV().transpose();
  R_bias_eli = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1, 1, UVt.determinant()) * svd.matrixV().transpose();

  /// Step6: Refine the estimate using the GN iterations
  Eigen::Matrix<double, 2, 3> E;
  E << 1, 0, 0, 0, 1, 0;
  Eigen::MatrixXd WE = W * E;
  Eigen::Vector3d e3(0, 0, 1);

  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(3, N);
  Eigen::MatrixXd t_tile = Eigen::MatrixXd::Zero(3, N);
  for (size_t i = 0; i < points_3d.size(); ++i) {
    s.block<3, 1>(0, i) = points_3d[i];
    t_tile.block<3, 1>(0, i) = t_bias_eli;
  }

  Eigen::MatrixXd LRst = R_bias_eli * s + t_tile;
  Eigen::MatrixXd g = WE * LRst;
  Eigen::MatrixXd h = e3.transpose() * LRst;

  Eigen::VectorXd f = Eigen::VectorXd::Zero(2 * N, 1);
  for (int i = 0; i < g.cols(); ++i) {
    f.segment<2>(2 * i) = g.col(i) / h(0, i);
  }

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2 * N, 6);
  for (int i = 0; i < N; ++i) {
    Eigen::Matrix<double, 2, 3> hWEge = h(0, i) * WE - g.col(i) * e3.transpose();
    Eigen::Matrix<double, 3, 6> pRP;
    pRP.col(0) = s(1, i) * R_bias_eli.col(2) - s(2, i) * R_bias_eli.col(1);
    pRP.col(1) = s(2, i) * R_bias_eli.col(0) - s(0, i) * R_bias_eli.col(2);
    pRP.col(2) = s(0, i) * R_bias_eli.col(1) - s(1, i) * R_bias_eli.col(0);
    pRP.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    J.block<2, 6>(2 * i, 0) = hWEge * pRP / (h(0, i) * h(0, i));
  }

  Eigen::VectorXd initial = Eigen::VectorXd::Zero(6);
  initial.tail<3>() = t_bias_eli;
  Eigen::VectorXd results = initial + (J.transpose() * J).inverse() * J.transpose() * (obs - f);
  Eigen::Vector3d r_GN = results.head<3>();
  Eigen::Vector3d t_GN = results.tail<3>();
  Eigen::Matrix3d R_GN = R_bias_eli * ExpSO3(r_GN);

  const Eigen::Quaterniond quat(R_bias_eli);
  qvec = Eigen::Vector4d(quat.w(), quat.x(), quat.y(), quat.z());
  tvec = t_bias_eli;

  const Eigen::Quaterniond quat_GN(R_GN);
  qvec_GN = Eigen::Vector4d(quat_GN.w(), quat_GN.x(), quat_GN.y(), quat_GN.z());
  tvec_GN = t_GN;

  return true;
}

}  // namespace pnpsolver
