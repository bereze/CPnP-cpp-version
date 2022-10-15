#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "pnp/cpnp.h"

bool ReadData(const std::string& data_path, std::vector<Eigen::Vector3d>& point_3d,
              std::vector<Eigen::Vector2d>& point_2d) {
  std::string p3d_file = data_path + "/p3d.txt";
  std::string p2d_file = data_path + "/p2d.txt";

  std::ifstream fp3d(p3d_file);
  if (!fp3d) {
    std::cout << "Couldn't open " << p3d_file << std::endl;
    return false;
  } else {
    while (!fp3d.eof()) {
      double pt3[3] = {0};
      for (auto& p : pt3) {
        fp3d >> p;
      }
      Eigen::Vector3d p3d(pt3[0], pt3[1], pt3[2]);
      point_3d.push_back(p3d);
    }
  }

  std::ifstream fp2d(p2d_file);
  if (!fp2d) {
    std::cout << "Couldn't open " << p2d_file << std::endl;
    return false;
  } else {
    while (!fp2d.eof()) {
      double pt2[2] = {0};
      for (auto& p : pt2) {
        fp2d >> p;
      }
      Eigen::Vector2d p2d(pt2[0], pt2[1]);
      point_2d.push_back(p2d);
    }
  }
  assert(point_2d.size() == point_3d.size());

  int nPoints = point_2d.size();
  std::cout << "Read " << nPoints << " points." << std::endl;
  return true;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: ./test_pnp data_path" << std::endl;
    return -1;
  }

  // read data
  std::vector<Eigen::Vector3d> Point3D;
  std::vector<Eigen::Vector2d> Point2D;
  ReadData(argv[1], Point3D, Point2D);

  std::vector<double> params = {520.9, 521.0, 325.1, 249.7};

  Eigen::Vector4d qvec_cpnp, qvec_cpnp_gn;
  Eigen::Vector3d tvec_cpnp, tvec_cpnp_gn;

  if (pnpsolver::CPnP(Point2D, Point3D, params, qvec_cpnp, tvec_cpnp, qvec_cpnp_gn, tvec_cpnp_gn)) {
    std::cout << "qvec: " << qvec_cpnp.transpose() << ", tvec: " << tvec_cpnp.transpose() << std::endl;
    std::cout << "qvec_GN: " << qvec_cpnp_gn.transpose() << ", tvec_GN: " << tvec_cpnp_gn.transpose() << std::endl;
  }

  return 0;
}