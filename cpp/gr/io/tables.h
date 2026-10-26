// Text tables: numeric CSV (imu.csv, wheel.csv, gnss.csv) and TUM
// trajectories ("t x y z qx qy qz qw" per line).

#ifndef GARAGE_RECON_GR_IO_TABLES_H_
#define GARAGE_RECON_GR_IO_TABLES_H_

#include <string>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"

namespace gr {

// Rows of a comma-separated file after its header line (numpy.loadtxt with
// skiprows=1). Throws std::runtime_error on a malformed number or ragged
// rows; an absent file throws too (check first where it is optional).
Eigen::MatrixXd ReadCsv(const std::string& path);

// Unit quaternion (x, y, z, w) of a rotation matrix, with the sign that
// scipy's Rotation.from_matrix gives.
Eigen::Vector4d QuaternionXyzw(const Mat3& rotation);

struct Trajectory {
  std::vector<double> stamps;
  std::vector<Pose> poses;
};

Trajectory ReadTum(const std::string& path);
// Written as geometry.write_tum: stamps with 9 decimals, positions 6,
// quaternions 9.
void WriteTum(const std::string& path, const Trajectory& trajectory);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_TABLES_H_
