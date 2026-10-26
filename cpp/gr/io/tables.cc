#include "gr/io/tables.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gr {
namespace {

// The numbers of one line split at `separator` (any blank run for ' ').
std::vector<double> ParseRow(const std::string& line, char separator,
                             const std::string& where) {
  std::vector<double> row;
  const char* p = line.c_str();
  while (true) {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0' || *p == '\r') break;
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p) throw std::runtime_error(where + ": not a number");
    row.push_back(v);
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == separator && separator != ' ') ++p;
  }
  return row;
}

Eigen::MatrixXd ReadTable(const std::string& path, char separator,
                          int skip) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<std::vector<double>> rows;
  std::string line;
  int number = 0;
  while (std::getline(in, line)) {
    ++number;
    if (number <= skip) continue;
    if (separator == ' ' && !line.empty() && line[0] == '#') continue;
    std::vector<double> row =
        ParseRow(line, separator, path + ":" + std::to_string(number));
    if (row.empty()) continue;
    if (!rows.empty() && row.size() != rows[0].size()) {
      throw std::runtime_error(path + ":" + std::to_string(number) +
                               ": ragged row");
    }
    rows.push_back(std::move(row));
  }
  Eigen::MatrixXd out(rows.size(), rows.empty() ? 0 : rows[0].size());
  for (size_t r = 0; r < rows.size(); ++r) {
    for (size_t c = 0; c < rows[r].size(); ++c) out(r, c) = rows[r][c];
  }
  return out;
}

}  // namespace

Eigen::MatrixXd ReadCsv(const std::string& path) {
  return ReadTable(path, ',', 1);
}

Eigen::Vector4d QuaternionXyzw(const Mat3& m) {
  // scipy.spatial.transform.Rotation.from_matrix (Markley's method).
  const double trace = m.trace();
  int choice = 3;
  double best = trace;
  for (int i = 0; i < 3; ++i) {
    if (m(i, i) > best) {
      best = m(i, i);
      choice = i;
    }
  }
  Eigen::Vector4d q;
  if (choice != 3) {
    const int i = choice;
    const int j = (i + 1) % 3;
    const int k = (j + 1) % 3;
    q[i] = 1.0 - trace + 2.0 * m(i, i);
    q[j] = m(j, i) + m(i, j);
    q[k] = m(k, i) + m(i, k);
    q[3] = m(k, j) - m(j, k);
  } else {
    q[0] = m(2, 1) - m(1, 2);
    q[1] = m(0, 2) - m(2, 0);
    q[2] = m(1, 0) - m(0, 1);
    q[3] = 1.0 + trace;
  }
  return q / q.norm();
}

Trajectory ReadTum(const std::string& path) {
  const Eigen::MatrixXd table = ReadTable(path, ' ', 0);
  if (table.rows() > 0 && table.cols() < 8) {
    throw std::runtime_error(path + ": TUM rows need 8 columns");
  }
  Trajectory out;
  for (Eigen::Index r = 0; r < table.rows(); ++r) {
    const Eigen::Quaterniond q(table(r, 7), table(r, 4), table(r, 5),
                               table(r, 6));
    out.stamps.push_back(table(r, 0));
    out.poses.emplace_back(q.normalized().toRotationMatrix(),
                           Vec3(table(r, 1), table(r, 2), table(r, 3)));
  }
  return out;
}

void WriteTum(const std::string& path, const Trajectory& trajectory) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  char line[256];
  for (size_t k = 0; k < trajectory.stamps.size(); ++k) {
    const Pose& T = trajectory.poses[k];
    const Eigen::Vector4d q = QuaternionXyzw(T.rotation);
    std::snprintf(line, sizeof(line),
                  "%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n",
                  trajectory.stamps[k], T.translation.x(), T.translation.y(),
                  T.translation.z(), q[0], q[1], q[2], q[3]);
    out << line;
  }
}

}  // namespace gr
