#include "gr/io/tables.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gr {
namespace {

// Header lines of the CSV tables, and of a TUM file (none: it may have
// '#' comments).
constexpr int kCsvHeaderLines = 1;
constexpr int kTumHeaderLines = 0;

// Positions in an (x, y, z, w) quaternion.
constexpr int kQuaternionX = 0;
constexpr int kQuaternionY = 1;
constexpr int kQuaternionZ = 2;
constexpr int kQuaternionW = 3;
constexpr int kQuaternionComponents = kQuaternionW + 1;

// Columns of a TUM row: the stamp, the position, then the quaternion
// x, y, z, w.
constexpr Eigen::Index kTumStampColumn = 0;
constexpr Eigen::Index kTumPositionColumn = 1;
constexpr Eigen::Index kTumQuaternionColumn = kTumPositionColumn + kSpaceDim;
constexpr Eigen::Index kTumColumns =
    kTumQuaternionColumn + kQuaternionComponents;

// Room for one formatted TUM line.
constexpr size_t kTumLineBuffer = 256;

// The numbers of one line split at `separator` (any blank run for ' ').
std::vector<double> ParseRow(const std::string& line, char separator,
                             const std::string& where) {
  std::vector<double> row;
  const char* cursor = line.c_str();
  while (true) {
    while (*cursor == ' ' || *cursor == '\t') ++cursor;
    if (*cursor == '\0' || *cursor == '\r') break;
    char* end = nullptr;
    const double value = std::strtod(cursor, &end);
    if (end == cursor) throw std::runtime_error(where + ": not a number");
    row.push_back(value);
    cursor = end;
    while (*cursor == ' ' || *cursor == '\t') ++cursor;
    if (*cursor == separator && separator != ' ') ++cursor;
  }
  return row;
}

Eigen::MatrixXd ReadTable(const std::string& path, char separator,
                          int header_lines) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<std::vector<double>> rows;
  std::string line;
  int number = 0;
  while (std::getline(in, line)) {
    ++number;
    if (number <= header_lines) continue;
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
  for (size_t row = 0; row < rows.size(); ++row) {
    for (size_t column = 0; column < rows[row].size(); ++column) {
      out(row, column) = rows[row][column];
    }
  }
  return out;
}

}  // namespace

Eigen::MatrixXd ReadCsv(const std::string& path) {
  return ReadTable(path, ',', kCsvHeaderLines);
}

Eigen::Vector4d QuaternionXyzw(const Mat3& rotation) {
  // scipy.spatial.transform.Rotation.from_matrix (Markley's method): start
  // from the largest of the trace and the diagonal entries. The axes x, y, z
  // are 0, 1, 2, as their quaternion positions.
  const double trace = rotation.trace();
  int largest = kQuaternionW;  // The trace.
  double best = trace;
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    if (rotation(axis, axis) > best) {
      best = rotation(axis, axis);
      largest = axis;
    }
  }
  Eigen::Vector4d quaternion;
  if (largest != kQuaternionW) {
    // The largest diagonal axis and the two after it, cyclically.
    const int first = largest;
    const int second = (first + 1) % kSpaceDim;
    const int third = (second + 1) % kSpaceDim;
    quaternion[first] = 1.0 - trace + 2.0 * rotation(first, first);
    quaternion[second] = rotation(second, first) + rotation(first, second);
    quaternion[third] = rotation(third, first) + rotation(first, third);
    quaternion[kQuaternionW] =
        rotation(third, second) - rotation(second, third);
  } else {
    const int x = kQuaternionX;
    const int y = kQuaternionY;
    const int z = kQuaternionZ;
    quaternion[x] = rotation(z, y) - rotation(y, z);
    quaternion[y] = rotation(x, z) - rotation(z, x);
    quaternion[z] = rotation(y, x) - rotation(x, y);
    quaternion[kQuaternionW] = 1.0 + trace;
  }
  return quaternion / quaternion.norm();
}

Trajectory ReadTum(const std::string& path) {
  const Eigen::MatrixXd table = ReadTable(path, ' ', kTumHeaderLines);
  if (table.rows() > 0 && table.cols() < kTumColumns) {
    throw std::runtime_error(path + ": TUM rows need " +
                             std::to_string(kTumColumns) + " columns");
  }
  Trajectory out;
  for (Eigen::Index row = 0; row < table.rows(); ++row) {
    const auto quaternion_at = [&](int component) {
      return table(row, kTumQuaternionColumn + component);
    };
    // Eigen's constructor takes w first.
    const Eigen::Quaterniond quaternion(
        quaternion_at(kQuaternionW), quaternion_at(kQuaternionX),
        quaternion_at(kQuaternionY), quaternion_at(kQuaternionZ));
    out.stamps.push_back(table(row, kTumStampColumn));
    out.poses.emplace_back(
        quaternion.normalized().toRotationMatrix(),
        table.block<1, kSpaceDim>(row, kTumPositionColumn).transpose());
  }
  return out;
}

void WriteTum(const std::string& path, const Trajectory& trajectory) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  char line[kTumLineBuffer];
  for (size_t k = 0; k < trajectory.stamps.size(); ++k) {
    const Pose& pose = trajectory.poses[k];
    const Eigen::Vector4d quaternion = QuaternionXyzw(pose.rotation);
    std::snprintf(line, sizeof(line),
                  "%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n",
                  trajectory.stamps[k], pose.translation.x(),
                  pose.translation.y(), pose.translation.z(),
                  quaternion[kQuaternionX], quaternion[kQuaternionY],
                  quaternion[kQuaternionZ], quaternion[kQuaternionW]);
    out << line;
  }
}

}  // namespace gr
