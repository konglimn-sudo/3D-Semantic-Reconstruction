#include "gr/nav/arrows.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Eigenvalues>

#include "gr/core/check.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;
// Components are 8-connected: the cells this many steps around.
constexpr int kNeighbourhoodRadius = 1;
// A cell not assigned to a component yet.
constexpr int kNoLabel = -1;
// Beyond the head the width must fall: the end bin below this share of the
// head width.
constexpr double kMaxTipWidthShare = 0.7;

// Cell indices of the 8-connected components of `mask`.
std::vector<std::vector<int>> Components(const std::vector<uint8_t>& mask,
                                         int width, int height) {
  std::vector<int> label(mask.size(), kNoLabel);
  std::vector<std::vector<int>> components;
  std::vector<int> stack;
  for (int start = 0; start < width * height; ++start) {
    if (!mask[static_cast<size_t>(start)] ||
        label[static_cast<size_t>(start)] != kNoLabel) {
      continue;
    }
    const int id = static_cast<int>(components.size());
    components.emplace_back();
    stack.assign(1, start);
    label[static_cast<size_t>(start)] = id;
    while (!stack.empty()) {
      const int cell = stack.back();
      stack.pop_back();
      components.back().push_back(cell);
      const int row = cell / width;
      const int column = cell % width;
      for (int dr = -kNeighbourhoodRadius; dr <= kNeighbourhoodRadius; ++dr) {
        for (int dc = -kNeighbourhoodRadius; dc <= kNeighbourhoodRadius;
             ++dc) {
          const int neighbour_row = row + dr;
          const int neighbour_column = column + dc;
          if (neighbour_row < 0 || neighbour_column < 0 ||
              neighbour_row >= height || neighbour_column >= width) {
            continue;
          }
          const size_t neighbour =
              static_cast<size_t>(neighbour_row * width + neighbour_column);
          if (!mask[neighbour] || label[neighbour] != kNoLabel) continue;
          label[neighbour] = id;
          stack.push_back(static_cast<int>(neighbour));
        }
      }
    }
  }
  return components;
}

}  // namespace

std::vector<FloorArrow> DetectArrows(const std::vector<uint8_t>& mask,
                                     int width, int height, double res,
                                     double x0, double y0,
                                     const ArrowConfig& config) {
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  GR_CHECK(res > 0.0 && config.bin > 0.0);
  const double cell_area = res * res;
  std::vector<FloorArrow> arrows;
  for (const std::vector<int>& cells : Components(mask, width, height)) {
    const double area = static_cast<double>(cells.size()) * cell_area;
    if (area < config.min_area || area > config.max_area) continue;
    std::vector<Vec2> points;
    points.reserve(cells.size());
    Vec2 center = Vec2::Zero();
    for (int cell : cells) {
      const Vec2 point(x0 + (cell % width + kCellCentre) * res,
                       y0 + (cell / width + kCellCentre) * res);
      points.push_back(point);
      center += point;
    }
    center /= static_cast<double>(points.size());
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (const Vec2& point : points) {
      scatter += (point - center) * (point - center).transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(scatter);
    // The eigenvector of the largest eigenvalue (eigenvalues ascend).
    const Vec2 axis = solver.eigenvectors().col(kPlaneDim - 1);
    double along_min = kInf;
    double along_max = -kInf;
    std::vector<double> along(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      along[i] = (points[i] - center).dot(axis);
      along_min = std::min(along_min, along[i]);
      along_max = std::max(along_max, along[i]);
    }
    const double length = along_max - along_min + res;
    if (length < config.min_length || length > config.max_length) continue;
    // Width profile: painted area per bin over the bin length.
    const int bins = std::max(
        1, static_cast<int>(std::ceil((along_max - along_min + res) /
                                      config.bin)));
    std::vector<double> profile(static_cast<size_t>(bins), 0.0);
    for (double position : along) {
      const int bin = std::min(
          bins - 1, static_cast<int>((position - along_min +
                                      kCellCentre * res) /
                                     config.bin));
      profile[static_cast<size_t>(bin)] += cell_area / config.bin;
    }
    const double shaft = Median(profile);
    const int widest = static_cast<int>(
        std::max_element(profile.begin(), profile.end()) - profile.begin());
    const double head = profile[static_cast<size_t>(widest)];
    if (shaft > config.max_shaft_width || head < config.min_head_width ||
        head < config.min_head_ratio * std::max(shaft, res)) {
      continue;
    }
    const double head_position =
        along_min + (widest + kCellCentre) * config.bin;
    const double middle = 0.5 * (along_min + along_max);
    if (std::abs(head_position - middle) < config.min_head_offset * length) {
      continue;
    }
    const double sign = head_position > middle ? 1.0 : -1.0;
    // The tip: beyond the widest bin the width must fall off.
    const double tip_length =
        sign > 0 ? along_max - head_position : head_position - along_min;
    if (tip_length < config.min_tip_length) continue;
    const int end_bin = sign > 0 ? bins - 1 : 0;
    if (profile[static_cast<size_t>(end_bin)] > kMaxTipWidthShare * head) {
      continue;
    }
    FloorArrow arrow;
    arrow.center = center;
    arrow.direction = sign * axis;
    arrow.tail = center + (sign > 0 ? along_min : along_max) * axis;
    arrow.tip = center + (sign > 0 ? along_max : along_min) * axis;
    arrow.length = length;
    arrow.shaft_width = shaft;
    arrow.head_width = head;
    arrow.area = area;
    arrow.cells = cells;
    arrows.push_back(arrow);
  }
  return arrows;
}

}  // namespace gr
