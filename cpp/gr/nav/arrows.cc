#include "gr/nav/arrows.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Eigenvalues>

#include "gr/core/check.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

// Cell indices of the 8-connected components of `mask`.
std::vector<std::vector<int>> Components(const std::vector<uint8_t>& mask,
                                         int width, int height) {
  std::vector<int> label(mask.size(), -1);
  std::vector<std::vector<int>> components;
  std::vector<int> stack;
  for (int start = 0; start < width * height; ++start) {
    if (!mask[static_cast<size_t>(start)] ||
        label[static_cast<size_t>(start)] >= 0) {
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
      const int col = cell % width;
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          const int r = row + dr;
          const int c = col + dc;
          if (r < 0 || c < 0 || r >= height || c >= width) continue;
          const size_t n = static_cast<size_t>(r * width + c);
          if (!mask[n] || label[n] >= 0) continue;
          label[n] = id;
          stack.push_back(static_cast<int>(n));
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
  for (const std::vector<int>& cells :
       Components(mask, width, height)) {
    const double area = static_cast<double>(cells.size()) * cell_area;
    if (area < config.min_area || area > config.max_area) continue;
    std::vector<Vec2> points;
    points.reserve(cells.size());
    Vec2 center = Vec2::Zero();
    for (int cell : cells) {
      const Vec2 p(x0 + (cell % width + 0.5) * res,
                   y0 + (cell / width + 0.5) * res);
      points.push_back(p);
      center += p;
    }
    center /= static_cast<double>(points.size());
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (const Vec2& p : points) {
      scatter += (p - center) * (p - center).transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(scatter);
    const Vec2 axis = solver.eigenvectors().col(1);  // Largest eigenvalue.
    double s_min = 1e18;
    double s_max = -1e18;
    std::vector<double> along(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      along[i] = (points[i] - center).dot(axis);
      s_min = std::min(s_min, along[i]);
      s_max = std::max(s_max, along[i]);
    }
    const double length = s_max - s_min + res;
    if (length < config.min_length || length > config.max_length) continue;
    // Width profile: painted area per bin over the bin length.
    const int bins = std::max(1, static_cast<int>(std::ceil(
                                     (s_max - s_min + res) / config.bin)));
    std::vector<double> profile(static_cast<size_t>(bins), 0.0);
    for (double s : along) {
      const int b = std::min(
          bins - 1, static_cast<int>((s - s_min + 0.5 * res) / config.bin));
      profile[static_cast<size_t>(b)] += cell_area / config.bin;
    }
    const double shaft = Median(profile);
    const int widest = static_cast<int>(
        std::max_element(profile.begin(), profile.end()) - profile.begin());
    const double head = profile[static_cast<size_t>(widest)];
    if (shaft > config.max_shaft_width || head < config.min_head_width ||
        head < config.min_head_ratio * std::max(shaft, res)) {
      continue;
    }
    const double s_head = s_min + (widest + 0.5) * config.bin;
    const double middle = 0.5 * (s_min + s_max);
    if (std::abs(s_head - middle) < config.min_head_offset * length) continue;
    const double sign = s_head > middle ? 1.0 : -1.0;
    // The tip: beyond the widest bin the width must fall off.
    const double tip_length = sign > 0 ? s_max - s_head : s_head - s_min;
    if (tip_length < config.min_tip_length) continue;
    const int end_bin = sign > 0 ? bins - 1 : 0;
    if (profile[static_cast<size_t>(end_bin)] > 0.7 * head) continue;
    FloorArrow arrow;
    arrow.center = center;
    arrow.direction = sign * axis;
    arrow.tail = center + (sign > 0 ? s_min : s_max) * axis;
    arrow.tip = center + (sign > 0 ? s_max : s_min) * axis;
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
