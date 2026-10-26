#include "gr/semantics/objects.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gr/core/pose.h"

namespace gr {
namespace {

// numpy.arange(-0.5, 0.5 + 1e-9, step): unit-box coordinates along one axis.
std::vector<double> UnitSamples(double step) {
  const double stop = 0.5 + 1e-9;
  const int64_t n = static_cast<int64_t>(std::ceil((stop - (-0.5)) / step));
  std::vector<double> out;
  out.reserve(static_cast<size_t>(std::max<int64_t>(0, n)));
  for (int64_t i = 0; i < n; ++i) {
    out.push_back(-0.5 + static_cast<double>(i) * step);
  }
  return out;
}

}  // namespace

bool ObjectBox::Contains(const Vec3& point, double margin) const {
  const double c = std::cos(-yaw);
  const double s = std::sin(-yaw);
  const Vec3 d = point - center;
  const double local_x = c * d.x() - s * d.y();
  const double local_y = s * d.x() + c * d.y();
  return std::abs(local_x) <= 0.5 * size.x() + margin &&
         std::abs(local_y) <= 0.5 * size.y() + margin &&
         std::abs(d.z()) <= 0.5 * size.z() + margin;
}

Points ObjectBox::SampleSurface(double step) const {
  const double length = size.x();
  const double width = size.y();
  const double height = size.z();
  const std::vector<double> us = UnitSamples(step / std::max(length, step));
  const std::vector<double> vs = UnitSamples(step / std::max(width, step));
  const std::vector<double> ws = UnitSamples(step / std::max(height, step));
  Points local;
  local.reserve(2 * vs.size() * ws.size() + 2 * us.size() * ws.size() +
                us.size() * vs.size());
  for (double sx : {-0.5, 0.5}) {
    for (double w : ws) {
      for (double v : vs) local.emplace_back(sx, v, w);
    }
  }
  for (double sy : {-0.5, 0.5}) {
    for (double w : ws) {
      for (double u : us) local.emplace_back(u, sy, w);
    }
  }
  for (double v : vs) {
    for (double u : us) local.emplace_back(u, v, 0.5);
  }
  const Mat3 rotation =
      Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix();
  Points out;
  out.reserve(local.size());
  for (const Vec3& q : local) {
    out.push_back(rotation * q.cwiseProduct(size) + center);
  }
  return out;
}

void BoxTimeline::AddStatic(int trip, const ObjectBox& box) {
  static_boxes_[trip].push_back(box);
}

void BoxTimeline::AddMoving(int trip, double stamp, const ObjectBox& box) {
  auto [it, inserted] = tracks_.try_emplace(box.track_id);
  if (inserted) track_order_.push_back(box.track_id);
  std::vector<Observation>& track = it->second;
  const auto position = std::upper_bound(
      track.begin(), track.end(), stamp,
      [](double s, const Observation& o) { return s < o.stamp; });
  track.insert(position, Observation{stamp, trip, box});
}

std::vector<ObjectBox> BoxTimeline::At(int trip, double stamp,
                                       double slack) const {
  std::vector<ObjectBox> out;
  if (const auto it = static_boxes_.find(trip); it != static_boxes_.end()) {
    out = it->second;
  }
  for (int id : track_order_) {
    const std::vector<Observation>& track = tracks_.find(id)->second;
    if (track.front().trip != trip || stamp < track.front().stamp - slack ||
        stamp > track.back().stamp + slack) {
      continue;
    }
    if (track.size() == 1) {
      out.push_back(track.front().box);
      continue;
    }
    // First observation at or after the stamp, kept inside [1, n - 1].
    size_t j = static_cast<size_t>(
        std::lower_bound(
            track.begin(), track.end(), stamp,
            [](const Observation& o, double s) { return o.stamp < s; }) -
        track.begin());
    j = std::clamp<size_t>(j, 1, track.size() - 1);
    const Observation& a = track[j - 1];
    const Observation& b = track[j];
    const double w = std::clamp(
        (stamp - a.stamp) / std::max(b.stamp - a.stamp, 1e-6), 0.0, 1.0);
    ObjectBox box = a.box;
    box.center = (1.0 - w) * a.box.center + w * b.box.center;
    box.yaw = a.box.yaw + w * WrapAngle(b.box.yaw - a.box.yaw);
    box.is_static = false;
    out.push_back(box);
  }
  return out;
}

}  // namespace gr
