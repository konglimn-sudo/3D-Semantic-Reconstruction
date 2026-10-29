#include "gr/semantics/objects.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gr/core/pose.h"

namespace gr {
namespace {

// A unit box spans [-kHalfUnit, kHalfUnit] along each axis.
constexpr double kHalfUnit = 0.5;
// Ranges include their end up to this rounding slack.
constexpr double kRangeEndSlack = 1e-9;
// Two faces (sides) per axis.
constexpr size_t kFacesPerAxis = 2;
// Keyframes closer in time than this (s) do not divide.
constexpr double kMinKeyframeGap = 1e-6;

// numpy.arange(-0.5, 0.5 + 1e-9, step): unit-box coordinates along one axis.
std::vector<double> UnitSamples(double step) {
  const double stop = kHalfUnit + kRangeEndSlack;
  const int64_t count =
      static_cast<int64_t>(std::ceil((stop - (-kHalfUnit)) / step));
  std::vector<double> out;
  out.reserve(static_cast<size_t>(std::max<int64_t>(0, count)));
  for (int64_t i = 0; i < count; ++i) {
    out.push_back(-kHalfUnit + static_cast<double>(i) * step);
  }
  return out;
}

}  // namespace

bool ObjectBox::Contains(const Vec3& point, double margin) const {
  const double cos_yaw = std::cos(-yaw);
  const double sin_yaw = std::sin(-yaw);
  const Vec3 offset = point - center;
  const double local_x = cos_yaw * offset.x() - sin_yaw * offset.y();
  const double local_y = sin_yaw * offset.x() + cos_yaw * offset.y();
  return std::abs(local_x) <= 0.5 * size.x() + margin &&
         std::abs(local_y) <= 0.5 * size.y() + margin &&
         std::abs(offset.z()) <= 0.5 * size.z() + margin;
}

Points ObjectBox::SampleSurface(double step) const {
  const double length = size.x();
  const double width = size.y();
  const double height = size.z();
  const std::vector<double> along_length =
      UnitSamples(step / std::max(length, step));
  const std::vector<double> along_width =
      UnitSamples(step / std::max(width, step));
  const std::vector<double> along_height =
      UnitSamples(step / std::max(height, step));
  Points local;
  local.reserve(kFacesPerAxis * along_width.size() * along_height.size() +
                kFacesPerAxis * along_length.size() * along_height.size() +
                along_length.size() * along_width.size());
  for (double side_x : {-kHalfUnit, kHalfUnit}) {
    for (double z : along_height) {
      for (double y : along_width) local.emplace_back(side_x, y, z);
    }
  }
  for (double side_y : {-kHalfUnit, kHalfUnit}) {
    for (double z : along_height) {
      for (double x : along_length) local.emplace_back(x, side_y, z);
    }
  }
  for (double y : along_width) {
    for (double x : along_length) local.emplace_back(x, y, kHalfUnit);
  }
  const Mat3 rotation =
      Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix();
  Points out;
  out.reserve(local.size());
  for (const Vec3& unit_point : local) {
    out.push_back(rotation * unit_point.cwiseProduct(size) + center);
  }
  return out;
}

void BoxTimeline::AddStatic(int trip, const ObjectBox& box) {
  static_boxes_[trip].push_back(box);
}

void BoxTimeline::AddMoving(int trip, double stamp, const ObjectBox& box) {
  auto [found, inserted] = tracks_.try_emplace(box.track_id);
  if (inserted) track_order_.push_back(box.track_id);
  std::vector<Observation>& track = found->second;
  const auto position = std::upper_bound(
      track.begin(), track.end(), stamp,
      [](double time, const Observation& observation) {
        return time < observation.stamp;
      });
  track.insert(position, Observation{stamp, trip, box});
}

std::vector<ObjectBox> BoxTimeline::At(int trip, double stamp,
                                       double slack) const {
  std::vector<ObjectBox> out;
  if (const auto found = static_boxes_.find(trip);
      found != static_boxes_.end()) {
    out = found->second;
  }
  for (int track_id : track_order_) {
    const std::vector<Observation>& track = tracks_.find(track_id)->second;
    if (track.front().trip != trip || stamp < track.front().stamp - slack ||
        stamp > track.back().stamp + slack) {
      continue;
    }
    if (track.size() == 1) {
      out.push_back(track.front().box);
      continue;
    }
    // First observation at or after the stamp, kept inside [1, n - 1].
    size_t next = static_cast<size_t>(
        std::lower_bound(track.begin(), track.end(), stamp,
                         [](const Observation& observation, double time) {
                           return observation.stamp < time;
                         }) -
        track.begin());
    next = std::clamp<size_t>(next, 1, track.size() - 1);
    const Observation& before = track[next - 1];
    const Observation& after = track[next];
    const double weight = std::clamp(
        (stamp - before.stamp) /
            std::max(after.stamp - before.stamp, kMinKeyframeGap),
        0.0, 1.0);
    ObjectBox box = before.box;
    box.center = (1.0 - weight) * before.box.center + weight * after.box.center;
    box.yaw =
        before.box.yaw + weight * WrapAngle(after.box.yaw - before.box.yaw);
    box.is_static = false;
    out.push_back(box);
  }
  return out;
}

}  // namespace gr
