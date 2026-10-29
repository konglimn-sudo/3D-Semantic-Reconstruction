#include "gr/semantics/labeling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
// Every polyline edge gets at least its two ends as samples; runs need at
// least kMinRunPoints visible samples.
constexpr int kMinEdgeSamples = 2;
constexpr size_t kMinRunPoints = 2;
// Sobel: weight of the centre row / column of the [1, 2, 1] smoothing.
constexpr double kSobelCentreWeight = 2.0;
// Edge strengths are normalised by this percentile (at least kMinEdgeScale).
constexpr double kEdgeScalePercentile = 95.0;
constexpr double kMinEdgeScale = 1e-6;
// Directions of shorter vectors have no meaning.
constexpr double kMinNorm = 1e-9;
// The frame z-buffer: splats of kMinSplatRadius to kMaxSplatRadius pixels,
// points from kMinRenderRange (m) to kRenderRangeMargin (m) beyond the label
// range; boxes up to kBoxRangeMargin (m) beyond it.
constexpr int kMinSplatRadius = 1;
constexpr int kMaxSplatRadius = 4;
constexpr double kMinRenderRange = 0.1;
constexpr double kRenderRangeMargin = 2.0;
constexpr double kBoxRangeMargin = 5.0;
// The lifting z-buffer: splats of at most kMaxLiftSplatRadius pixels, points
// up to kLiftRangeSlack (m) beyond the range.
constexpr int kMaxLiftSplatRadius = 3;
constexpr double kLiftRangeSlack = 1e-6;
// Outlines and markings are lifted this far (m) above the floor so that the
// floor does not occlude them.
constexpr double kFloorLift = 0.01;
// The entrance is the first edge of a slot outline.
constexpr int kEntranceEdge = 0;
// A paint edge needs this many visible samples for a score.
constexpr int kMinScoredSamples = 5;

struct PolylineSamples {
  Points points;
  std::vector<int> edge;  // Edge index of every sample.
};

// Dense samples along a polyline (closed: back to the first vertex); both
// ends of each edge are included.
PolylineSamples SamplePolyline(const Points& vertices, double step,
                               bool closed) {
  PolylineSamples samples;
  const size_t num_vertices = vertices.size();
  const size_t num_edges =
      closed ? num_vertices : (num_vertices > 0 ? num_vertices - 1 : 0);
  for (size_t edge = 0; edge < num_edges; ++edge) {
    const Vec3& start = vertices[edge];
    const Vec3& end = vertices[(edge + 1) % num_vertices];
    const int edge_samples = std::max(
        kMinEdgeSamples,
        static_cast<int>(std::ceil((end - start).norm() / step)) + 1);
    const double fraction_step = 1.0 / (edge_samples - 1);
    for (int sample = 0; sample < edge_samples; ++sample) {
      const double fraction =
          sample == edge_samples - 1 ? 1.0 : sample * fraction_step;
      samples.points.push_back(start + fraction * (end - start));
      samples.edge.push_back(static_cast<int>(edge));
    }
  }
  return samples;
}

// Runs of consecutive visible samples (at least kMinRunPoints each).
std::vector<Polyline2> VisibleRuns(const RowMatX2d& uv,
                                   const std::vector<uint8_t>& visible) {
  std::vector<Polyline2> runs;
  Polyline2 current;
  for (size_t sample = 0; sample < visible.size(); ++sample) {
    const Eigen::Index row = static_cast<Eigen::Index>(sample);
    if (visible[sample]) {
      current.emplace_back(uv(row, 0), uv(row, 1));
    } else if (!current.empty()) {
      if (current.size() >= kMinRunPoints) runs.push_back(current);
      current.clear();
    }
  }
  if (current.size() >= kMinRunPoints) runs.push_back(current);
  return runs;
}

double MeanOf(const std::vector<uint8_t>& flags) {
  if (flags.empty()) return 0.0;
  size_t sum = 0;
  for (uint8_t flag : flags) sum += flag;
  return static_cast<double>(sum) / static_cast<double>(flags.size());
}

bool AnyOf(const std::vector<uint8_t>& flags) {
  return std::any_of(flags.begin(), flags.end(),
                     [](uint8_t flag) { return flag != 0; });
}

// scipy.ndimage mode "reflect": (d c b a | a b c d | d c b a).
int Reflect(int index, int size) {
  if (index < 0) index = -index - 1;
  if (index >= size) index = 2 * size - index - 1;
  return std::clamp(index, 0, size - 1);
}

}  // namespace

// Projects 3D samples into one camera frame with the occlusion test of the
// frame's z-buffer.
class FrameLabeler::Projector {
 public:
  struct Projection {
    RowMatX2d uv;
    std::vector<uint8_t> visible;  // In the image, unoccluded, within range.
  };

  Projector(const CameraModel& camera, const CameraExposure& exposure,
            const std::vector<float>& depth, const FrameLabelConfig& config)
      : camera_(camera),
        exposure_(exposure),
        origin_(exposure.center()),
        depth_(depth),
        config_(config) {}
  Projector(const Projector&) = delete;
  Projector& operator=(const Projector&) = delete;

  Projection Project(const Points& samples) const {
    VisibilityResult result =
        ProjectVisibility(camera_, exposure_, samples, depth_,
                          config_.depth_abs_tol, config_.depth_rel_tol);
    Projection projection;
    projection.visible.resize(samples.size());
    for (size_t sample = 0; sample < samples.size(); ++sample) {
      projection.visible[sample] = result.in_image[sample] &&
                                   result.visible[sample] &&
                                   result.range[sample] <= config_.max_range;
    }
    projection.uv = std::move(result.uv);
    return projection;
  }

  // Smallest distance from the camera centre to any of the points.
  double MinDistance(const Points& points) const {
    double nearest = kInfinity;
    for (const Vec3& point : points) {
      nearest = std::min(nearest, (point - origin_).norm());
    }
    return nearest;
  }
  const Vec3& origin() const { return origin_; }

 private:
  const CameraModel& camera_;
  const CameraExposure& exposure_;
  Vec3 origin_;
  const std::vector<float>& depth_;
  const FrameLabelConfig& config_;
};

std::vector<float> EdgeStrength(const ImageView& image) {
  const int height = image.height;
  const int width = image.width;
  const size_t pixels = static_cast<size_t>(std::max(height, 0)) *
                        static_cast<size_t>(std::max(width, 0));
  std::vector<float> out(pixels, 0.0f);
  if (image.empty() || image.channels < 1) return out;
  std::vector<double> gray(pixels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const uint8_t* pixel = image.PixelAt(y, x);
      double sum = 0.0;
      for (int channel = 0; channel < image.channels; ++channel) {
        sum += pixel[channel];
      }
      gray[static_cast<size_t>(y) * static_cast<size_t>(width) +
           static_cast<size_t>(x)] = sum / image.channels;
    }
  }
  const auto at = [&](const std::vector<double>& values, int y, int x) {
    return values[static_cast<size_t>(Reflect(y, height)) *
                      static_cast<size_t>(width) +
                  static_cast<size_t>(Reflect(x, width))];
  };
  // Sobel: central difference along one axis, [1, 2, 1] smoothing along the
  // other.
  std::vector<double> difference_x(pixels);
  std::vector<double> difference_y(pixels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(width) +
                           static_cast<size_t>(x);
      difference_x[pixel] = at(gray, y, x + 1) - at(gray, y, x - 1);
      difference_y[pixel] = at(gray, y + 1, x) - at(gray, y - 1, x);
    }
  }
  std::vector<double> magnitude(pixels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(width) +
                           static_cast<size_t>(x);
      const double gradient_x = at(difference_x, y - 1, x) +
                                kSobelCentreWeight * difference_x[pixel] +
                                at(difference_x, y + 1, x);
      const double gradient_y = at(difference_y, y, x - 1) +
                                kSobelCentreWeight * difference_y[pixel] +
                                at(difference_y, y, x + 1);
      magnitude[pixel] = std::hypot(gradient_x, gradient_y);
    }
  }
  const double scale =
      std::max(Percentile(magnitude, kEdgeScalePercentile), kMinEdgeScale);
  for (size_t pixel = 0; pixel < pixels; ++pixel) {
    out[pixel] = static_cast<float>(magnitude[pixel] / scale);
  }
  return out;
}

FrameLabeler::FrameLabeler(const Points& static_world,
                           const std::vector<int>& static_labels,
                           std::vector<SlotGeometry> slots,
                           std::vector<MarkingGeometry> markings,
                           const FrameLabelConfig& config)
    : config_(config),
      index_(static_world),
      labels_(static_labels),
      slots_(std::move(slots)),
      markings_(std::move(markings)) {
  GR_CHECK(labels_.size() == static_world.size());
  // Markings lie on the floor: use the floor height of the nearest slot.
  marking_z_.assign(markings_.size(), 0.0);
  for (size_t marking = 0; marking < markings_.size(); ++marking) {
    double best = kInfinity;
    for (const SlotGeometry& slot : slots_) {
      Vec2 corner_sum = slot.corners[0];
      for (int corner = 1; corner < kParkingSlotCorners; ++corner) {
        corner_sum += slot.corners[corner];
      }
      const Vec2 center =
          corner_sum / static_cast<double>(kParkingSlotCorners);
      const double distance = (center - markings_[marking].p0).norm();
      if (distance < best) {
        best = distance;
        marking_z_[marking] = slot.z;
      }
    }
  }
}

FrameLabels FrameLabeler::Label(const CameraModel& camera,
                                const CameraExposure& exposure,
                                const std::vector<ObjectBox>& boxes,
                                const ImageView& image) const {
  FrameLabels labels;
  labels.width = camera.width();
  labels.height = camera.height();
  const Vec3& origin = exposure.center();
  const double max_range = config_.max_range;

  ZBuffer buffer(camera, exposure, config_.render_spacing, kMinSplatRadius,
                 kMaxSplatRadius, kMinRenderRange,
                 max_range + kRenderRangeMargin);
  for (int point :
       index_.RadiusSearch(origin, max_range + kRenderRangeMargin)) {
    buffer.Splat(index_.points()[static_cast<size_t>(point)],
                 labels_[static_cast<size_t>(point)], point);
  }
  for (const ObjectBox& box : boxes) {
    if ((box.center - origin).norm() >= max_range + kBoxRangeMargin) continue;
    for (const Vec3& point : box.SampleSurface(config_.box_render_step)) {
      buffer.Splat(point, box.label, RenderResult::kEmpty);
    }
  }
  const RenderResult render = buffer.TakeResult();
  labels.label_image.resize(render.label.size());
  for (size_t pixel = 0; pixel < render.label.size(); ++pixel) {
    labels.label_image[pixel] = static_cast<uint8_t>(
        std::clamp(render.label[pixel], 0,
                   static_cast<int>(std::numeric_limits<uint8_t>::max())));
  }

  const Projector projector(camera, exposure, render.depth, config_);
  ProjectSlots(projector, labels);
  const std::vector<float> edges =
      image.empty() ? std::vector<float>() : EdgeStrength(image);
  ProjectMarkings(projector, edges, image, labels);
  ProjectBoxes(projector, boxes, labels);
  return labels;
}

void FrameLabeler::ProjectSlots(const Projector& projector,
                                FrameLabels& labels) const {
  for (size_t slot_index = 0; slot_index < slots_.size(); ++slot_index) {
    const SlotGeometry& geometry = slots_[slot_index];
    Points corners;
    for (const Vec2& corner : geometry.corners) {
      corners.emplace_back(corner.x(), corner.y(), geometry.z + kFloorLift);
    }
    if (projector.MinDistance(corners) > config_.max_range) continue;
    const PolylineSamples outline =
        SamplePolyline(corners, config_.sample_step, true);
    const Projector::Projection projection = projector.Project(outline.points);
    if (!AnyOf(projection.visible)) continue;
    SlotProjection slot;
    slot.index = static_cast<int>(slot_index);
    const Projector::Projection corner_projection = projector.Project(corners);
    for (int corner = 0; corner < kParkingSlotCorners; ++corner) {
      const Vec2 uv = corner_projection.uv.row(corner).transpose();
      slot.corners_uv[corner] =
          uv.allFinite() ? uv : Vec2::Constant(SlotProjection::kUnprojected);
      slot.corner_visible[corner] = corner_projection.visible[corner] != 0;
    }
    size_t entrance = 0;
    size_t entrance_visible = 0;
    for (size_t sample = 0; sample < outline.edge.size(); ++sample) {
      if (outline.edge[sample] != kEntranceEdge) continue;
      ++entrance;
      entrance_visible += projection.visible[sample];
    }
    slot.entrance_visible = entrance > 0
                                ? static_cast<double>(entrance_visible) /
                                      static_cast<double>(entrance)
                                : 0.0;
    slot.visible_fraction = MeanOf(projection.visible);
    slot.polylines = VisibleRuns(projection.uv, projection.visible);
    labels.slots.push_back(std::move(slot));
  }
}

void FrameLabeler::ProjectMarkings(const Projector& projector,
                                   const std::vector<float>& edges,
                                   const ImageView& image,
                                   FrameLabels& labels) const {
  std::vector<double> edge_scores;
  for (size_t marking_index = 0; marking_index < markings_.size();
       ++marking_index) {
    const MarkingGeometry& marking = markings_[marking_index];
    const double z = marking_z_[marking_index] + kFloorLift;
    const Points line{Vec3(marking.p0.x(), marking.p0.y(), z),
                      Vec3(marking.p1.x(), marking.p1.y(), z)};
    if (projector.MinDistance(line) > config_.max_range) continue;
    const PolylineSamples samples =
        SamplePolyline(line, config_.sample_step, false);
    const Projector::Projection projection = projector.Project(samples.points);
    if (!AnyOf(projection.visible)) continue;
    MarkingProjection projected;
    projected.index = static_cast<int>(marking_index);
    projected.polylines = VisibleRuns(projection.uv, projection.visible);
    labels.markings.push_back(std::move(projected));
    if (edges.empty()) continue;
    // Both paint edges must land on image edges.
    const Vec2 direction =
        (marking.p1 - marking.p0) /
        std::max((marking.p1 - marking.p0).norm(), kMinNorm);
    const Vec3 offset =
        Vec3(-direction.y(), direction.x(), 0.0) * (marking.width / 2.0);
    for (double sign : {-1.0, 1.0}) {
      Points shifted(samples.points.size());
      for (size_t sample = 0; sample < shifted.size(); ++sample) {
        shifted[sample] = samples.points[sample] + sign * offset;
      }
      const Projector::Projection edge_projection = projector.Project(shifted);
      double sum = 0.0;
      int count = 0;
      for (size_t sample = 0; sample < shifted.size(); ++sample) {
        const Eigen::Index row = static_cast<Eigen::Index>(sample);
        const double u = edge_projection.uv(row, 0);
        const double v = edge_projection.uv(row, 1);
        if (!edge_projection.visible[sample] || !std::isfinite(u) ||
            !std::isfinite(v)) {
          continue;
        }
        const int column =
            std::clamp(static_cast<int>(u), 0, image.width - 1);
        const int image_row =
            std::clamp(static_cast<int>(v), 0, image.height - 1);
        sum += edges[static_cast<size_t>(image_row) *
                         static_cast<size_t>(image.width) +
                     static_cast<size_t>(column)];
        ++count;
      }
      if (count >= kMinScoredSamples) edge_scores.push_back(sum / count);
    }
  }
  if (edge_scores.empty()) {
    labels.consistency_score = std::numeric_limits<double>::quiet_NaN();
  } else {
    double sum = 0.0;
    for (double score : edge_scores) sum += score;
    labels.consistency_score = sum / static_cast<double>(edge_scores.size());
  }
}

void FrameLabeler::ProjectBoxes(const Projector& projector,
                                const std::vector<ObjectBox>& boxes,
                                FrameLabels& labels) const {
  for (size_t box_index = 0; box_index < boxes.size(); ++box_index) {
    const ObjectBox& box = boxes[box_index];
    if ((box.center - projector.origin()).norm() > config_.max_range) continue;
    const Points surface = box.SampleSurface(config_.box_sample_step);
    const Projector::Projection projection = projector.Project(surface);
    if (!AnyOf(projection.visible)) continue;
    BoxProjection projected;
    projected.index = static_cast<int>(box_index);
    Eigen::Vector4d bbox(kInfinity, kInfinity, -kInfinity, -kInfinity);
    for (size_t sample = 0; sample < surface.size(); ++sample) {
      if (!projection.visible[sample]) continue;
      const double u = projection.uv(static_cast<Eigen::Index>(sample), 0);
      const double v = projection.uv(static_cast<Eigen::Index>(sample), 1);
      bbox[0] = std::min(bbox[0], u);
      bbox[1] = std::min(bbox[1], v);
      bbox[2] = std::max(bbox[2], u);
      bbox[3] = std::max(bbox[3], v);
    }
    projected.bbox = bbox;
    projected.visible_fraction = MeanOf(projection.visible);
    labels.boxes.push_back(projected);
  }
}

LabelLifter::LabelLifter(const Points& points, int num_classes, double spacing,
                         double max_range, int boundary, int min_views)
    : index_(points),
      num_classes_(num_classes),
      spacing_(spacing),
      max_range_(max_range),
      boundary_(boundary),
      min_views_(min_views),
      votes_(points.size() * static_cast<size_t>(std::max(num_classes, 0)),
             0.0f),
      num_views_(points.size(), 0) {
  GR_CHECK(num_classes > 0);
}

void LabelLifter::AddView(const CameraModel& camera,
                          const CameraExposure& exposure, const int32_t* labels,
                          const float* weights) {
  const Vec3& origin = exposure.center();
  ZBuffer buffer(camera, exposure, spacing_, kMinSplatRadius,
                 kMaxLiftSplatRadius, kMinRenderRange, max_range_);
  for (int point : index_.RadiusSearch(origin, max_range_ + kLiftRangeSlack)) {
    buffer.Splat(index_.points()[static_cast<size_t>(point)],
                 RenderResult::kEmpty, point);
  }
  const RenderResult& render = buffer.result();
  const int height = render.height;
  const int width = render.width;
  const auto label_at = [&](int y, int x) {
    return labels[static_cast<size_t>(Reflect(y, height)) *
                      static_cast<size_t>(width) +
                  static_cast<size_t>(Reflect(x, width))];
  };
  std::vector<int> touched;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(width) +
                           static_cast<size_t>(x);
      const int point = render.index[pixel];
      const int label = labels[pixel];
      if (point == RenderResult::kEmpty || label < 0 ||
          label >= num_classes_) {
        continue;
      }
      bool uniform = true;  // Skip pixels on label boundaries.
      for (int row_offset = -boundary_; row_offset <= boundary_ && uniform;
           ++row_offset) {
        for (int column_offset = -boundary_;
             column_offset <= boundary_ && uniform; ++column_offset) {
          uniform = label_at(y + row_offset, x + column_offset) == label;
        }
      }
      if (!uniform) continue;
      votes_[static_cast<size_t>(point) * static_cast<size_t>(num_classes_) +
             static_cast<size_t>(label)] +=
          weights != nullptr ? weights[pixel] : 1.0f;
      touched.push_back(point);
    }
  }
  std::sort(touched.begin(), touched.end());
  touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
  for (int point : touched) ++num_views_[static_cast<size_t>(point)];
}

LabelLifter::Lifted LabelLifter::Lift() const {
  const size_t points = num_views_.size();
  Lifted lifted;
  lifted.labels.assign(points, kNoLabel);
  lifted.confidence.assign(points, 0.0);
  for (size_t point = 0; point < points; ++point) {
    const float* point_votes =
        &votes_[point * static_cast<size_t>(num_classes_)];
    float total = 0.0f;
    int best = 0;
    for (int class_id = 0; class_id < num_classes_; ++class_id) {
      total += point_votes[class_id];
      if (point_votes[class_id] > point_votes[best]) best = class_id;
    }
    if (total <= 0.0f) continue;
    lifted.confidence[point] = static_cast<double>(point_votes[best]) /
                               static_cast<double>(total);
    if (num_views_[point] >= min_views_) lifted.labels[point] = best;
  }
  return lifted;
}

}  // namespace gr
