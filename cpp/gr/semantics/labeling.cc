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
  for (size_t k = 0; k < num_edges; ++k) {
    const Vec3& a = vertices[k];
    const Vec3& b = vertices[(k + 1) % num_vertices];
    const int n =
        std::max(2, static_cast<int>(std::ceil((b - a).norm() / step)) + 1);
    const double dt = 1.0 / (n - 1);
    for (int i = 0; i < n; ++i) {
      const double t = i == n - 1 ? 1.0 : i * dt;
      samples.points.push_back(a + t * (b - a));
      samples.edge.push_back(static_cast<int>(k));
    }
  }
  return samples;
}

// Runs of consecutive visible samples (at least two points each).
std::vector<Polyline2> VisibleRuns(const RowMatX2d& uv,
                                   const std::vector<uint8_t>& visible) {
  std::vector<Polyline2> runs;
  Polyline2 current;
  for (size_t m = 0; m < visible.size(); ++m) {
    const Eigen::Index row = static_cast<Eigen::Index>(m);
    if (visible[m]) {
      current.emplace_back(uv(row, 0), uv(row, 1));
    } else if (!current.empty()) {
      if (current.size() > 1) runs.push_back(current);
      current.clear();
    }
  }
  if (current.size() > 1) runs.push_back(current);
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
int Reflect(int i, int n) {
  if (i < 0) i = -i - 1;
  if (i >= n) i = 2 * n - i - 1;
  return std::clamp(i, 0, n - 1);
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
    for (size_t m = 0; m < samples.size(); ++m) {
      projection.visible[m] = result.in_image[m] && result.visible[m] &&
                              result.range[m] <= config_.max_range;
    }
    projection.uv = std::move(result.uv);
    return projection;
  }

  // Smallest distance from the camera centre to any of the points.
  double MinDistance(const Points& points) const {
    double d = kInfinity;
    for (const Vec3& p : points) d = std::min(d, (p - origin_).norm());
    return d;
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
  const size_t n = static_cast<size_t>(std::max(height, 0)) *
                   static_cast<size_t>(std::max(width, 0));
  std::vector<float> out(n, 0.0f);
  if (image.empty() || image.channels < 1) return out;
  std::vector<double> gray(n);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const uint8_t* px = image.PixelAt(y, x);
      double sum = 0.0;
      for (int c = 0; c < image.channels; ++c) sum += px[c];
      gray[static_cast<size_t>(y) * static_cast<size_t>(width) +
           static_cast<size_t>(x)] = sum / image.channels;
    }
  }
  const auto at = [&](const std::vector<double>& a, int y, int x) {
    return a[static_cast<size_t>(Reflect(y, height)) *
                 static_cast<size_t>(width) +
             static_cast<size_t>(Reflect(x, width))];
  };
  // Sobel: central difference along one axis, [1, 2, 1] smoothing along the
  // other.
  std::vector<double> dx(n);
  std::vector<double> dy(n);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t p = static_cast<size_t>(y) * static_cast<size_t>(width) +
                       static_cast<size_t>(x);
      dx[p] = at(gray, y, x + 1) - at(gray, y, x - 1);
      dy[p] = at(gray, y + 1, x) - at(gray, y - 1, x);
    }
  }
  std::vector<double> magnitude(n);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t p = static_cast<size_t>(y) * static_cast<size_t>(width) +
                       static_cast<size_t>(x);
      const double sx = at(dx, y - 1, x) + 2.0 * dx[p] + at(dx, y + 1, x);
      const double sy = at(dy, y, x - 1) + 2.0 * dy[p] + at(dy, y, x + 1);
      magnitude[p] = std::hypot(sx, sy);
    }
  }
  const double scale = std::max(Percentile(magnitude, 95.0), 1e-6);
  for (size_t p = 0; p < n; ++p) {
    out[p] = static_cast<float>(magnitude[p] / scale);
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
  for (size_t k = 0; k < markings_.size(); ++k) {
    double best = kInfinity;
    for (const SlotGeometry& slot : slots_) {
      const Vec2 center = 0.25 * (slot.corners[0] + slot.corners[1] +
                                  slot.corners[2] + slot.corners[3]);
      const double d = (center - markings_[k].p0).norm();
      if (d < best) {
        best = d;
        marking_z_[k] = slot.z;
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

  ZBuffer buffer(camera, exposure, config_.render_spacing, 1, 4, 0.1,
                 max_range + 2.0);
  for (int i : index_.RadiusSearch(origin, max_range + 2.0)) {
    buffer.Splat(index_.points()[static_cast<size_t>(i)],
                 labels_[static_cast<size_t>(i)], i);
  }
  for (const ObjectBox& box : boxes) {
    if ((box.center - origin).norm() >= max_range + 5.0) continue;
    for (const Vec3& p : box.SampleSurface(config_.box_render_step)) {
      buffer.Splat(p, box.label, -1);
    }
  }
  const RenderResult render = buffer.TakeResult();
  labels.label_image.resize(render.label.size());
  for (size_t p = 0; p < render.label.size(); ++p) {
    labels.label_image[p] =
        static_cast<uint8_t>(std::clamp(render.label[p], 0, 255));
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
  for (size_t s = 0; s < slots_.size(); ++s) {
    Points corners;
    for (const Vec2& c : slots_[s].corners) {
      corners.emplace_back(c.x(), c.y(), slots_[s].z + 0.01);
    }
    if (projector.MinDistance(corners) > config_.max_range) continue;
    const PolylineSamples outline =
        SamplePolyline(corners, config_.sample_step, true);
    const Projector::Projection projection = projector.Project(outline.points);
    if (!AnyOf(projection.visible)) continue;
    SlotProjection slot;
    slot.index = static_cast<int>(s);
    const Projector::Projection corner_projection = projector.Project(corners);
    for (size_t a = 0; a < 4; ++a) {
      const Vec2 uv =
          corner_projection.uv.row(static_cast<Eigen::Index>(a)).transpose();
      slot.corners_uv[a] = uv.allFinite() ? uv : Vec2(-1.0, -1.0);
      slot.corner_visible[a] = corner_projection.visible[a] != 0;
    }
    size_t entrance = 0;
    size_t entrance_visible = 0;
    for (size_t m = 0; m < outline.edge.size(); ++m) {
      if (outline.edge[m] != 0) continue;
      ++entrance;
      entrance_visible += projection.visible[m];
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
  for (size_t k = 0; k < markings_.size(); ++k) {
    const MarkingGeometry& marking = markings_[k];
    const double z = marking_z_[k] + 0.01;
    const Points line{Vec3(marking.p0.x(), marking.p0.y(), z),
                      Vec3(marking.p1.x(), marking.p1.y(), z)};
    if (projector.MinDistance(line) > config_.max_range) continue;
    const PolylineSamples samples =
        SamplePolyline(line, config_.sample_step, false);
    const Projector::Projection projection = projector.Project(samples.points);
    if (!AnyOf(projection.visible)) continue;
    MarkingProjection projected;
    projected.index = static_cast<int>(k);
    projected.polylines = VisibleRuns(projection.uv, projection.visible);
    labels.markings.push_back(std::move(projected));
    if (edges.empty()) continue;
    // Both paint edges must land on image edges.
    const Vec2 direction = (marking.p1 - marking.p0) /
                           std::max((marking.p1 - marking.p0).norm(), 1e-9);
    const Vec3 offset =
        Vec3(-direction.y(), direction.x(), 0.0) * (marking.width / 2.0);
    for (double sign : {-1.0, 1.0}) {
      Points shifted(samples.points.size());
      for (size_t a = 0; a < shifted.size(); ++a) {
        shifted[a] = samples.points[a] + sign * offset;
      }
      const Projector::Projection edge_projection = projector.Project(shifted);
      double sum = 0.0;
      int count = 0;
      for (size_t a = 0; a < shifted.size(); ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(a);
        const double u = edge_projection.uv(row, 0);
        const double v = edge_projection.uv(row, 1);
        if (!edge_projection.visible[a] || !std::isfinite(u) ||
            !std::isfinite(v)) {
          continue;
        }
        const int col = std::clamp(static_cast<int>(u), 0, image.width - 1);
        const int r = std::clamp(static_cast<int>(v), 0, image.height - 1);
        sum += edges[static_cast<size_t>(r) * static_cast<size_t>(image.width) +
                     static_cast<size_t>(col)];
        ++count;
      }
      if (count >= 5) edge_scores.push_back(sum / count);
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
  for (size_t b = 0; b < boxes.size(); ++b) {
    const ObjectBox& box = boxes[b];
    if ((box.center - projector.origin()).norm() > config_.max_range) continue;
    const Points surface = box.SampleSurface(config_.box_sample_step);
    const Projector::Projection projection = projector.Project(surface);
    if (!AnyOf(projection.visible)) continue;
    BoxProjection projected;
    projected.index = static_cast<int>(b);
    Eigen::Vector4d bbox(kInfinity, kInfinity, -kInfinity, -kInfinity);
    for (size_t a = 0; a < surface.size(); ++a) {
      if (!projection.visible[a]) continue;
      const double u = projection.uv(static_cast<Eigen::Index>(a), 0);
      const double v = projection.uv(static_cast<Eigen::Index>(a), 1);
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
  ZBuffer buffer(camera, exposure, spacing_, 1, 3, 0.1, max_range_);
  for (int i : index_.RadiusSearch(origin, max_range_ + 1e-6)) {
    buffer.Splat(index_.points()[static_cast<size_t>(i)], -1, i);
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
      if (point < 0 || label < 0 || label >= num_classes_) continue;
      bool uniform = true;  // Skip pixels on label boundaries.
      for (int dy = -boundary_; dy <= boundary_ && uniform; ++dy) {
        for (int dx = -boundary_; dx <= boundary_ && uniform; ++dx) {
          uniform = label_at(y + dy, x + dx) == label;
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
  const size_t n = num_views_.size();
  Lifted lifted;
  lifted.labels.assign(n, -1);
  lifted.confidence.assign(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const float* v = &votes_[i * static_cast<size_t>(num_classes_)];
    float total = 0.0f;
    int best = 0;
    for (int c = 0; c < num_classes_; ++c) {
      total += v[c];
      if (v[c] > v[best]) best = c;
    }
    if (total <= 0.0f) continue;
    lifted.confidence[i] =
        static_cast<double>(v[best]) / static_cast<double>(total);
    if (num_views_[i] >= min_views_) lifted.labels[i] = best;
  }
  return lifted;
}

}  // namespace gr
