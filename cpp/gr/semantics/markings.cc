#include "gr/semantics/markings.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Eigenvalues>

#include "gr/core/check.h"
#include "gr/core/morphology.h"

namespace gr {
namespace {

// Directions of vectors shorter than this have no meaning; weights below it
// do not divide.
constexpr double kMinNorm = 1e-9;
// Confidences are at most this.
constexpr double kMaxConfidence = 1.0;
// Skeleton cells with at least this many skeleton neighbours (of the 8
// within kNeighbourhoodRadius) are junctions.
constexpr int kNeighbourhoodRadius = 1;
constexpr int kMinJunctionNeighbours = 3;
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;
// Branches shorter than this share of the minimum marking length are noise.
constexpr double kMinBranchShare = 0.5;
// Branches whose cells lie farther than this (cells) from their line on
// average are curved (arrow heads, text).
constexpr double kMaxMeanResidual = 2.0;
// The eigenvector of the larger eigenvalue (Eigen sorts them ascending).
constexpr int kMajorAxis = 1;
// The distance to the background is half the width on the centre line.
constexpr double kWidthPerDepth = 2.0;
// Each side is sampled kSideSamples times over the middle kSampleSpan of the
// line, starting kSampleStart in.
constexpr int kSideSamples = 20;
constexpr double kSampleStart = 0.1;
constexpr double kSampleSpan = 0.8;
// Lane markings run within this angle (deg) of the path, measured over
// kTangentHalfWindow samples to both sides of the closest one (needing
// kMinPathSamples), whose chord is at least kMinTangentLength (m) long.
constexpr double kLaneMaxAngle = 10.0;
constexpr size_t kMinPathSamples = 3;
constexpr size_t kTangentHalfWindow = 5;
constexpr double kMinTangentLength = 0.5;

Vec2 Unit(const Vec2& vector) {
  return vector / std::max(vector.norm(), kMinNorm);
}

// The centre of a row-major cell, in cells.
Vec2 CellCentre(int cell, int width) {
  return Vec2(cell % width + kCellCentre, cell / width + kCellCentre);
}

}  // namespace

std::vector<MarkingSegment> MergeCollinear(
    std::vector<MarkingSegment> segments, const MarkingConfig& config) {
  const size_t count = segments.size();
  if (count < 2) return segments;
  std::vector<uint8_t> alive(count, 1);
  const double cos_max = std::cos(DegToRad(config.merge_angle));
  for (size_t base = 0; base < count; ++base) {
    while (alive[base]) {
      const Vec2 direction = Unit(segments[base].p1 - segments[base].p0);
      const Vec2 normal(-direction.y(), direction.x());
      const double base_length = (segments[base].p1 - segments[base].p0).norm();
      size_t partner = count;
      double partner_start = 0.0;  // Along the base, from its p0.
      double partner_end = 0.0;
      for (size_t other = 0; other < count; ++other) {
        if (other == base || !alive[other]) continue;
        const MarkingSegment& candidate = segments[other];
        const Vec2 candidate_direction = Unit(candidate.p1 - candidate.p0);
        if (std::abs(candidate_direction.dot(direction)) < cos_max) continue;
        if (std::max(
                std::abs((candidate.p0 - segments[base].p0).dot(normal)),
                std::abs((candidate.p1 - segments[base].p0).dot(normal))) >
            config.merge_offset) {
          continue;
        }
        const double start = (candidate.p0 - segments[base].p0).dot(direction);
        const double end = (candidate.p1 - segments[base].p0).dot(direction);
        const double gap = std::max(std::min(start, end) - base_length,
                                    -std::max(start, end));
        if (gap > config.merge_gap) continue;
        partner = other;
        partner_start = start;
        partner_end = end;
        break;
      }
      if (partner == count) break;
      const MarkingSegment& kept = segments[base];
      const MarkingSegment& absorbed = segments[partner];
      const double start = std::min({0.0, partner_start, partner_end});
      const double end = std::max({base_length, partner_start, partner_end});
      const double absorbed_length = absorbed.length();
      const double total = std::max(base_length + absorbed_length, kMinNorm);
      MarkingSegment merged;
      merged.level = kept.level;
      merged.p0 = kept.p0 + start * direction;
      merged.p1 = kept.p0 + end * direction;
      merged.width =
          (kept.width * base_length + absorbed.width * absorbed_length) /
          total;
      merged.lane = kept.lane;
      merged.confidence = std::min(
          kMaxConfidence, (kept.confidence * base_length +
                           absorbed.confidence * absorbed_length) /
                              total);
      segments[base] = merged;
      alive[partner] = 0;
    }
  }
  std::vector<MarkingSegment> out;
  for (size_t i = 0; i < count; ++i) {
    if (alive[i]) out.push_back(segments[i]);
  }
  return out;
}

std::vector<MarkingSegment> VectorizeMarkings(
    const std::vector<uint8_t>& mask, const RasterFrame& frame,
    const MarkingConfig& config, int level,
    const std::vector<uint8_t>& given_skeleton) {
  const int width = frame.width;
  const int height = frame.height;
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  GR_CHECK(given_skeleton.empty() || given_skeleton.size() == mask.size());
  const double res = frame.res;
  const Mask skeleton = given_skeleton.empty()
                            ? Skeletonize(mask, width, height)
                            : given_skeleton;
  // Junctions: skeleton cells with three or more skeleton neighbours; the
  // branches are what is left away from them.
  Mask junction(skeleton.size(), 0);
  for (int row = 0; row < height; ++row) {
    for (int column = 0; column < width; ++column) {
      if (!skeleton[static_cast<size_t>(row * width + column)]) continue;
      int neighbours = 0;
      for (int row_offset = -kNeighbourhoodRadius;
           row_offset <= kNeighbourhoodRadius; ++row_offset) {
        for (int column_offset = -kNeighbourhoodRadius;
             column_offset <= kNeighbourhoodRadius; ++column_offset) {
          if (column_offset == 0 && row_offset == 0) continue;
          const int neighbour_column = column + column_offset;
          const int neighbour_row = row + row_offset;
          neighbours +=
              neighbour_column >= 0 && neighbour_row >= 0 &&
              neighbour_column < width && neighbour_row < height &&
              skeleton[static_cast<size_t>(neighbour_row * width +
                                           neighbour_column)];
        }
      }
      junction[static_cast<size_t>(row * width + column)] =
          neighbours >= kMinJunctionNeighbours;
    }
  }
  const Mask near_junction = DilateCross(junction, width, height);
  Mask branches(skeleton.size(), 0);
  for (size_t cell = 0; cell < branches.size(); ++cell) {
    branches[cell] = skeleton[cell] && !near_junction[cell];
  }
  Mask background(mask.size());
  for (size_t cell = 0; cell < mask.size(); ++cell) {
    background[cell] = mask[cell] == 0;
  }
  const std::vector<float> distance =
      DistanceTransform(background, width, height);
  std::vector<MarkingSegment> segments;
  for (const std::vector<int>& cells : Label8(branches, width, height).cells) {
    if (static_cast<double>(cells.size()) * res <
        config.min_length * kMinBranchShare) {
      continue;
    }
    Vec2 mean = Vec2::Zero();
    for (int cell : cells) mean += CellCentre(cell, width);
    mean /= static_cast<double>(cells.size());
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (int cell : cells) {
      const Vec2 offset = CellCentre(cell, width) - mean;
      scatter += offset * offset.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(scatter);
    Vec2 direction = solver.eigenvectors().col(kMajorAxis);
    // A deterministic orientation: the larger component positive.
    if ((std::abs(direction.x()) >= std::abs(direction.y()) ? direction.x()
                                                            : direction.y()) <
        0.0) {
      direction = -direction;
    }
    const Vec2 normal(-direction.y(), direction.x());
    double along_min = std::numeric_limits<double>::infinity();
    double along_max = -along_min;
    double residual = 0.0;
    double depth = 0.0;
    for (int cell : cells) {
      const Vec2 offset = CellCentre(cell, width) - mean;
      const double along = offset.dot(direction);
      along_min = std::min(along_min, along);
      along_max = std::max(along_max, along);
      residual += std::abs(offset.dot(normal));
      depth += distance[static_cast<size_t>(cell)];
    }
    if (residual / static_cast<double>(cells.size()) > kMaxMeanResidual) {
      continue;  // A curved branch (arrow head, text).
    }
    MarkingSegment segment;
    segment.level = level;
    const Vec2 origin(frame.x0, frame.y0);
    segment.p0 = origin + res * (mean + along_min * direction);
    segment.p1 = origin + res * (mean + along_max * direction);
    segment.width =
        kWidthPerDepth * depth / static_cast<double>(cells.size()) * res;
    segments.push_back(segment);
  }
  segments = MergeCollinear(std::move(segments), config);
  std::vector<MarkingSegment> out;
  for (const MarkingSegment& segment : segments) {
    if (segment.length() >= config.min_length &&
        segment.width >= config.min_width &&
        segment.width <= config.max_width) {
      out.push_back(segment);
    }
  }
  return out;
}

bool BothSidesObserved(const MarkingSegment& segment,
                       const std::vector<uint8_t>& floor,
                       const RasterFrame& frame, double offset,
                       double min_fraction) {
  const Vec2 direction = Unit(segment.p1 - segment.p0);
  const Vec2 side_offset =
      Vec2(-direction.y(), direction.x()) * (segment.width / 2.0 + offset);
  for (double side : {1.0, -1.0}) {
    int inside = 0;
    int observed = 0;
    for (int sample = 0; sample < kSideSamples; ++sample) {
      const double fraction_along =
          kSampleStart + kSampleSpan * sample / (kSideSamples - 1);
      const Vec2 point = segment.p0 +
                         fraction_along * (segment.p1 - segment.p0) +
                         side * side_offset;
      const int column =
          static_cast<int>(std::floor((point.x() - frame.x0) / frame.res));
      const int row =
          static_cast<int>(std::floor((point.y() - frame.y0) / frame.res));
      if (column < 0 || row < 0 || column >= frame.width ||
          row >= frame.height) {
        continue;
      }
      ++inside;
      observed += floor[static_cast<size_t>(row * frame.width + column)] != 0;
    }
    const double fraction =
        inside > 0 ? static_cast<double>(observed) / inside : 0.0;
    if (fraction < min_fraction) return false;
  }
  return true;
}

bool GridAligned(const MarkingSegment& segment, double yaw,
                 double max_angle_deg) {
  const Vec2 direction = segment.p1 - segment.p0;
  const double quarter = kPi / 2.0;
  double angle = std::atan2(direction.y(), direction.x()) - yaw;
  // Python's modulo: [0, pi / 2).
  angle -= std::floor(angle / quarter) * quarter;
  return std::min(angle, quarter - angle) <= DegToRad(max_angle_deg);
}

int RefineSegments(std::vector<MarkingSegment>& segments,
                   const std::vector<const FloorImage*>& images,
                   const LineFitConfig& config) {
  int refined = 0;
  for (MarkingSegment& segment : segments) {
    std::optional<LineFit> fit;
    for (const FloorImage* image : images) {
      fit = FitLine(segment.p0, segment.p1, *image, config);
      if (fit) break;
    }
    if (!fit) continue;
    segment.p0 = fit->p0;
    segment.p1 = fit->p1;
    segment.width = fit->width;
    for (int end = 0; end < kSegmentEnds; ++end) {
      segment.end_ok[end] = fit->end_ok[end];
    }
    ++refined;
  }
  return refined;
}

void ClassifyMarkings(std::vector<MarkingSegment>& segments,
                      const std::map<int, std::vector<Vec2>>& paths,
                      const MarkingConfig& config) {
  const double cos_max = std::cos(DegToRad(kLaneMaxAngle));
  for (MarkingSegment& segment : segments) {
    segment.lane = false;
    const auto found = paths.find(segment.level);
    if (found == paths.end() || found->second.size() < kMinPathSamples ||
        segment.length() < config.lane_min_length) {
      continue;
    }
    const std::vector<Vec2>& path = found->second;
    const Vec2 middle = 0.5 * (segment.p0 + segment.p1);
    size_t closest = 0;
    for (size_t i = 1; i < path.size(); ++i) {
      if ((path[i] - middle).norm() < (path[closest] - middle).norm()) {
        closest = i;
      }
    }
    const Vec2 tangent =
        path[std::min(path.size() - 1, closest + kTangentHalfWindow)] -
        path[closest >= kTangentHalfWindow ? closest - kTangentHalfWindow
                                           : 0];
    const Vec2 direction = segment.p1 - segment.p0;
    if (tangent.norm() > kMinTangentLength &&
        std::abs(tangent.dot(direction)) /
                (tangent.norm() * direction.norm()) >
            cos_max) {
      segment.lane = true;
    }
  }
}

}  // namespace gr
