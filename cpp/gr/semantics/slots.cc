#include "gr/semantics/slots.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/geometry.h"
#include "gr/core/morphology.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
// Orientations of a row frame: separators along its y (slots side by side
// along x) or along its x.
constexpr int kSeparatorsAlongY = 1;
constexpr int kSeparatorsAlongX = 0;

// Separators: at most this many slot depths long (back-to-back rows share
// one line) plus kSeparatorSlack (m).
constexpr double kMaxSeparatorDepths = 2.0;
constexpr double kSeparatorSlack = 0.5;
// Overlaps over bases shorter than this (m) do not divide.
constexpr double kMinOverlapBase = 1e-6;
// Row tilts: Theil-Sen slopes over separator pairs at least kMinTiltBaseline
// (m) apart, at least kMinTiltSlopes of them; the ends must move by at least
// kMinTiltChange (m) and kTiltSigmas robust sigmas (MAD x kMadToSigma).
constexpr double kMinTiltBaseline = 2.0;
constexpr size_t kMinTiltSlopes = 3;
constexpr double kMinTiltChange = 0.15;
constexpr double kTiltSigmas = 4.0;
constexpr double kMadToSigma = 1.4826;
// Rows without their own tilt take the median of at least this many others.
constexpr size_t kMinSignificantTilts = 2;
// Row ends: the measured separator ends within this (m) of the extreme.
constexpr double kEndNear = 0.15;
// Hypothetical separators are sampled every kLineSampleStep (m), at least
// kMinLineSamples times, kLineEndInset (m) in from both ends; their paint
// ridge is taken against lines kRidgeSideOffset (m) to both sides.
constexpr double kLineSampleStep = 0.1;
constexpr int kMinLineSamples = 2;
constexpr double kLineEndInset = 0.3;
constexpr double kRidgeSideOffset = 0.35;
// Back-to-back rows: the separator ends' kEndPercentileLow / High
// percentiles span more than a slot depth plus kBackToBackSlack (m); a single
// row's ends are clipped kEndClipMargin (m) beyond them; split halves keep
// separators reaching kSplitMargin (m) past the middle.
constexpr double kEndPercentileLow = 10.0;
constexpr double kEndPercentileHigh = 90.0;
constexpr double kBackToBackSlack = 0.5;
constexpr double kEndClipMargin = 0.3;
constexpr double kSplitMargin = 0.5;
// Columns within this (m) of a row's extent bound its bays; their faces are
// separators unless one lies within kVirtualSeparatorGap (m).
constexpr double kColumnRowMargin = 1.0;
constexpr double kVirtualSeparatorGap = 0.1;
// Slot width (m) without any measured gap.
constexpr double kDefaultSlotWidth = 2.5;
// A row as long as a slot within this (m) has its hidden end at the slot
// depth; one shorter by more than kShortRowSlack (m) is extended.
constexpr double kDepthMatchTolerance = 0.4;
constexpr double kShortRowSlack = 0.25;
// Gaps of at least this many slot widths are split into inferred slots.
constexpr int kMinInferredSlots = 2;
// Slot outlines are inset by this (m) for the driven-through test and the
// occupancy.
constexpr double kSlotInset = 0.3;
// Confidence of inferred and painted slots.
constexpr double kInferredConfidence = 0.6;
constexpr double kPaintedConfidence = 1.0;
// Painted slots overlapping a kept one by this area share are duplicates.
constexpr double kMaxPaintedOverlap = 0.5;

Eigen::Matrix2d Rot2(double angle) {
  Eigen::Matrix2d rotation;
  rotation << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  return rotation;
}

// A slot boundary in a row frame: across position, along extent.
struct Separator {
  static constexpr int kNoSegment = -1;
  double across = 0.0;
  double along_start = 0.0;
  double along_end = 0.0;
  bool is_virtual = false;  // Column face, no paint.
  // along_start / along_end measured with floor observed beyond.
  bool start_measured = false;
  bool end_measured = false;
  int segment = kNoSegment;  // Marking segment it comes from.
};

using Row = std::vector<Separator>;

// Separators of the given segments in the frame rotated by row_yaw;
// `orientation`: kSeparatorsAlongY or kSeparatorsAlongX.
Row Project(const std::vector<MarkingSegment>& segments,
            const std::vector<int>& indices, int orientation, double row_yaw,
            const SlotConfig& config) {
  const Eigen::Matrix2d to_row = Rot2(-row_yaw);
  const Vec2 axis =
      orientation == kSeparatorsAlongY ? Vec2(0.0, 1.0) : Vec2(1.0, 0.0);
  Row out;
  for (int index : indices) {
    const MarkingSegment& segment = segments[static_cast<size_t>(index)];
    const Vec2 p0 = to_row * segment.p0;
    const Vec2 p1 = to_row * segment.p1;
    const Vec2 direction = p1 - p0;
    const double length = direction.norm();
    if (length < config.min_separator_length ||
        length > kMaxSeparatorDepths * config.depth_max + kSeparatorSlack) {
      continue;
    }
    const double angle = RadToDeg(
        std::acos(std::min(1.0, std::abs(direction.dot(axis)) / length)));
    if (angle > config.angle_tol) continue;
    Separator separator;
    separator.across = 0.5 * (p0(1 - orientation) + p1(1 - orientation));
    separator.segment = index;
    if (p0(orientation) <= p1(orientation)) {
      separator.along_start = p0(orientation);
      separator.along_end = p1(orientation);
      separator.start_measured = segment.end_ok[0];
      separator.end_measured = segment.end_ok[1];
    } else {
      separator.along_start = p1(orientation);
      separator.along_end = p0(orientation);
      separator.start_measured = segment.end_ok[1];
      separator.end_measured = segment.end_ok[0];
    }
    out.push_back(separator);
  }
  return out;
}

// The along extent of a row's separators.
std::pair<double, double> AlongExtent(const Row& row) {
  double start = row[0].along_start;
  double end = row[0].along_end;
  for (const Separator& separator : row) {
    start = std::min(start, separator.along_start);
    end = std::max(end, separator.along_end);
  }
  return {start, end};
}

// Groups separators whose along-intervals overlap.
std::vector<Row> Rows(Row separators, double overlap) {
  std::stable_sort(separators.begin(), separators.end(),
                   [](const Separator& a, const Separator& b) {
                     return a.along_start + a.along_end <
                            b.along_start + b.along_end;
                   });
  std::vector<Row> rows;
  for (const Separator& separator : separators) {
    bool placed = false;
    for (Row& row : rows) {
      const auto [row_start, row_end] = AlongExtent(row);
      const double shared = std::min(row_end, separator.along_end) -
                            std::max(row_start, separator.along_start);
      if (shared / std::max(std::min(row_end - row_start,
                                     separator.along_end -
                                         separator.along_start),
                            kMinOverlapBase) >=
          overlap) {
        row.push_back(separator);
        placed = true;
        break;
      }
    }
    if (!placed) rows.push_back({separator});
  }
  return rows;
}

// Angle (rad) of a row's entrance / back lines against the frame: the
// Theil-Sen slope of the measured separator ends over their across
// position, kept only if the ends move enough across the row (end noise
// over short rows would otherwise rotate aligned rows).
double RowTilt(const Row& separators) {
  // (across, along) of the measured start ends, then of the end ends.
  std::vector<std::vector<std::pair<double, double>>> sets(2);
  for (const Separator& separator : separators) {
    if (separator.start_measured) {
      sets[0].emplace_back(separator.across, separator.along_start);
    }
    if (separator.end_measured) {
      sets[1].emplace_back(separator.across, separator.along_end);
    }
  }
  std::vector<double> slopes;
  for (const auto& ends : sets) {
    for (size_t i = 0; i < ends.size(); ++i) {
      for (size_t j = i + 1; j < ends.size(); ++j) {
        const double baseline = ends[j].first - ends[i].first;
        if (std::abs(baseline) >= kMinTiltBaseline) {
          slopes.push_back((ends[j].second - ends[i].second) / baseline);
        }
      }
    }
  }
  if (slopes.size() < kMinTiltSlopes) return 0.0;
  const double slope = Median(slopes);
  std::vector<double> residuals;
  std::vector<double> acrosses;
  for (const auto& ends : sets) {
    if (ends.empty()) continue;
    std::vector<double> offsets;
    for (const auto& [across, along] : ends) {
      offsets.push_back(along - slope * across);
      acrosses.push_back(across);
    }
    const double median = Median(offsets);
    for (double offset : offsets) residuals.push_back(offset - median);
  }
  std::vector<double> absolute;
  for (double residual : residuals) absolute.push_back(std::abs(residual));
  const double sigma = kMadToSigma * Median(absolute);
  const double change =
      std::abs(slope) * (*std::max_element(acrosses.begin(), acrosses.end()) -
                         *std::min_element(acrosses.begin(), acrosses.end()));
  if (change < std::max(kMinTiltChange, kTiltSigmas * sigma)) return 0.0;
  return std::atan(slope);
}

struct RowEnds {
  double start = 0.0;
  double end = 0.0;
  bool start_measured = false;
  bool end_measured = false;
};

// Along extent of a row and whether each end was measured: rows share their
// entrance and back lines, so each end is the median of the measured
// separator ends near the row's extreme; without any, the paint envelope.
RowEnds EndsOf(const Row& painted) {
  const auto [envelope_start, envelope_end] = AlongExtent(painted);
  std::vector<double> starts;
  std::vector<double> ends;
  for (const Separator& separator : painted) {
    if (separator.start_measured &&
        separator.along_start < envelope_start + kEndNear) {
      starts.push_back(separator.along_start);
    }
    if (separator.end_measured &&
        separator.along_end > envelope_end - kEndNear) {
      ends.push_back(separator.along_end);
    }
  }
  RowEnds out;
  out.start = starts.empty() ? envelope_start : Median(starts);
  out.end = ends.empty() ? envelope_end : Median(ends);
  out.start_measured = !starts.empty();
  out.end_measured = !ends.empty();
  return out;
}

Vec2 PointAt(double across, double along, int orientation) {
  return orientation == kSeparatorsAlongY ? Vec2(across, along)
                                          : Vec2(along, across);
}

// numpy.linspace.
std::vector<double> Linspace(double start, double stop, int count) {
  std::vector<double> out(static_cast<size_t>(count));
  const double step = count > 1 ? (stop - start) / (count - 1) : 0.0;
  for (int k = 0; k < count; ++k) {
    out[static_cast<size_t>(k)] = start + k * step;
  }
  if (count > 1) out.back() = stop;
  return out;
}

// Cells of a line at `across`, from `along_start` to `along_end` (row frame),
// inset from both ends, inside the raster; also the number of samples.
std::pair<std::vector<size_t>, int> LineCells(double across,
                                              double along_start,
                                              double along_end,
                                              int orientation,
                                              const Eigen::Matrix2d& to_world,
                                              const RasterFrame& frame) {
  const int samples = std::max(
      static_cast<int>((along_end - along_start) / kLineSampleStep),
      kMinLineSamples);
  std::vector<size_t> cells;
  for (double along : Linspace(along_start + kLineEndInset,
                               along_end - kLineEndInset, samples)) {
    const Vec2 world = to_world * PointAt(across, along, orientation);
    const int column =
        static_cast<int>(std::floor((world.x() - frame.x0) / frame.res));
    const int row =
        static_cast<int>(std::floor((world.y() - frame.y0) / frame.res));
    if (column < 0 || row < 0 || column >= frame.width ||
        row >= frame.height) {
      continue;
    }
    cells.push_back(static_cast<size_t>(row) * frame.width + column);
  }
  return {cells, samples};
}

double NanMean(const std::vector<float>& raster,
               const std::vector<size_t>& cells) {
  double sum = 0.0;
  int count = 0;
  for (size_t cell : cells) {
    if (std::isfinite(raster[cell])) {
      sum += raster[cell];
      ++count;
    }
  }
  return count ? sum / count : kNaN;
}

// Whether nothing stands on a cell of height `height` (NaN: nothing seen).
bool EmptyCell(float height, double occupied_height) {
  return !(std::isfinite(height) ? height >= occupied_height
                                 : 0.0 >= occupied_height);
}

// (visible share, paint ridge) of a hypothetical separator: its share on
// observed floor with nothing standing on it, and the mean intensity on the
// line minus that to both sides (NaN without intensity).
std::pair<double, double> SeparatorEvidence(double across, double along_start,
                                            double along_end, int orientation,
                                            const Eigen::Matrix2d& to_world,
                                            const SlotRasters& rasters,
                                            double occupied_height) {
  const auto [cells, samples] = LineCells(across, along_start, along_end,
                                          orientation, to_world, rasters.frame);
  if (cells.empty()) return {0.0, kNaN};
  int visible = 0;
  for (size_t cell : cells) {
    visible += rasters.floor_count[cell] > 0 &&
               EmptyCell(rasters.height_max[cell], occupied_height);
  }
  const double share = static_cast<double>(visible) / samples;
  if (rasters.intensity.empty()) return {share, kNaN};
  // The line itself, then the lines to either side.
  constexpr int kLines = 3;
  const double offsets[kLines] = {0.0, -kRidgeSideOffset, kRidgeSideOffset};
  double means[kLines];
  for (int line = 0; line < kLines; ++line) {
    const auto [side_cells, unused] =
        LineCells(across + offsets[line], along_start, along_end,
                  orientation, to_world, rasters.frame);
    means[line] =
        side_cells.empty() ? kNaN : NanMean(rasters.intensity, side_cells);
  }
  return {share, means[0] - 0.5 * (means[1] + means[2])};
}

// Whether a hypothetical separator is visibly not there: on observed, empty
// floor and without a paint ridge of `paint_ridge_share` of the row's.
bool Absent(double visible, double ridge, double paint,
            const SlotConfig& config) {
  if (visible < config.visible_separator_share) return false;
  if (!(std::isfinite(ridge) && std::isfinite(paint) && paint > 0.0)) {
    return true;
  }
  return ridge < config.paint_ridge_share * paint;
}

struct Candidate {
  int orientation;
  double across_start;
  double across_end;
  double along_start;
  double along_end;
  bool inferred;
  std::string row_key;
  double row_yaw;
};

double MinDistance(const std::vector<Vec2>& points, const Vec2& query) {
  double best = kInf;
  for (const Vec2& point : points) {
    best = std::min(best, (point - query).norm());
  }
  return best;
}

// Cells of a world polygon; vertices to cells by rounding or flooring.
Mask PolygonCells(const std::vector<Vec2>& polygon, const RasterFrame& frame,
                  bool round_vertices) {
  std::vector<std::pair<int, int>> vertices;
  for (const Vec2& vertex : polygon) {
    const double x = (vertex.x() - frame.x0) / frame.res;
    const double y = (vertex.y() - frame.y0) / frame.res;
    vertices.emplace_back(
        static_cast<int>(round_vertices ? std::nearbyint(x) : std::floor(x)),
        static_cast<int>(round_vertices ? std::nearbyint(y) : std::floor(y)));
  }
  return FillPolygon(vertices, frame.width, frame.height);
}

// The painted (non-virtual) separators of a row.
Row Painted(const Row& row) {
  Row painted;
  for (const Separator& separator : row) {
    if (!separator.is_virtual) painted.push_back(separator);
  }
  return painted;
}

}  // namespace

std::vector<ParkingSlot> AssembleSlots(
    const std::vector<MarkingSegment>& segments, int level, double yaw,
    const std::vector<Vec2>& path_xy, const std::vector<SlotColumn>& columns,
    const SlotRasters& rasters, const SlotConfig& config, const Logger& log) {
  const RasterFrame& frame = rasters.frame;
  const size_t cells = static_cast<size_t>(frame.width) * frame.height;
  GR_CHECK(rasters.height_max.size() == cells &&
           rasters.floor_z.size() == cells);
  GR_CHECK(rasters.floor_count.empty() || rasters.floor_count.size() == cells);
  GR_CHECK(rasters.intensity.empty() || rasters.intensity.size() == cells);
  const bool have_counts = !rasters.floor_count.empty();
  const double width_min = config.width_min;
  const double width_max = config.width_max;
  const double depth_min = config.depth_min;
  const double depth_max = config.depth_max;
  std::vector<std::vector<Vec2>> column_boxes;
  for (const SlotColumn& column : columns) {
    const Vec2 half = column.size / 2.0;
    column_boxes.push_back({column.center + Vec2(-half.x(), -half.y()),
                            column.center + Vec2(half.x(), -half.y()),
                            column.center + Vec2(half.x(), half.y()),
                            column.center + Vec2(-half.x(), half.y())});
  }
  std::vector<int> all(segments.size());
  std::iota(all.begin(), all.end(), 0);
  std::vector<Candidate> candidates;
  for (int orientation : {kSeparatorsAlongY, kSeparatorsAlongX}) {
    std::vector<std::pair<Row, double>> rows;
    const std::vector<Row> groups =
        Rows(Project(segments, all, orientation, yaw, config),
             config.row_overlap);
    // Rows too short or noisy for their own direction take the median of
    // the significant ones.
    std::vector<double> tilts;
    std::vector<double> significant;
    for (const Row& group : groups) {
      tilts.push_back(RowTilt(group));
      if (tilts.back() != 0.0) significant.push_back(tilts.back());
    }
    const double fallback =
        significant.size() >= kMinSignificantTilts ? Median(significant) : 0.0;
    for (size_t index = 0; index < groups.size(); ++index) {
      const double tilt = tilts[index] != 0.0 ? tilts[index] : fallback;
      const double row_yaw =
          yaw + (orientation == kSeparatorsAlongY ? tilt : -tilt);
      Row row;
      if (tilt != 0.0) {
        std::vector<int> indices;
        for (const Separator& separator : groups[index]) {
          indices.push_back(separator.segment);
        }
        row = Project(segments, indices, orientation, row_yaw, config);
      } else {
        row = groups[index];
      }
      if (row.empty()) continue;
      // Back-to-back rows share painted lines: split the whole group at the
      // middle of its extent. A few long fragments do not make a single row
      // back-to-back.
      std::vector<double> starts;
      std::vector<double> ends;
      for (const Separator& separator : row) {
        starts.push_back(separator.along_start);
        ends.push_back(separator.along_end);
      }
      const double start_low = Percentile(starts, kEndPercentileLow);
      const double end_high = Percentile(ends, kEndPercentileHigh);
      if (end_high - start_low <= depth_max + kBackToBackSlack) {
        if (*std::max_element(ends.begin(), ends.end()) -
                *std::min_element(starts.begin(), starts.end()) >
            depth_max + kBackToBackSlack) {
          for (Separator& separator : row) {
            separator.along_start =
                std::max(separator.along_start, start_low - kEndClipMargin);
            separator.along_end =
                std::min(separator.along_end, end_high + kEndClipMargin);
          }
        }
        rows.emplace_back(row, row_yaw);
        continue;
      }
      const RowEnds row_ends = EndsOf(row);
      const double middle = 0.5 * (row_ends.start + row_ends.end);
      const bool measured = row_ends.start_measured && row_ends.end_measured;
      Row low;
      Row high;
      for (const Separator& separator : row) {
        if (separator.along_start < middle - kSplitMargin) {
          Separator half = separator;
          half.along_end = std::min(separator.along_end, middle);
          half.end_measured =
              separator.along_end > middle ? measured : separator.end_measured;
          low.push_back(half);
        }
      }
      for (const Separator& separator : row) {
        if (separator.along_end > middle + kSplitMargin) {
          Separator half = separator;
          half.along_start = std::max(separator.along_start, middle);
          half.start_measured = separator.along_start < middle
                                    ? measured
                                    : separator.start_measured;
          high.push_back(half);
        }
      }
      if (!low.empty()) rows.emplace_back(low, row_yaw);
      if (!high.empty()) rows.emplace_back(high, row_yaw);
    }
    // Columns at the row's corners bound bays as reliably as paint: their
    // faces become virtual separators (they make hidden lines inferable).
    for (auto& [row, row_yaw] : rows) {
      const Eigen::Matrix2d to_row = Rot2(-row_yaw);
      const auto [row_start, row_end] = AlongExtent(row);
      for (const SlotColumn& column : columns) {
        const Vec2 center = to_row * column.center;
        const double half_across = column.size(1 - orientation) / 2.0;
        if (center(orientation) < row_start - kColumnRowMargin ||
            center(orientation) > row_end + kColumnRowMargin) {
          continue;
        }
        for (double face : {center(1 - orientation) - half_across,
                            center(1 - orientation) + half_across}) {
          bool apart = true;
          for (const Separator& separator : row) {
            apart &= std::abs(face - separator.across) > kVirtualSeparatorGap;
          }
          if (apart) {
            Separator face_separator;
            face_separator.across = face;
            face_separator.along_start = row_start;
            face_separator.along_end = row_end;
            face_separator.is_virtual = true;
            row.push_back(face_separator);
          }
        }
      }
    }
    std::vector<double> all_gaps;
    for (const auto& [row, unused] : rows) {
      std::vector<double> acrosses;
      for (const Separator& separator : row) {
        if (!separator.is_virtual) acrosses.push_back(separator.across);
      }
      std::sort(acrosses.begin(), acrosses.end());
      for (size_t k = 1; k < acrosses.size(); ++k) {
        const double gap = acrosses[k] - acrosses[k - 1];
        if (gap >= width_min && gap <= width_max) all_gaps.push_back(gap);
      }
    }
    const double global_width =
        all_gaps.empty() ? kDefaultSlotWidth : Median(all_gaps);
    // Slot depth from the rows whose two ends were both measured.
    std::vector<double> measured_depths;
    for (const auto& [row, unused] : rows) {
      const Row painted = Painted(row);
      if (painted.empty()) continue;
      const RowEnds row_ends = EndsOf(painted);
      const double depth = row_ends.end - row_ends.start;
      if (row_ends.start_measured && row_ends.end_measured &&
          depth >= depth_min && depth <= depth_max) {
        measured_depths.push_back(depth);
      }
    }
    const double typical_depth = measured_depths.empty()
                                     ? config.typical_depth
                                     : Median(measured_depths);
    for (auto& [unsorted_row, row_yaw] : rows) {
      const Eigen::Matrix2d to_row = Rot2(-row_yaw);
      const Eigen::Matrix2d to_world = Rot2(row_yaw);
      std::vector<Vec2> path_row;
      for (const Vec2& sample : path_xy) path_row.push_back(to_row * sample);
      Row row = unsorted_row;
      std::stable_sort(row.begin(), row.end(),
                       [](const Separator& a, const Separator& b) {
                         return a.across < b.across;
                       });
      const Row painted = Painted(row);
      if (row.size() < 2 || painted.empty()) continue;
      const RowEnds row_ends = EndsOf(painted);
      double start = row_ends.start;
      double end = row_ends.end;
      if (end - start > depth_max) continue;
      if (row_ends.start_measured != row_ends.end_measured &&
          std::abs(end - start - typical_depth) < kDepthMatchTolerance) {
        // One end measured, the other hidden, the paint about as long as a
        // slot: the hidden end at the measured slot depth.
        if (row_ends.start_measured) {
          end = start + typical_depth;
        } else {
          start = end - typical_depth;
        }
      } else if (end - start < typical_depth - kShortRowSlack) {
        // One end hidden: keep the measured end and extend the other;
        // without measured ends extend away from the driven aisle.
        bool entrance_at_start;
        if (row_ends.start_measured != row_ends.end_measured) {
          entrance_at_start = row_ends.start_measured;
        } else {
          std::vector<double> acrosses;
          for (const Separator& separator : row) {
            acrosses.push_back(separator.across);
          }
          const double middle_across = Median(acrosses);
          entrance_at_start =
              path_row.empty() ||
              MinDistance(path_row,
                          PointAt(middle_across, start, orientation)) <=
                  MinDistance(path_row,
                              PointAt(middle_across, end, orientation));
        }
        if (entrance_at_start) {
          end = start + typical_depth;
        } else {
          start = end - typical_depth;
        }
      }
      if (end - start < depth_min || end - start > depth_max) continue;
      std::vector<double> widths;
      for (size_t k = 1; k < painted.size(); ++k) {
        const double gap = painted[k].across - painted[k - 1].across;
        if (gap >= width_min && gap <= width_max) widths.push_back(gap);
      }
      const double typical_width =
          widths.empty() ? global_width : Median(widths);
      // Paint contrast of this row's separators, for the hidden ones.
      std::vector<double> ridges;
      if (have_counts && !rasters.intensity.empty()) {
        for (const Separator& separator : painted) {
          const double ridge =
              SeparatorEvidence(separator.across,
                                std::max(separator.along_start, start),
                                std::min(separator.along_end, end),
                                orientation, to_world, rasters,
                                config.occupied_height)
                  .second;
          if (std::isfinite(ridge)) ridges.push_back(ridge);
        }
      }
      const double paint = ridges.empty() ? kNaN : Median(ridges);
      const std::string row_key = StringPrintf(
          "%d:%.1f:%.4f", orientation,
          0.5 * (row[0].along_start + row[0].along_end), row_yaw);
      for (size_t k = 0; k + 1 < row.size(); ++k) {
        const double across = row[k].across;
        const double gap = row[k + 1].across - across;
        if (row[k].is_virtual && row[k + 1].is_virtual) continue;
        std::vector<std::pair<double, double>> pieces;
        bool inferred = false;
        if (gap >= width_min && gap <= width_max) {
          pieces.emplace_back(across, across + gap);
        } else if (gap > width_max) {
          const int slots_in_gap =
              static_cast<int>(std::nearbyint(gap / typical_width));
          if (slots_in_gap < kMinInferredSlots ||
              std::abs(gap - slots_in_gap * typical_width) >
                  config.infer_tolerance * slots_in_gap) {
            continue;
          }
          if (have_counts) {
            bool absent = false;
            for (int line = 1; line < slots_in_gap && !absent; ++line) {
              const auto [visible, ridge] = SeparatorEvidence(
                  across + line * gap / slots_in_gap, start, end, orientation,
                  to_world, rasters, config.occupied_height);
              absent = Absent(visible, ridge, paint, config);
            }
            if (absent) continue;  // The missing line would have been seen.
          }
          for (int slot = 0; slot < slots_in_gap; ++slot) {
            pieces.emplace_back(across + slot * gap / slots_in_gap,
                                across + (slot + 1) * gap / slots_in_gap);
          }
          inferred = true;
        } else {
          continue;
        }
        for (const auto& [piece_start, piece_end] : pieces) {
          candidates.push_back({orientation, piece_start, piece_end, start,
                                end, inferred, row_key, row_yaw});
        }
      }
    }
  }

  Mask seen;
  if (have_counts) {
    seen.assign(cells, 0);
    for (size_t cell = 0; cell < cells; ++cell) {
      seen[cell] = rasters.floor_count[cell] > 0 ||
                   !EmptyCell(rasters.height_max[cell], config.occupied_height);
    }
  }
  std::vector<ParkingSlot> slots;
  for (const Candidate& candidate : candidates) {
    const Eigen::Matrix2d to_row = Rot2(-candidate.row_yaw);
    const Eigen::Matrix2d to_world = Rot2(candidate.row_yaw);
    bool entrance_at_start = true;
    if (!path_xy.empty()) {
      std::vector<Vec2> path_row;
      for (const Vec2& sample : path_xy) path_row.push_back(to_row * sample);
      const double middle =
          0.5 * (candidate.across_start + candidate.across_end);
      entrance_at_start =
          MinDistance(path_row, PointAt(middle, candidate.along_start,
                                        candidate.orientation)) <=
          MinDistance(path_row, PointAt(middle, candidate.along_end,
                                        candidate.orientation));
    }
    const double c0 = candidate.across_start;
    const double c1 = candidate.across_end;
    const double a0 = candidate.along_start;
    const double a1 = candidate.along_end;
    using Corners = std::array<Vec2, kParkingSlotCorners>;
    Corners corners;
    if (candidate.orientation == kSeparatorsAlongY) {
      corners = entrance_at_start ? Corners{Vec2(c0, a0), Vec2(c1, a0),
                                            Vec2(c1, a1), Vec2(c0, a1)}
                                  : Corners{Vec2(c1, a1), Vec2(c0, a1),
                                            Vec2(c0, a0), Vec2(c1, a0)};
    } else {
      corners = entrance_at_start ? Corners{Vec2(a0, c1), Vec2(a0, c0),
                                            Vec2(a1, c0), Vec2(a1, c1)}
                                  : Corners{Vec2(a1, c0), Vec2(a1, c1),
                                            Vec2(a0, c1), Vec2(a0, c0)};
    }
    for (Vec2& corner : corners) corner = to_world * corner;
    const std::vector<Vec2> polygon(corners.begin(), corners.end());
    bool on_column = false;
    for (const std::vector<Vec2>& box : column_boxes) {
      on_column |=
          ConvexIntersectionArea(polygon, box) > config.column_overlap_area;
    }
    if (on_column) continue;
    if (candidate.inferred && !path_xy.empty()) {
      const std::vector<Vec2> core = InsetConvex(polygon, kSlotInset);
      bool driven = false;
      if (!core.empty()) {
        for (const Vec2& sample : path_xy) {
          if (PointInPolygon(core, sample)) {
            driven = true;
            break;
          }
        }
      }
      if (driven) continue;  // The vehicle drove through it: an aisle.
    }
    if (candidate.inferred && have_counts) {
      // Evidence for the slot itself: observed floor or a parked car.
      const Mask inside = PolygonCells(polygon, frame, false);
      int inside_cells = 0;
      int observed = 0;
      for (size_t cell = 0; cell < cells; ++cell) {
        if (!inside[cell]) continue;
        ++inside_cells;
        observed += seen[cell];
      }
      if (inside_cells == 0 || static_cast<double>(observed) / inside_cells <
                                   config.inferred_min_evidence) {
        continue;
      }
    }
    ParkingSlot slot;
    slot.level = level;
    slot.corners = corners;
    slot.width = c1 - c0;
    slot.depth = a1 - a0;
    slot.inferred = candidate.inferred;
    slot.confidence =
        candidate.inferred ? kInferredConfidence : kPaintedConfidence;
    slot.row = candidate.row_key;
    slots.push_back(slot);
  }
  // Duplicates (both orientations, overlapping rows): higher confidence first.
  std::stable_sort(slots.begin(), slots.end(),
                   [](const ParkingSlot& a, const ParkingSlot& b) {
                     return a.confidence > b.confidence;
                   });
  std::vector<ParkingSlot> kept;
  for (const ParkingSlot& slot : slots) {
    const std::vector<Vec2> polygon(slot.corners.begin(), slot.corners.end());
    const double area = std::abs(PolygonArea(polygon));
    const double max_share =
        slot.inferred ? config.inferred_max_overlap : kMaxPaintedOverlap;
    bool apart = true;
    for (const ParkingSlot& other : kept) {
      const std::vector<Vec2> other_polygon(other.corners.begin(),
                                            other.corners.end());
      if (ConvexIntersectionArea(polygon, other_polygon) >= max_share * area) {
        apart = false;
        break;
      }
    }
    if (apart) kept.push_back(slot);
  }
  // Occupancy at capture time, floor height and ids.
  std::vector<std::string> keys;
  for (const ParkingSlot& slot : kept) {
    if (std::find(keys.begin(), keys.end(), slot.row) == keys.end()) {
      keys.push_back(slot.row);
    }
  }
  // Row keys are "orientation:along centre:yaw".
  const auto key_order = [](const std::string& key) {
    const size_t first = key.find(':');
    const size_t second = key.find(':', first + 1);
    return std::make_pair(key.substr(0, first),
                          std::stod(key.substr(first + 1, second - first - 1)));
  };
  std::stable_sort(keys.begin(), keys.end(),
                   [&](const std::string& a, const std::string& b) {
                     return key_order(a) < key_order(b);
                   });
  for (ParkingSlot& slot : kept) {
    const std::vector<Vec2> inset = InsetConvex(
        std::vector<Vec2>(slot.corners.begin(), slot.corners.end()),
        kSlotInset);
    if (inset.empty()) continue;
    const Mask inside = PolygonCells(inset, frame, true);
    int inside_cells = 0;
    int high = 0;
    std::vector<double> floor;
    for (size_t cell = 0; cell < cells; ++cell) {
      if (!inside[cell]) continue;
      ++inside_cells;
      const float height = rasters.height_max[cell];
      high += (std::isfinite(height) ? height : 0.0f) >=
              config.occupied_height;
      if (std::isfinite(rasters.floor_z[cell])) {
        floor.push_back(rasters.floor_z[cell]);
      }
    }
    if (inside_cells == 0) continue;
    slot.occupied = static_cast<double>(high) / inside_cells >=
                    config.occupied_fraction;
    slot.z = floor.empty() ? kNaN : Median(floor);
  }
  const Eigen::Matrix2d to_grid = Rot2(-yaw);
  for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
    const std::string& key = keys[key_index];
    const int orientation = key[0] - '0';
    std::vector<size_t> members;
    for (size_t i = 0; i < kept.size(); ++i) {
      if (kept[i].row == key) members.push_back(i);
    }
    std::vector<double> along;
    for (size_t member : members) {
      Vec2 corner_sum = Vec2::Zero();
      for (const Vec2& corner : kept[member].corners) {
        corner_sum += to_grid * corner;
      }
      along.push_back(orientation == kSeparatorsAlongY
                          ? corner_sum.x() / kParkingSlotCorners
                          : corner_sum.y() / kParkingSlotCorners);
    }
    std::vector<size_t> order(members.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return along[a] < along[b];
    });
    const std::string row = StringPrintf("R%02zu", key_index + 1);
    for (size_t k = 0; k < order.size(); ++k) {
      ParkingSlot& member = kept[members[order[k]]];
      member.row = row;
      member.slot_id = StringPrintf("L%d-%s-%03zu", level, row.c_str(), k + 1);
    }
  }
  int inferred = 0;
  int occupied = 0;
  for (const ParkingSlot& slot : kept) {
    inferred += slot.inferred;
    occupied += slot.occupied == 1;
  }
  LogLine(log, StringPrintf("[slots] level %d: %zu slots (%d inferred, %d "
                            "occupied at capture)",
                            level, kept.size(), inferred, occupied));
  return kept;
}

}  // namespace gr
