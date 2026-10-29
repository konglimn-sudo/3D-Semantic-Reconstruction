#include "gr/core/morphology.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The largest 8-bit intensity: where a grey erosion starts.
constexpr int kMaxIntensity = std::numeric_limits<uint8_t>::max();
// The number of 8-bit intensities: the bins of an 8-bit histogram.
constexpr int kIntensityLevels = kMaxIntensity + 1;

// Cells of a 3 x 3 neighbourhood reach this far from its centre.
constexpr int kNeighbourhoodRadius = 1;
// The 8-neighbours of a cell.
constexpr int kNeighbourCount = 8;
// The distinct 8-neighbour codes (one bit per neighbour).
constexpr int kNeighbourCodes = 1 << kNeighbourCount;

// Offset from a cell to one of its neighbours.
struct Offset {
  int dx = 0;
  int dy = 0;
};

// Index of cell (x, y) in a row-major raster `width` cells wide.
size_t CellIndex(int x, int y, int width) {
  return static_cast<size_t>(y) * static_cast<size_t>(width) +
         static_cast<size_t>(x);
}

// Number of cells of a `width` x `height` raster.
size_t CellCount(int width, int height) {
  return static_cast<size_t>(width) * static_cast<size_t>(height);
}

// Mask of a grey image: 1 where it is nonzero.
Mask ToMask(const std::vector<uint8_t>& image) {
  Mask out(image.size());
  for (size_t i = 0; i < image.size(); ++i) out[i] = image[i] != 0;
  return out;
}

// --- Grey morphology by a kernel ---------------------------------------------

// Grey erosion (min) or dilation (max) by a binary kernel anchored at its
// centre; cells outside the image do not take part.
std::vector<uint8_t> MorphKernel(const std::vector<uint8_t>& image, int width,
                                 int height, const Mask& kernel,
                                 int kernel_size, bool erode) {
  GR_CHECK(image.size() == CellCount(width, height));
  GR_CHECK(kernel.size() == CellCount(kernel_size, kernel_size));
  const int anchor = kernel_size / 2;
  std::vector<Offset> offsets;
  for (int y = 0; y < kernel_size; ++y) {
    for (int x = 0; x < kernel_size; ++x) {
      if (kernel[CellIndex(x, y, kernel_size)]) {
        offsets.push_back({x - anchor, y - anchor});
      }
    }
  }
  std::vector<uint8_t> out(image.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int extremum = erode ? kMaxIntensity : 0;
      for (const Offset& offset : offsets) {
        const int neighbour_x = x + offset.dx;
        const int neighbour_y = y + offset.dy;
        if (neighbour_x < 0 || neighbour_y < 0 || neighbour_x >= width ||
            neighbour_y >= height) {
          continue;
        }
        const int intensity = image[CellIndex(neighbour_x, neighbour_y, width)];
        extremum = erode ? std::min(extremum, intensity)
                         : std::max(extremum, intensity);
      }
      out[CellIndex(x, y, width)] = static_cast<uint8_t>(extremum);
    }
  }
  return out;
}

// --- Lines and polygons (OpenCV drawing.cpp) ---------------------------------

// All bits set when `condition` holds, else none: the branch-free selection
// of cv::LineIterator.
int AllBitsIf(bool condition) { return condition ? ~0 : 0; }

// cv::LineIterator(img, pt1, pt2, 8, leftToRight = true) as cv::line: calls
// `set_cell(index)` for the cells of a line inside the raster.
template <typename SetCell>
void ForEachLine8Cell(int start_x, int start_y, int end_x, int end_y, int width,
                      SetCell set_cell) {
  // Along the major axis the line advances one cell per step, along the
  // minor one only when the error term says so.
  int major_delta = end_x - start_x;
  int minor_delta = end_y - start_y;
  // Leftwards: walk from the end instead.
  int swap = AllBitsIf(major_delta < 0);
  major_delta = (major_delta ^ swap) - swap;
  minor_delta = (minor_delta ^ swap) - swap;
  start_x ^= (start_x ^ end_x) & swap;
  start_y ^= (start_y ^ end_y) & swap;
  int major_step = 1;  // Index steps along x and along y.
  int minor_step = width;
  // Upwards: step the rows backwards.
  swap = AllBitsIf(minor_delta < 0);
  minor_delta = (minor_delta ^ swap) - swap;
  minor_step = (minor_step ^ swap) - swap;
  // Steep: y is the major axis.
  swap = AllBitsIf(minor_delta > major_delta);
  major_delta ^= minor_delta & swap;
  minor_delta ^= major_delta & swap;
  major_delta ^= minor_delta & swap;
  major_step ^= minor_step & swap;
  minor_step ^= major_step & swap;
  major_step ^= minor_step & swap;
  int error = major_delta - (minor_delta + minor_delta);
  const int error_per_step = -(minor_delta + minor_delta);
  const int error_per_minor_step = major_delta + major_delta;
  int64_t index = static_cast<int64_t>(start_y) * width + start_x;
  for (int step = 0; step <= major_delta; ++step) {
    set_cell(static_cast<size_t>(index));
    const int take_minor_step = AllBitsIf(error < 0);
    error += error_per_step + (error_per_minor_step & take_minor_step);
    index += major_step + (minor_step & take_minor_step);
  }
}

// Cohen-Sutherland outcode bits of cv::clipLine: where a point lies relative
// to the raster.
constexpr int kLeftOfRaster = 1;
constexpr int kRightOfRaster = 2;
constexpr int kAboveRaster = 4;
constexpr int kBelowRaster = 8;
constexpr int kAboveOrBelowRaster = kAboveRaster | kBelowRaster;

// How far a line with deltas (`delta`, `other_delta`) moves along the other
// axis while moving `step` along this one, truncated as cv::clipLine.
int64_t ProportionalStep(int64_t step, int64_t other_delta, int64_t delta) {
  return static_cast<int64_t>(static_cast<double>(step) *
                              static_cast<double>(other_delta) /
                              static_cast<double>(delta));
}

// cv::clipLine: the part of the segment inside [0, width) x [0, height);
// false if none.
bool ClipLine(int64_t width, int64_t height, int64_t* start_x, int64_t* start_y,
              int64_t* end_x, int64_t* end_y) {
  if (width <= 0 || height <= 0) return false;
  const int64_t right = width - 1;
  const int64_t bottom = height - 1;
  const auto horizontal_code = [right](int64_t x) {
    return (x < 0 ? kLeftOfRaster : 0) | (x > right ? kRightOfRaster : 0);
  };
  const auto outcode = [&](int64_t x, int64_t y) {
    return horizontal_code(x) | (y < 0 ? kAboveRaster : 0) |
           (y > bottom ? kBelowRaster : 0);
  };
  int start_code = outcode(*start_x, *start_y);
  int end_code = outcode(*end_x, *end_y);
  if ((start_code & end_code) == 0 && (start_code | end_code) != 0) {
    // First onto the top or bottom row ...
    if (start_code & kAboveOrBelowRaster) {
      const int64_t edge_y = (start_code & kBelowRaster) ? bottom : 0;
      *start_x += ProportionalStep(edge_y - *start_y, *end_x - *start_x,
                                   *end_y - *start_y);
      *start_y = edge_y;
      start_code = horizontal_code(*start_x);
    }
    if (end_code & kAboveOrBelowRaster) {
      const int64_t edge_y = (end_code & kBelowRaster) ? bottom : 0;
      *end_x += ProportionalStep(edge_y - *end_y, *end_x - *start_x,
                                 *end_y - *start_y);
      *end_y = edge_y;
      end_code = horizontal_code(*end_x);
    }
    // ... then onto the left or right column.
    if ((start_code & end_code) == 0 && (start_code | end_code) != 0) {
      if (start_code) {
        const int64_t edge_x = (start_code & kLeftOfRaster) ? 0 : right;
        *start_y += ProportionalStep(edge_x - *start_x, *end_y - *start_y,
                                     *end_x - *start_x);
        *start_x = edge_x;
        start_code = 0;
      }
      if (end_code) {
        const int64_t edge_x = (end_code & kLeftOfRaster) ? 0 : right;
        *end_y += ProportionalStep(edge_x - *end_x, *end_y - *start_y,
                                   *end_x - *start_x);
        *end_x = edge_x;
        end_code = 0;
      }
    }
  }
  return (start_code | end_code) == 0;
}

// cv::fillPoly works with x in 16.16 fixed point (XY_SHIFT); input vertices
// may carry at most as many fraction bits.
constexpr int kFixedPointShift = 16;
constexpr int64_t kFixedPointOne = int64_t{1} << kFixedPointShift;
constexpr int64_t kFixedPointHalf = kFixedPointOne >> 1;

// A polygon vertex as cv::CollectPolyEdges keeps it: x in fixed point, y in
// whole rows.
struct EdgePoint {
  int64_t x = 0;
  int64_t y = 0;
};

// A non-horizontal polygon edge of cv::FillEdgeCollection: it covers rows
// start_y <= y < end_y, and in its current row it is at `x` (fixed point).
struct PolyEdge {
  int start_y = 0;
  int end_y = 0;
  int64_t x = 0;
  int64_t x_step = 0;        // Per row.
  PolyEdge* next = nullptr;  // Next active edge.
};

// Whether 0 <= value < size, as OpenCV's unsigned comparison.
bool InRange(int64_t value, int size) {
  return static_cast<uint64_t>(value) < static_cast<uint64_t>(size);
}

// cv::CollectPolyEdges with LINE_8: draws the outline of the polygon into
// `image` and returns its non-horizontal edges.
std::vector<PolyEdge> CollectPolyEdges(
    const std::vector<std::pair<int64_t, int64_t>>& vertices, int fraction_bits,
    int width, int height, int32_t fill_value, std::vector<int32_t>* image) {
  const int64_t half_input_unit = (int64_t{1} << fraction_bits) >> 1;
  const auto to_edge_point = [&](const std::pair<int64_t, int64_t>& vertex) {
    return EdgePoint{
        vertex.first * (int64_t{1} << (kFixedPointShift - fraction_bits)),
        (vertex.second + half_input_unit) >> fraction_bits};
  };
  std::vector<PolyEdge> edges;
  edges.reserve(vertices.size() + 1);  // Room for FillEdgeCollection's end.
  EdgePoint previous = to_edge_point(vertices.back());
  for (const std::pair<int64_t, int64_t>& vertex : vertices) {
    const EdgePoint current = to_edge_point(vertex);
    // The ends in whole cells.
    int64_t start_x = (previous.x + kFixedPointHalf) >> kFixedPointShift;
    int64_t start_y = previous.y;
    int64_t end_x = (current.x + kFixedPointHalf) >> kFixedPointShift;
    int64_t end_y = current.y;
    const bool outside = !InRange(start_x, width) || !InRange(end_x, width) ||
                         !InRange(start_y, height) || !InRange(end_y, height);
    // The outline (cv::line through a clipping LineIterator).
    int64_t line_start_x = start_x;
    int64_t line_start_y = start_y;
    int64_t line_end_x = end_x;
    int64_t line_end_y = end_y;
    if (!outside || ClipLine(width, height, &line_start_x, &line_start_y,
                             &line_end_x, &line_end_y)) {
      ForEachLine8Cell(
          static_cast<int>(line_start_x), static_cast<int>(line_start_y),
          static_cast<int>(line_end_x), static_cast<int>(line_end_y), width,
          [&](size_t index) { (*image)[index] = fill_value; });
    }
    // Clipped end points give a more accurate edge.
    EdgePoint edge_start = previous;
    EdgePoint edge_end = current;
    if (outside) {
      ClipLine(width, height, &start_x, &start_y, &end_x, &end_y);
      if (start_y != end_y) {
        edge_start = {start_x * kFixedPointOne, start_y};
        edge_end = {end_x * kFixedPointOne, end_y};
      }
    } else {
      edge_start.x += kFixedPointHalf;
      edge_end.x += kFixedPointHalf;
    }
    if (previous.y != current.y) {
      PolyEdge edge;
      edge.x_step = (edge_end.x - edge_start.x) / (edge_end.y - edge_start.y);
      if (previous.y < current.y) {
        edge.start_y = static_cast<int>(previous.y);
        edge.end_y = static_cast<int>(current.y);
        edge.x = edge_start.x + (previous.y - edge_start.y) * edge.x_step;
      } else {
        edge.start_y = static_cast<int>(current.y);
        edge.end_y = static_cast<int>(previous.y);
        edge.x = edge_end.x + (current.y - edge_end.y) * edge.x_step;
      }
      edges.push_back(edge);
    }
    previous = current;
  }
  return edges;
}

// Sets row `y` of `image` between two edges at `edge_x` and `other_edge_x`
// (fixed point), clipped to the raster.
void FillRowSpan(int64_t edge_x, int64_t other_edge_x, int y, int width,
                 int32_t fill_value, std::vector<int32_t>* image) {
  // The edges carry the half-cell offset: both ends round to nearest
  // (OpenCV 4.x).
  const int first_x =
      static_cast<int>(std::min(edge_x, other_edge_x) >> kFixedPointShift);
  const int last_x =
      static_cast<int>(std::max(edge_x, other_edge_x) >> kFixedPointShift);
  if (first_x >= width || last_x < 0) return;
  int32_t* row = image->data() + CellIndex(0, y, width);
  for (int x = std::max(first_x, 0); x <= std::min(last_x, width - 1); ++x) {
    row[x] = fill_value;
  }
}

// Bubble sort by x of the active edge list that follows `head`.
void SortActiveEdgesByX(PolyEdge* head) {
  PolyEdge* settled = nullptr;  // A pass stops there: the rest is in order.
  bool sorted = false;
  do {
    sorted = true;
    PolyEdge* previous = head;
    PolyEdge* current = head->next;
    while (current != settled && current->next != nullptr) {
      PolyEdge* following = current->next;
      if (current->x > following->x) {  // Swap.
        previous->next = following;
        current->next = following->next;
        following->next = current;
        previous = following;
        sorted = false;
      } else {
        previous = current;
        current = following;
      }
    }
    settled = previous;
  } while (!sorted && settled != head->next && settled != head);
}

// cv::FillEdgeCollection: row by row, keeps the edges that cross the row in
// a list sorted by x and fills between consecutive pairs.
void FillEdgeCollection(int width, int height, int32_t fill_value,
                        std::vector<PolyEdge>* edges,
                        std::vector<int32_t>* image) {
  constexpr int kEdgesPerSpan = 2;  // A span runs from one edge to the next.
  const int edge_count = static_cast<int>(edges->size());
  if (edge_count < kEdgesPerSpan) return;
  int max_y = std::numeric_limits<int>::min();
  int min_y = std::numeric_limits<int>::max();
  int64_t max_x = std::numeric_limits<int64_t>::min();
  int64_t min_x = std::numeric_limits<int64_t>::max();
  for (const PolyEdge& edge : *edges) {
    const int64_t end_x = edge.x + (edge.end_y - edge.start_y) * edge.x_step;
    min_y = std::min(min_y, edge.start_y);
    max_y = std::max(max_y, edge.end_y);
    min_x = std::min({min_x, edge.x, end_x});
    max_x = std::max({max_x, edge.x, end_x});
  }
  if (max_y < 0 || min_y >= height || max_x < 0 ||
      min_x >= (static_cast<int64_t>(width) << kFixedPointShift)) {
    return;
  }
  std::sort(edges->begin(), edges->end(),
            [](const PolyEdge& a, const PolyEdge& b) {
              if (a.start_y != b.start_y) return a.start_y < b.start_y;
              if (a.x != b.x) return a.x < b.x;
              return a.x_step < b.x_step;
            });
  PolyEdge never_starts;
  never_starts.start_y = std::numeric_limits<int>::max();
  edges->push_back(never_starts);  // Ends the edges still to start.
  PolyEdge active;  // active.next: the edges crossing the row, by x.
  int next_index = 0;
  PolyEdge* next_edge = &(*edges)[0];  // The next edge to start.
  max_y = std::min(max_y, height);
  for (int y = next_edge->start_y; y < max_y; ++y) {
    PolyEdge* previous = &active;
    PolyEdge* current = active.next;
    bool in_span = false;
    const bool above_raster = y < 0;
    while (current != nullptr || next_edge->start_y == y) {
      if (current != nullptr && current->end_y == y) {
        // The edge ends: out of the active list.
        previous->next = current->next;
        current = current->next;
        continue;
      }
      PolyEdge* span_start = previous;
      if (current != nullptr &&
          (next_edge->start_y > y || current->x < next_edge->x)) {
        previous = current;  // Next edge of the active list.
        current = current->next;
      } else if (next_index < edge_count) {
        previous->next = next_edge;  // An edge starts.
        next_edge->next = current;
        previous = next_edge;
        next_edge = &(*edges)[static_cast<size_t>(++next_index)];
      } else {
        break;
      }
      if (in_span) {
        if (!above_raster) {
          FillRowSpan(span_start->x, previous->x, y, width, fill_value, image);
        }
        span_start->x += span_start->x_step;
        previous->x += previous->x_step;
      }
      in_span = !in_span;
    }
    SortActiveEdgesByX(&active);
  }
}

// --- Filters -----------------------------------------------------------------

// scipy gaussian_filter's default `truncate`: the kernel reaches this many
// standard deviations.
constexpr double kGaussianTruncate = 4.0;
// scipy rounds the kernel radius as int(truncate * sigma + 0.5).
constexpr double kRoundHalfUp = 0.5;

// --- Distance transform ------------------------------------------------------

// Squared distance of the cells without a source before the transform: far
// above any squared distance within a raster, yet finite so that the
// lower-envelope arithmetic stays exact.
constexpr double kNoSourceSquaredDistance = 1e20;
// Transformed squared distances from here on come from cells without a
// source: no source reached them.
constexpr double kUnreachedSquaredDistance = 0.5 * kNoSourceSquaredDistance;

// Lower envelope of the parabolas of one line, reused across lines.
struct LowerEnvelope {
  explicit LowerEnvelope(int max_length)
      : parabola_vertex(static_cast<size_t>(max_length)),
        boundary(static_cast<size_t>(max_length) + 1) {}

  // Where the parabolas of the envelope have their vertex, left to right.
  std::vector<int> parabola_vertex;
  // Parabola k is the lowest between boundary[k] and boundary[k + 1].
  std::vector<double> boundary;
};

// Squared distance transform of one line of `length` cells (Felzenszwalb &
// Huttenlocher): output[q] = min over p of (q - p)^2 + input[p].
void SquaredDistance1d(const std::vector<double>& input, int length,
                       LowerEnvelope* envelope, std::vector<double>* output) {
  std::vector<int>& vertex = envelope->parabola_vertex;
  std::vector<double>& boundary = envelope->boundary;
  int last = 0;  // The last parabola of the envelope so far.
  vertex[0] = 0;
  boundary[0] = -kInf;
  boundary[1] = kInf;
  for (int position = 1; position < length; ++position) {
    // Where the parabola rooted here meets the one rooted at `other`.
    const auto intersection = [&](int other) {
      return ((input[position] + static_cast<double>(position) * position) -
              (input[other] + static_cast<double>(other) * other)) /
             (2.0 * (position - other));
    };
    double crossing = intersection(vertex[last]);
    while (crossing <= boundary[last]) {
      --last;
      crossing = intersection(vertex[last]);
    }
    ++last;
    vertex[last] = position;
    boundary[last] = crossing;
    boundary[last + 1] = kInf;
  }
  int lowest = 0;  // The parabola of the envelope at `position`.
  for (int position = 0; position < length; ++position) {
    while (boundary[lowest + 1] < position) ++lowest;
    const int offset = position - vertex[lowest];
    (*output)[position] =
        static_cast<double>(offset) * offset + input[vertex[lowest]];
  }
}

// --- Thinning and skeletons --------------------------------------------------

// Zhang-Suen neighbours P2..P9 of a cell: clockwise from north.
constexpr std::array<Offset, kNeighbourCount> kClockwiseFromNorth = {
    {{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}}};
// Their positions in kClockwiseFromNorth.
constexpr int kNorth = 0;
constexpr int kEast = 2;
constexpr int kSouth = 4;
constexpr int kWest = 6;
// Zhang-Suen removes a cell with 2 <= B(P1) <= 6 set neighbours and
// A(P1) = 1 unset-to-set transition around it, in two sub-iterations.
constexpr int kMinSetNeighbours = 2;
constexpr int kMaxSetNeighbours = 6;
constexpr int kRequiredTransitions = 1;
constexpr int kSubIterations = 2;

// scikit-image's skeletonize lookup table (skimage/morphology/
// _skeletonize_cy.pyx, BSD-3): per 8-neighbour code, the passes that remove
// the cell (bit p for pass p: 1 the first, 2 the second, 3 both).
constexpr std::array<uint8_t, kNeighbourCodes> kSkeletonizeTable = {
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3,  //
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,  //
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,  //
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,  //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,  //
    3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,  //
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,  //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  //
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,  //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  //
    3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,  //
};
// The neighbour of code bit k: NW, N, NE, E, SE, S, SW, W.
constexpr std::array<Offset, kNeighbourCount> kSkeletonizeNeighbours = {
    {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}}};
constexpr int kSkeletonizePasses = 2;
// scikit-image pads the image with a border of unset cells this wide.
constexpr int kSkeletonizePadding = 1;

// --- Polygons in cell coordinates --------------------------------------------

// Edge crossings this close to a cell centre count as on it.
constexpr double kCrossingTolerance = 1e-9;
// Sorted crossings of a row pair up: each pair bounds an inside span.
constexpr size_t kCrossingsPerSpan = 2;

}  // namespace

Mask DilateCross(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == CellCount(width, height));
  Mask out(mask.size(), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (!mask[CellIndex(x, y, width)]) continue;
      out[CellIndex(x, y, width)] = 1;
      if (x > 0) out[CellIndex(x - 1, y, width)] = 1;
      if (x + 1 < width) out[CellIndex(x + 1, y, width)] = 1;
      if (y > 0) out[CellIndex(x, y - 1, width)] = 1;
      if (y + 1 < height) out[CellIndex(x, y + 1, width)] = 1;
    }
  }
  return out;
}

Mask ErodeCross(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == CellCount(width, height));
  Mask out(mask.size(), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[CellIndex(x, y, width)] =
          mask[CellIndex(x, y, width)] && x > 0 &&
          mask[CellIndex(x - 1, y, width)] && x + 1 < width &&
          mask[CellIndex(x + 1, y, width)] && y > 0 &&
          mask[CellIndex(x, y - 1, width)] && y + 1 < height &&
          mask[CellIndex(x, y + 1, width)];
    }
  }
  return out;
}

Mask CloseCross(const Mask& mask, int width, int height, int iterations) {
  Mask out = mask;
  for (int i = 0; i < iterations; ++i) out = DilateCross(out, width, height);
  for (int i = 0; i < iterations; ++i) out = ErodeCross(out, width, height);
  return out;
}

Mask FillHolesCross(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == CellCount(width, height));
  // Background reached from the border through background cells.
  std::vector<uint8_t> outside(mask.size(), 0);
  std::vector<int> frontier;
  const auto reach = [&](int x, int y) {
    const size_t cell = CellIndex(x, y, width);
    if (mask[cell] || outside[cell]) return;
    outside[cell] = 1;
    frontier.push_back(static_cast<int>(cell));
  };
  for (int x = 0; x < width; ++x) {
    reach(x, 0);
    reach(x, height - 1);
  }
  for (int y = 0; y < height; ++y) {
    reach(0, y);
    reach(width - 1, y);
  }
  while (!frontier.empty()) {
    const int cell = frontier.back();
    frontier.pop_back();
    const int x = cell % width;
    const int y = cell / width;
    if (x > 0) reach(x - 1, y);
    if (x + 1 < width) reach(x + 1, y);
    if (y > 0) reach(x, y - 1);
    if (y + 1 < height) reach(x, y + 1);
  }
  Mask out(mask.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = outside[i] ? 0 : 1;
  return out;
}

std::vector<uint8_t> ErodeKernel(const std::vector<uint8_t>& image, int width,
                                 int height, const Mask& kernel,
                                 int kernel_size) {
  return MorphKernel(image, width, height, kernel, kernel_size,
                     /*erode=*/true);
}

std::vector<uint8_t> DilateKernel(const std::vector<uint8_t>& image, int width,
                                  int height, const Mask& kernel,
                                  int kernel_size) {
  return MorphKernel(image, width, height, kernel, kernel_size,
                     /*erode=*/false);
}

Mask CloseKernel(const Mask& mask, int width, int height, const Mask& kernel,
                 int kernel_size) {
  return ToMask(
      ErodeKernel(DilateKernel(mask, width, height, kernel, kernel_size), width,
                  height, kernel, kernel_size));
}

Mask OpenKernel(const Mask& mask, int width, int height, const Mask& kernel,
                int kernel_size) {
  return ToMask(
      DilateKernel(ErodeKernel(mask, width, height, kernel, kernel_size), width,
                   height, kernel, kernel_size));
}

Mask EllipseKernel(int kernel_size) {
  // cv::getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size)).
  Mask out(CellCount(kernel_size, kernel_size), 0);
  const int radius = kernel_size / 2;  // Also the centre.
  const double inverse_squared_radius =
      radius ? 1.0 / (static_cast<double>(radius) * radius) : 0.0;
  for (int y = 0; y < kernel_size; ++y) {
    const int row_offset = y - radius;
    if (std::abs(row_offset) > radius) continue;
    const int half_width = static_cast<int>(std::nearbyint(
        radius * std::sqrt((radius * radius - row_offset * row_offset) *
                           inverse_squared_radius)));
    const int first_x = std::max(radius - half_width, 0);
    const int end_x = std::min(radius + half_width + 1, kernel_size);
    for (int x = first_x; x < end_x; ++x) {
      out[CellIndex(x, y, kernel_size)] = 1;
    }
  }
  return out;
}

void DrawLine8(int start_x, int start_y, int end_x, int end_y, int width,
               Mask* mask) {
  ForEachLine8Cell(start_x, start_y, end_x, end_y, width,
                   [mask](size_t index) { (*mask)[index] = 1; });
}

void FillPoly8(const std::vector<std::pair<int64_t, int64_t>>& vertices,
               int fraction_bits, int width, int height, int32_t fill_value,
               std::vector<int32_t>* image) {
  // cv::fillPoly(img, pts, color, LINE_8, shift = fraction_bits).
  GR_CHECK(fraction_bits >= 0 && fraction_bits <= kFixedPointShift);
  GR_CHECK(image->size() == CellCount(width, height));
  if (vertices.empty()) return;
  std::vector<PolyEdge> edges = CollectPolyEdges(vertices, fraction_bits, width,
                                                 height, fill_value, image);
  FillEdgeCollection(width, height, fill_value, &edges, image);
}

std::vector<double> GaussianFilter(const std::vector<double>& image, int width,
                                   int height, double sigma) {
  GR_CHECK(image.size() == CellCount(width, height));
  const int radius = static_cast<int>(kGaussianTruncate * sigma + kRoundHalfUp);
  std::vector<double> weights(static_cast<size_t>(2 * radius + 1));
  const double variance = sigma * sigma;
  double total = 0.0;
  for (int offset = -radius; offset <= radius; ++offset) {
    weights[static_cast<size_t>(offset + radius)] =
        std::exp(-0.5 * offset * offset / variance);
    total += weights[static_cast<size_t>(offset + radius)];
  }
  for (double& weight : weights) weight /= total;
  // 'reflect': d c b a | a b c d | d c b a.
  const auto reflect = [](int position, int length) {
    if (length == 1) return 0;
    const int period = 2 * length;
    position %= period;
    if (position < 0) position += period;
    return position < length ? position : period - 1 - position;
  };
  // Axis 0 (along y) first, then axis 1, as scipy.
  std::vector<double> along_y(image.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double sum = 0.0;
      for (int offset = -radius; offset <= radius; ++offset) {
        sum += weights[static_cast<size_t>(offset + radius)] *
               image[CellIndex(x, reflect(y + offset, height), width)];
      }
      along_y[CellIndex(x, y, width)] = sum;
    }
  }
  std::vector<double> out(image.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double sum = 0.0;
      for (int offset = -radius; offset <= radius; ++offset) {
        sum += weights[static_cast<size_t>(offset + radius)] *
               along_y[CellIndex(reflect(x + offset, width), y, width)];
      }
      out[CellIndex(x, y, width)] = sum;
    }
  }
  return out;
}

int OtsuThreshold(const std::vector<uint8_t>& values) {
  // cv::threshold's getThreshVal_Otsu_8u: the threshold that maximizes the
  // between-class variance.
  if (values.empty()) return 0;
  std::array<double, kIntensityLevels> histogram{};
  for (uint8_t value : values) histogram[value] += 1.0;
  const double inverse_count = 1.0 / values.size();
  double mean = 0.0;
  for (int level = 0; level < kIntensityLevels; ++level) {
    mean += level * histogram[level];
  }
  mean *= inverse_count;
  // Classes with less weight than FLT_EPSILON are taken as empty.
  constexpr double kMinClassWeight = std::numeric_limits<float>::epsilon();
  double mean_below = 0.0;  // Of the levels up to `level`.
  double weight_below = 0.0;
  double best_variance = 0.0;
  int best_threshold = 0;
  for (int level = 0; level < kIntensityLevels; ++level) {
    const double probability = histogram[level] * inverse_count;
    mean_below *= weight_below;
    weight_below += probability;
    const double weight_above = 1.0 - weight_below;
    if (std::min(weight_below, weight_above) < kMinClassWeight ||
        std::max(weight_below, weight_above) > 1.0 - kMinClassWeight) {
      continue;
    }
    mean_below = (mean_below + level * probability) / weight_below;
    const double mean_above = (mean - weight_below * mean_below) / weight_above;
    const double between_class_variance = weight_below * weight_above *
                                          (mean_below - mean_above) *
                                          (mean_below - mean_above);
    if (between_class_variance > best_variance) {
      best_variance = between_class_variance;
      best_threshold = level;
    }
  }
  return best_threshold;
}

std::vector<double> MaxFilter3x3(const std::vector<double>& grid, int width,
                                 int height) {
  GR_CHECK(grid.size() == CellCount(width, height));
  std::vector<double> out(grid.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double maximum = -kInf;
      for (int dy = -kNeighbourhoodRadius; dy <= kNeighbourhoodRadius; ++dy) {
        for (int dx = -kNeighbourhoodRadius; dx <= kNeighbourhoodRadius; ++dx) {
          maximum = std::max(
              maximum,
              grid[CellIndex(std::clamp(x + dx, 0, width - 1),
                             std::clamp(y + dy, 0, height - 1), width)]);
        }
      }
      out[CellIndex(x, y, width)] = maximum;
    }
  }
  return out;
}

Mask OpenLine(const Mask& mask, int width, int height, int line_length,
              bool along_x) {
  GR_CHECK(mask.size() == CellCount(width, height));
  GR_CHECK(line_length >= 1);
  const int anchor = line_length / 2;
  const int line_count = along_x ? height : width;
  const int line_cells = along_x ? width : height;
  const auto cell = [&](int line, int position) {
    return along_x ? CellIndex(position, line, width)
                   : CellIndex(line, position, width);
  };
  // Both passes take the window [position - anchor,
  // position - anchor + line_length - 1].
  const auto pass = [&](const Mask& input, bool erode) {
    Mask out(input.size(), 0);
    for (int line = 0; line < line_count; ++line) {
      for (int position = 0; position < line_cells; ++position) {
        bool result = erode;
        for (int k = 0; k < line_length; ++k) {
          const int window_position = position - anchor + k;
          // Outside: set for the erosion, unset for the dilation.
          const bool set = window_position < 0 || window_position >= line_cells
                               ? erode
                               : input[cell(line, window_position)] != 0;
          if (set != erode) {
            result = set;
            break;
          }
        }
        out[cell(line, position)] = result;
      }
    }
    return out;
  };
  return pass(pass(mask, /*erode=*/true), /*erode=*/false);
}

std::vector<float> DistanceTransform(const std::vector<uint8_t>& sources,
                                     int width, int height) {
  GR_CHECK(width >= 0 && height >= 0);
  GR_CHECK(sources.size() == CellCount(width, height));
  std::vector<double> squared_distance(sources.size());
  for (size_t i = 0; i < sources.size(); ++i) {
    squared_distance[i] = sources[i] != 0 ? 0.0 : kNoSourceSquaredDistance;
  }
  const int max_length = std::max(width, height);
  std::vector<double> line(static_cast<size_t>(max_length));
  std::vector<double> line_result(static_cast<size_t>(max_length));
  LowerEnvelope envelope(max_length);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      line[y] = squared_distance[CellIndex(x, y, width)];
    }
    SquaredDistance1d(line, height, &envelope, &line_result);
    for (int y = 0; y < height; ++y) {
      squared_distance[CellIndex(x, y, width)] = line_result[y];
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      line[x] = squared_distance[CellIndex(x, y, width)];
    }
    SquaredDistance1d(line, width, &envelope, &line_result);
    for (int x = 0; x < width; ++x) {
      squared_distance[CellIndex(x, y, width)] = line_result[x];
    }
  }
  std::vector<float> out(squared_distance.size());
  for (size_t i = 0; i < squared_distance.size(); ++i) {
    out[i] = squared_distance[i] >= kUnreachedSquaredDistance
                 ? std::numeric_limits<float>::infinity()
                 : static_cast<float>(std::sqrt(squared_distance[i]));
  }
  return out;
}

std::vector<uint8_t> ThinMask(const std::vector<uint8_t>& mask, int width,
                              int height) {
  GR_CHECK(mask.size() == CellCount(width, height));
  std::vector<uint8_t> image(mask.size());
  for (size_t i = 0; i < mask.size(); ++i) image[i] = mask[i] != 0 ? 1 : 0;
  const auto value_at = [&](int x, int y) -> int {
    if (x < 0 || y < 0 || x >= width || y >= height) return 0;
    return image[CellIndex(x, y, width)];
  };
  std::vector<size_t> removed;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int sub_iteration = 0; sub_iteration < kSubIterations;
         ++sub_iteration) {
      removed.clear();
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          if (image[CellIndex(x, y, width)] == 0) continue;
          std::array<int, kNeighbourCount> neighbours{};  // P2..P9.
          for (int k = 0; k < kNeighbourCount; ++k) {
            neighbours[k] = value_at(x + kClockwiseFromNorth[k].dx,
                                     y + kClockwiseFromNorth[k].dy);
          }
          int set_neighbours = 0;
          int transitions = 0;
          for (int k = 0; k < kNeighbourCount; ++k) {
            set_neighbours += neighbours[k];
            if (neighbours[k] == 0 &&
                neighbours[(k + 1) % kNeighbourCount] == 1) {
              ++transitions;
            }
          }
          if (set_neighbours < kMinSetNeighbours ||
              set_neighbours > kMaxSetNeighbours ||
              transitions != kRequiredTransitions) {
            continue;
          }
          const auto all_set = [&](int first, int second, int third) {
            return neighbours[first] && neighbours[second] && neighbours[third];
          };
          const bool keep = sub_iteration == 0
                                ? all_set(kNorth, kEast, kSouth) ||
                                      all_set(kEast, kSouth, kWest)
                                : all_set(kNorth, kEast, kWest) ||
                                      all_set(kNorth, kSouth, kWest);
          if (!keep) removed.push_back(CellIndex(x, y, width));
        }
      }
      for (size_t cell : removed) image[cell] = 0;
      changed |= !removed.empty();
    }
  }
  return image;
}

Mask Skeletonize(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == CellCount(width, height));
  const int padded_width = width + 2 * kSkeletonizePadding;
  const int padded_height = height + 2 * kSkeletonizePadding;
  const auto padded_cell = [padded_width](int x, int y) {
    return CellIndex(x + kSkeletonizePadding, y + kSkeletonizePadding,
                     padded_width);
  };
  Mask skeleton(CellCount(padded_width, padded_height), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      skeleton[padded_cell(x, y)] = mask[CellIndex(x, y, width)] != 0;
    }
  }
  Mask cleaned = skeleton;
  bool removed = true;
  while (removed) {
    removed = false;
    for (int pass = 0; pass < kSkeletonizePasses; ++pass) {
      const uint8_t pass_bit = 1 << pass;
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          if (!skeleton[padded_cell(x, y)]) continue;
          int code = 0;
          for (int k = 0; k < kNeighbourCount; ++k) {
            code |= skeleton[padded_cell(x + kSkeletonizeNeighbours[k].dx,
                                         y + kSkeletonizeNeighbours[k].dy)]
                    << k;
          }
          if (kSkeletonizeTable[code] & pass_bit) {
            cleaned[padded_cell(x, y)] = 0;
            removed = true;
          }
        }
      }
      skeleton = cleaned;
    }
  }
  Mask out(CellCount(width, height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[CellIndex(x, y, width)] = skeleton[padded_cell(x, y)];
    }
  }
  return out;
}

Mask FillPolygon(const std::vector<std::pair<int, int>>& vertices, int width,
                 int height) {
  Mask out(CellCount(width, height), 0);
  const size_t vertex_count = vertices.size();
  if (vertex_count == 0) return out;
  int min_y = vertices[0].second;
  int max_y = min_y;
  for (const auto& [vertex_x, vertex_y] : vertices) {
    min_y = std::min(min_y, vertex_y);
    max_y = std::max(max_y, vertex_y);
  }
  const auto set_cell = [&](int x, int y) {
    if (x >= 0 && y >= 0 && x < width && y < height) {
      out[CellIndex(x, y, width)] = 1;
    }
  };
  for (int y = std::max(min_y, 0); y <= std::min(max_y, height - 1); ++y) {
    std::vector<double> crossings;  // x of the edges crossing the row.
    for (size_t i = 0; i < vertex_count; ++i) {
      const auto [start_x, start_y] = vertices[i];
      const auto [end_x, end_y] = vertices[(i + 1) % vertex_count];
      // A horizontal edge: its cells are on the boundary.
      if (start_y == end_y) {
        if (y == start_y) {
          for (int x = std::min(start_x, end_x); x <= std::max(start_x, end_x);
               ++x) {
            set_cell(x, y);
          }
        }
        continue;
      }
      // Half-open in y so that shared vertices count once.
      if (y >= std::min(start_y, end_y) && y < std::max(start_y, end_y)) {
        crossings.push_back(start_x + static_cast<double>(y - start_y) *
                                          (end_x - start_x) /
                                          (end_y - start_y));
      }
      // The top vertex of the edge is on the boundary too.
      if (y == std::max(start_y, end_y)) {
        set_cell(y == start_y ? start_x : end_x, y);
      }
    }
    std::sort(crossings.begin(), crossings.end());
    for (size_t enter = 0; enter + 1 < crossings.size();
         enter += kCrossingsPerSpan) {
      const int first_x =
          static_cast<int>(std::ceil(crossings[enter] - kCrossingTolerance));
      const int last_x = static_cast<int>(
          std::floor(crossings[enter + 1] + kCrossingTolerance));
      for (int x = first_x; x <= last_x; ++x) set_cell(x, y);
    }
    // Edge crossings are boundary cells.
    for (double x : crossings) set_cell(static_cast<int>(std::lround(x)), y);
  }
  return out;
}

Components Label8(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == CellCount(width, height));
  Components out;
  out.label.assign(mask.size(), 0);
  std::vector<int> frontier;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t start = CellIndex(x, y, width);
      if (!mask[start] || out.label[start]) continue;
      const int label = static_cast<int>(out.cells.size()) + 1;
      std::vector<int> cells;
      out.label[start] = label;
      frontier.assign(1, static_cast<int>(start));
      while (!frontier.empty()) {
        const int cell = frontier.back();
        frontier.pop_back();
        cells.push_back(cell);
        const int cell_x = cell % width;
        const int cell_y = cell / width;
        for (int dy = -kNeighbourhoodRadius; dy <= kNeighbourhoodRadius; ++dy) {
          for (int dx = -kNeighbourhoodRadius; dx <= kNeighbourhoodRadius;
               ++dx) {
            const int neighbour_x = cell_x + dx;
            const int neighbour_y = cell_y + dy;
            if (neighbour_x < 0 || neighbour_y < 0 || neighbour_x >= width ||
                neighbour_y >= height) {
              continue;
            }
            const size_t neighbour = CellIndex(neighbour_x, neighbour_y, width);
            if (!mask[neighbour] || out.label[neighbour]) continue;
            out.label[neighbour] = label;
            frontier.push_back(static_cast<int>(neighbour));
          }
        }
      }
      std::sort(cells.begin(), cells.end());  // Raster order.
      out.cells.push_back(std::move(cells));
    }
  }
  return out;
}

}  // namespace gr
