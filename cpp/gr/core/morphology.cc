#include "gr/core/morphology.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr double kBig = 1e20;
constexpr double kInf = std::numeric_limits<double>::infinity();

// Squared distance transform of one line (Felzenszwalb & Huttenlocher).
void SquaredDistance1d(const std::vector<double>& f, int n,
                       std::vector<double>& d, std::vector<int>& v,
                       std::vector<double>& z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kInf;
  z[1] = kInf;
  for (int q = 1; q < n; ++q) {
    auto intersection = [&](int p) {
      return ((f[q] + static_cast<double>(q) * q) -
              (f[p] + static_cast<double>(p) * p)) /
             (2.0 * q - 2.0 * p);
    };
    double s = intersection(v[k]);
    while (s <= z[k]) {
      --k;
      s = intersection(v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kInf;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    d[q] = static_cast<double>(q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

size_t Index(int u, int v, int width) {
  return static_cast<size_t>(v) * static_cast<size_t>(width) +
         static_cast<size_t>(u);
}

size_t Cell(int i, int j, int width) {
  return static_cast<size_t>(j) * static_cast<size_t>(width) +
         static_cast<size_t>(i);
}

}  // namespace

Mask DilateCross(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  Mask out(mask.size(), 0);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      if (!mask[Cell(i, j, width)]) continue;
      out[Cell(i, j, width)] = 1;
      if (i > 0) out[Cell(i - 1, j, width)] = 1;
      if (i + 1 < width) out[Cell(i + 1, j, width)] = 1;
      if (j > 0) out[Cell(i, j - 1, width)] = 1;
      if (j + 1 < height) out[Cell(i, j + 1, width)] = 1;
    }
  }
  return out;
}

Mask ErodeCross(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  Mask out(mask.size(), 0);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      out[Cell(i, j, width)] =
          mask[Cell(i, j, width)] && i > 0 && mask[Cell(i - 1, j, width)] &&
          i + 1 < width && mask[Cell(i + 1, j, width)] && j > 0 &&
          mask[Cell(i, j - 1, width)] && j + 1 < height &&
          mask[Cell(i, j + 1, width)];
    }
  }
  return out;
}

Mask CloseCross(const Mask& mask, int width, int height, int iterations) {
  Mask out = mask;
  for (int k = 0; k < iterations; ++k) out = DilateCross(out, width, height);
  for (int k = 0; k < iterations; ++k) out = ErodeCross(out, width, height);
  return out;
}

namespace {

// Grey erosion (min) or dilation (max) by a binary kernel anchored at its
// centre; cells outside the image do not take part.
std::vector<uint8_t> MorphKernel(const std::vector<uint8_t>& image, int width,
                                 int height, const Mask& kernel, int size,
                                 bool erode) {
  GR_CHECK(image.size() == static_cast<size_t>(width) * height);
  GR_CHECK(kernel.size() == static_cast<size_t>(size) * size);
  const int anchor = size / 2;
  std::vector<std::pair<int, int>> offsets;
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      if (kernel[Cell(x, y, size)]) {
        offsets.emplace_back(x - anchor, y - anchor);
      }
    }
  }
  std::vector<uint8_t> out(image.size());
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      int value = erode ? 255 : 0;
      for (const auto& [dx, dy] : offsets) {
        const int x = i + dx;
        const int y = j + dy;
        if (x < 0 || y < 0 || x >= width || y >= height) continue;
        const int v = image[Cell(x, y, width)];
        value = erode ? std::min(value, v) : std::max(value, v);
      }
      out[Cell(i, j, width)] = static_cast<uint8_t>(value);
    }
  }
  return out;
}

}  // namespace

std::vector<uint8_t> ErodeKernel(const std::vector<uint8_t>& image, int width,
                                 int height, const Mask& kernel,
                                 int kernel_size) {
  return MorphKernel(image, width, height, kernel, kernel_size, true);
}

std::vector<uint8_t> DilateKernel(const std::vector<uint8_t>& image,
                                  int width, int height, const Mask& kernel,
                                  int kernel_size) {
  return MorphKernel(image, width, height, kernel, kernel_size, false);
}

Mask EllipseKernel(int size) {
  // cv::getStructuringElement(MORPH_ELLIPSE, Size(size, size)).
  Mask out(static_cast<size_t>(size) * size, 0);
  const int r = size / 2;
  const int c = size / 2;
  const double inv_r2 = r ? 1.0 / (static_cast<double>(r) * r) : 0.0;
  for (int i = 0; i < size; ++i) {
    const int dy = i - r;
    if (std::abs(dy) > r) continue;
    const int dx = static_cast<int>(
        std::nearbyint(c * std::sqrt((r * r - dy * dy) * inv_r2)));
    const int j1 = std::max(c - dx, 0);
    const int j2 = std::min(c + dx + 1, size);
    for (int j = j1; j < j2; ++j) out[Cell(j, i, size)] = 1;
  }
  return out;
}

namespace {

Mask ToMask(const std::vector<uint8_t>& image) {
  Mask out(image.size());
  for (size_t k = 0; k < image.size(); ++k) out[k] = image[k] != 0;
  return out;
}

}  // namespace

Mask CloseKernel(const Mask& mask, int width, int height, const Mask& kernel,
                 int size) {
  return ToMask(ErodeKernel(DilateKernel(mask, width, height, kernel, size),
                            width, height, kernel, size));
}

Mask OpenKernel(const Mask& mask, int width, int height, const Mask& kernel,
                int size) {
  return ToMask(DilateKernel(ErodeKernel(mask, width, height, kernel, size),
                             width, height, kernel, size));
}

namespace {

// cv::LineIterator(img, pt1, pt2, 8, leftToRight = true) as cv::line: calls
// `set(index)` for the cells of a line inside the raster.
template <typename Set>
void ForEachLine8Cell(int x0, int y0, int x1, int y1, int width, Set set) {
  int dx = x1 - x0;
  int dy = y1 - y0;
  int s = dx < 0 ? -1 : 0;
  dx = (dx ^ s) - s;
  dy = (dy ^ s) - s;
  x0 ^= (x0 ^ x1) & s;
  y0 ^= (y0 ^ y1) & s;
  int bt_pix = 1;
  int istep = width;
  s = dy < 0 ? -1 : 0;
  dy = (dy ^ s) - s;
  istep = (istep ^ s) - s;
  s = dy > dx ? -1 : 0;
  dx ^= dy & s;
  dy ^= dx & s;
  dx ^= dy & s;
  bt_pix ^= istep & s;
  istep ^= bt_pix & s;
  bt_pix ^= istep & s;
  int err = dx - (dy + dy);
  const int plus_delta = dx + dx;
  const int minus_delta = -(dy + dy);
  const int plus_step = istep;
  const int minus_step = bt_pix;
  long at = static_cast<long>(y0) * width + x0;
  for (int k = 0; k <= dx; ++k) {
    set(static_cast<size_t>(at));
    const int m = err < 0 ? -1 : 0;
    err += minus_delta + (plus_delta & m);
    at += minus_step + (plus_step & m);
  }
}

// cv::clipLine: the part of the segment inside [0, width) x [0, height);
// false if none.
bool ClipLine(int64_t width, int64_t height, int64_t* x1, int64_t* y1,
              int64_t* x2, int64_t* y2) {
  if (width <= 0 || height <= 0) return false;
  const int64_t right = width - 1;
  const int64_t bottom = height - 1;
  const auto code = [&](int64_t x, int64_t y) {
    return (x < 0) + (x > right) * 2 + (y < 0) * 4 + (y > bottom) * 8;
  };
  int c1 = code(*x1, *y1);
  int c2 = code(*x2, *y2);
  if ((c1 & c2) == 0 && (c1 | c2) != 0) {
    if (c1 & 12) {
      const int64_t a = c1 < 8 ? 0 : bottom;
      *x1 += static_cast<int64_t>(static_cast<double>(a - *y1) *
                                  static_cast<double>(*x2 - *x1) /
                                  static_cast<double>(*y2 - *y1));
      *y1 = a;
      c1 = (*x1 < 0) + (*x1 > right) * 2;
    }
    if (c2 & 12) {
      const int64_t a = c2 < 8 ? 0 : bottom;
      *x2 += static_cast<int64_t>(static_cast<double>(a - *y2) *
                                  static_cast<double>(*x2 - *x1) /
                                  static_cast<double>(*y2 - *y1));
      *y2 = a;
      c2 = (*x2 < 0) + (*x2 > right) * 2;
    }
    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
      if (c1) {
        const int64_t a = c1 == 1 ? 0 : right;
        *y1 += static_cast<int64_t>(static_cast<double>(a - *x1) *
                                    static_cast<double>(*y2 - *y1) /
                                    static_cast<double>(*x2 - *x1));
        *x1 = a;
        c1 = 0;
      }
      if (c2) {
        const int64_t a = c2 == 1 ? 0 : right;
        *y2 += static_cast<int64_t>(static_cast<double>(a - *x2) *
                                    static_cast<double>(*y2 - *y1) /
                                    static_cast<double>(*x2 - *x1));
        *x2 = a;
        c2 = 0;
      }
    }
  }
  return (c1 | c2) == 0;
}

struct PolyEdge {
  int y0 = 0;
  int y1 = 0;
  int64_t x = 0;
  int64_t dx = 0;
  PolyEdge* next = nullptr;
};

}  // namespace

void FillPoly8(const std::vector<std::pair<int64_t, int64_t>>& vertices,
               int shift, int width, int height, int32_t value,
               std::vector<int32_t>* image) {
  // cv::fillPoly(img, pts, color, LINE_8, shift): CollectPolyEdges then
  // FillEdgeCollection (drawing.cpp), x in 16.16 fixed point.
  GR_CHECK(shift >= 0 && shift <= 16);
  GR_CHECK(image->size() == static_cast<size_t>(width) * height);
  constexpr int kShift = 16;
  constexpr int64_t kOne = int64_t{1} << kShift;
  const size_t count = vertices.size();
  if (count == 0) return;
  const int64_t delta = (int64_t{1} << shift) >> 1;
  std::vector<PolyEdge> edges;
  edges.reserve(count + 1);
  const auto fixed = [&](const std::pair<int64_t, int64_t>& v) {
    return std::make_pair(v.first * (int64_t{1} << (kShift - shift)),
                          (v.second + delta) >> shift);
  };
  std::pair<int64_t, int64_t> pt0 = fixed(vertices[count - 1]);
  for (size_t i = 0; i < count; ++i) {
    const std::pair<int64_t, int64_t> pt1 = fixed(vertices[i]);
    std::pair<int64_t, int64_t> pt0c = pt0;
    std::pair<int64_t, int64_t> pt1c = pt1;
    int64_t tx0 = (pt0.first + (kOne >> 1)) >> kShift;
    int64_t ty0 = pt0.second;
    int64_t tx1 = (pt1.first + (kOne >> 1)) >> kShift;
    int64_t ty1 = pt1.second;
    const bool outside =
        static_cast<uint64_t>(tx0) >= static_cast<uint64_t>(width) ||
        static_cast<uint64_t>(tx1) >= static_cast<uint64_t>(width) ||
        static_cast<uint64_t>(ty0) >= static_cast<uint64_t>(height) ||
        static_cast<uint64_t>(ty1) >= static_cast<uint64_t>(height);
    // The outline (cv::line through a clipping LineIterator).
    int64_t lx0 = tx0, ly0 = ty0, lx1 = tx1, ly1 = ty1;
    const bool visible =
        !outside || ClipLine(width, height, &lx0, &ly0, &lx1, &ly1);
    if (visible) {
      int ax = static_cast<int>(lx0), ay = static_cast<int>(ly0);
      int bx = static_cast<int>(lx1), by = static_cast<int>(ly1);
      if (bx < ax) {  // leftToRight.
        std::swap(ax, bx);
        std::swap(ay, by);
      }
      ForEachLine8Cell(ax, ay, bx, by, width,
                       [&](size_t at) { (*image)[at] = value; });
    }
    // Clipped end points give a more accurate edge.
    if (outside) {
      ClipLine(width, height, &tx0, &ty0, &tx1, &ty1);
      if (ty0 != ty1) {
        pt0c = {tx0 * kOne, ty0};
        pt1c = {tx1 * kOne, ty1};
      }
    } else {
      pt0c.first += kOne >> 1;
      pt1c.first += kOne >> 1;
    }
    if (pt0.second != pt1.second) {
      PolyEdge edge;
      edge.dx = (pt1c.first - pt0c.first) / (pt1c.second - pt0c.second);
      if (pt0.second < pt1.second) {
        edge.y0 = static_cast<int>(pt0.second);
        edge.y1 = static_cast<int>(pt1.second);
        edge.x = pt0c.first + (pt0.second - pt0c.second) * edge.dx;
      } else {
        edge.y0 = static_cast<int>(pt1.second);
        edge.y1 = static_cast<int>(pt0.second);
        edge.x = pt1c.first + (pt1.second - pt1c.second) * edge.dx;
      }
      edges.push_back(edge);
    }
    pt0 = pt1;
  }

  // FillEdgeCollection.
  const int total = static_cast<int>(edges.size());
  if (total < 2) return;
  int y_max = std::numeric_limits<int>::min();
  int y_min = std::numeric_limits<int>::max();
  int64_t x_max = std::numeric_limits<int64_t>::min();
  int64_t x_min = std::numeric_limits<int64_t>::max();
  for (const PolyEdge& e : edges) {
    const int64_t x1 = e.x + (e.y1 - e.y0) * e.dx;
    y_min = std::min(y_min, e.y0);
    y_max = std::max(y_max, e.y1);
    x_min = std::min({x_min, e.x, x1});
    x_max = std::max({x_max, e.x, x1});
  }
  if (y_max < 0 || y_min >= height || x_max < 0 ||
      x_min >= (static_cast<int64_t>(width) << kShift)) {
    return;
  }
  std::sort(edges.begin(), edges.end(),
            [](const PolyEdge& a, const PolyEdge& b) {
              if (a.y0 != b.y0) return a.y0 < b.y0;
              if (a.x != b.x) return a.x < b.x;
              return a.dx < b.dx;
            });
  PolyEdge tmp;
  tmp.y0 = std::numeric_limits<int>::max();
  edges.push_back(tmp);  // No more growth: pointers stay valid.
  int i = 0;
  tmp.next = nullptr;
  PolyEdge* e = &edges[0];
  y_max = std::min(y_max, height);
  for (int y = e->y0; y < y_max; ++y) {
    PolyEdge* prelast = &tmp;
    PolyEdge* last = tmp.next;
    bool draw = false;
    const bool clip = y < 0;
    while (last != nullptr || e->y0 == y) {
      if (last != nullptr && last->y1 == y) {
        // The edge ends: out of the active list.
        prelast->next = last->next;
        last = last->next;
        continue;
      }
      PolyEdge* keep_prelast = prelast;
      if (last != nullptr && (e->y0 > y || last->x < e->x)) {
        prelast = last;  // Next edge of the active list.
        last = last->next;
      } else if (i < total) {
        prelast->next = e;  // An edge starts.
        e->next = last;
        prelast = e;
        e = &edges[static_cast<size_t>(++i)];
      } else {
        break;
      }
      if (draw) {
        if (!clip) {
          // The edges carry the half-pixel offset: both ends round to
          // nearest (OpenCV 4.x).
          int x1, x2;
          if (keep_prelast->x > prelast->x) {
            x1 = static_cast<int>(prelast->x >> kShift);
            x2 = static_cast<int>(keep_prelast->x >> kShift);
          } else {
            x1 = static_cast<int>(keep_prelast->x >> kShift);
            x2 = static_cast<int>(prelast->x >> kShift);
          }
          if (x1 < width && x2 >= 0) {
            x1 = std::max(x1, 0);
            x2 = std::min(x2, width - 1);
            int32_t* row = image->data() + static_cast<size_t>(y) * width;
            for (int x = x1; x <= x2; ++x) row[x] = value;
          }
        }
        keep_prelast->x += keep_prelast->dx;
        prelast->x += prelast->dx;
      }
      draw = !draw;
    }
    // Bubble sort of the active list by x.
    PolyEdge* keep_prelast = nullptr;
    bool sorted;
    do {
      sorted = true;
      prelast = &tmp;
      last = tmp.next;
      while (last != keep_prelast && last->next != nullptr) {
        PolyEdge* te = last->next;
        if (last->x > te->x) {  // Swap.
          prelast->next = te;
          last->next = te->next;
          te->next = last;
          prelast = te;
          sorted = false;
        } else {
          prelast = last;
          last = te;
        }
      }
      keep_prelast = prelast;
    } while (!sorted && keep_prelast != tmp.next && keep_prelast != &tmp);
  }
}

void DrawLine8(int x0, int y0, int x1, int y1, int width, Mask* mask) {
  // cv::LineIterator(img, pt1, pt2, 8, leftToRight = true) as cv::line.
  int dx = x1 - x0;
  int dy = y1 - y0;
  int s = dx < 0 ? -1 : 0;
  dx = (dx ^ s) - s;
  dy = (dy ^ s) - s;
  x0 ^= (x0 ^ x1) & s;
  y0 ^= (y0 ^ y1) & s;
  int bt_pix = 1;
  int istep = width;
  s = dy < 0 ? -1 : 0;
  dy = (dy ^ s) - s;
  istep = (istep ^ s) - s;
  s = dy > dx ? -1 : 0;
  dx ^= dy & s;
  dy ^= dx & s;
  dx ^= dy & s;
  bt_pix ^= istep & s;
  istep ^= bt_pix & s;
  bt_pix ^= istep & s;
  int err = dx - (dy + dy);
  const int plus_delta = dx + dx;
  const int minus_delta = -(dy + dy);
  const int plus_step = istep;
  const int minus_step = bt_pix;
  long at = static_cast<long>(y0) * width + x0;
  for (int k = 0; k <= dx; ++k) {
    (*mask)[static_cast<size_t>(at)] = 1;
    const int m = err < 0 ? -1 : 0;
    err += minus_delta + (plus_delta & m);
    at += minus_step + (plus_step & m);
  }
}

std::vector<double> GaussianFilter(const std::vector<double>& image,
                                   int width, int height, double sigma) {
  GR_CHECK(image.size() == static_cast<size_t>(width) * height);
  const int radius = static_cast<int>(4.0 * sigma + 0.5);
  std::vector<double> weights(static_cast<size_t>(2 * radius + 1));
  double total = 0.0;
  for (int k = -radius; k <= radius; ++k) {
    weights[static_cast<size_t>(k + radius)] =
        std::exp(-0.5 * k * k / (sigma * sigma));
    total += weights[static_cast<size_t>(k + radius)];
  }
  for (double& w : weights) w /= total;
  // 'reflect': d c b a | a b c d | d c b a.
  const auto reflect = [](int t, int n) {
    if (n == 1) return 0;
    const int period = 2 * n;
    t %= period;
    if (t < 0) t += period;
    return t < n ? t : period - 1 - t;
  };
  // Axis 0 (rows) first, then axis 1, as scipy.
  std::vector<double> rows(image.size());
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      double sum = 0.0;
      for (int k = -radius; k <= radius; ++k) {
        sum += weights[static_cast<size_t>(k + radius)] *
               image[Cell(i, reflect(j + k, height), width)];
      }
      rows[Cell(i, j, width)] = sum;
    }
  }
  std::vector<double> out(image.size());
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      double sum = 0.0;
      for (int k = -radius; k <= radius; ++k) {
        sum += weights[static_cast<size_t>(k + radius)] *
               rows[Cell(reflect(i + k, width), j, width)];
      }
      out[Cell(i, j, width)] = sum;
    }
  }
  return out;
}

int OtsuThreshold(const std::vector<uint8_t>& values) {
  // cv::threshold's getThreshVal_Otsu_8u.
  if (values.empty()) return 0;
  double hist[256] = {0.0};
  for (uint8_t v : values) hist[v] += 1.0;
  const double scale = 1.0 / values.size();
  double mu = 0.0;
  for (int i = 0; i < 256; ++i) mu += i * hist[i];
  mu *= scale;
  double mu1 = 0.0;
  double q1 = 0.0;
  double max_sigma = 0.0;
  int max_val = 0;
  constexpr double kEps = 1.1920928955078125e-07;  // FLT_EPSILON
  for (int i = 0; i < 256; ++i) {
    const double p = hist[i] * scale;
    mu1 *= q1;
    q1 += p;
    const double q2 = 1.0 - q1;
    if (std::min(q1, q2) < kEps || std::max(q1, q2) > 1.0 - kEps) continue;
    mu1 = (mu1 + i * p) / q1;
    const double mu2 = (mu - q1 * mu1) / q2;
    const double sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > max_sigma) {
      max_sigma = sigma;
      max_val = i;
    }
  }
  return max_val;
}

std::vector<double> MaxFilter3x3(const std::vector<double>& grid, int width,
                                 int height) {
  GR_CHECK(grid.size() == static_cast<size_t>(width) * height);
  std::vector<double> out(grid.size());
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      double best = -std::numeric_limits<double>::infinity();
      for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          best = std::max(best,
                          grid[Cell(std::clamp(i + di, 0, width - 1),
                                    std::clamp(j + dj, 0, height - 1),
                                    width)]);
        }
      }
      out[Cell(i, j, width)] = best;
    }
  }
  return out;
}

Mask OpenLine(const Mask& mask, int width, int height, int length,
              bool along_x) {
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  GR_CHECK(length >= 1);
  const int anchor = length / 2;
  // Both passes take the window [x - anchor, x - anchor + length - 1].
  auto pass = [&](const Mask& in, bool erode) {
    Mask out(in.size(), 0);
    const int lines = along_x ? height : width;
    const int span = along_x ? width : height;
    for (int line = 0; line < lines; ++line) {
      for (int x = 0; x < span; ++x) {
        bool value = erode;
        for (int k = 0; k < length; ++k) {
          const int t = x - anchor + k;
          bool v;
          if (t < 0 || t >= span) {
            v = erode;  // Outside: set for erosion, unset for dilation.
          } else {
            v = in[along_x ? Cell(t, line, width) : Cell(line, t, width)] != 0;
          }
          if (erode && !v) {
            value = false;
            break;
          }
          if (!erode && v) {
            value = true;
            break;
          }
        }
        out[along_x ? Cell(x, line, width) : Cell(line, x, width)] = value;
      }
    }
    return out;
  };
  return pass(pass(mask, true), false);
}

Components Label8(const Mask& mask, int width, int height) {
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  Components out;
  out.label.assign(mask.size(), 0);
  std::vector<int> stack;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      const size_t start = Cell(i, j, width);
      if (!mask[start] || out.label[start]) continue;
      const int id = static_cast<int>(out.cells.size()) + 1;
      std::vector<int> cells;
      out.label[start] = id;
      stack.assign(1, static_cast<int>(start));
      while (!stack.empty()) {
        const int c = stack.back();
        stack.pop_back();
        cells.push_back(c);
        const int ci = c % width;
        const int cj = c / width;
        for (int dj = -1; dj <= 1; ++dj) {
          for (int di = -1; di <= 1; ++di) {
            const int ni = ci + di;
            const int nj = cj + dj;
            if (ni < 0 || nj < 0 || ni >= width || nj >= height) continue;
            const size_t n = Cell(ni, nj, width);
            if (!mask[n] || out.label[n]) continue;
            out.label[n] = id;
            stack.push_back(static_cast<int>(n));
          }
        }
      }
      std::sort(cells.begin(), cells.end());  // Raster order.
      out.cells.push_back(std::move(cells));
    }
  }
  return out;
}

std::vector<float> DistanceTransform(const std::vector<uint8_t>& sources,
                                     int width, int height) {
  GR_CHECK(width >= 0 && height >= 0);
  GR_CHECK(sources.size() == static_cast<size_t>(width) * height);
  std::vector<double> grid(sources.size());
  for (size_t i = 0; i < sources.size(); ++i) {
    grid[i] = sources[i] != 0 ? 0.0 : kBig;
  }
  const int n = std::max(width, height);
  std::vector<double> f(static_cast<size_t>(n)), d(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n) + 1);
  std::vector<int> v(static_cast<size_t>(n));
  for (int u = 0; u < width; ++u) {
    for (int r = 0; r < height; ++r) f[r] = grid[Index(u, r, width)];
    SquaredDistance1d(f, height, d, v, z);
    for (int r = 0; r < height; ++r) grid[Index(u, r, width)] = d[r];
  }
  for (int r = 0; r < height; ++r) {
    for (int u = 0; u < width; ++u) f[u] = grid[Index(u, r, width)];
    SquaredDistance1d(f, width, d, v, z);
    for (int u = 0; u < width; ++u) grid[Index(u, r, width)] = d[u];
  }
  std::vector<float> out(grid.size());
  for (size_t i = 0; i < grid.size(); ++i) {
    out[i] = grid[i] >= kBig * 0.5
                 ? std::numeric_limits<float>::infinity()
                 : static_cast<float>(std::sqrt(grid[i]));
  }
  return out;
}

std::vector<uint8_t> ThinMask(const std::vector<uint8_t>& mask, int width,
                              int height) {
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  std::vector<uint8_t> image(mask.size());
  for (size_t i = 0; i < mask.size(); ++i) image[i] = mask[i] != 0 ? 1 : 0;
  auto at = [&](int u, int v) -> int {
    if (u < 0 || v < 0 || u >= width || v >= height) return 0;
    return image[Index(u, v, width)];
  };
  std::vector<size_t> remove;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int pass = 0; pass < 2; ++pass) {
      remove.clear();
      for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
          if (image[Index(u, v, width)] == 0) continue;
          // P2..P9 clockwise from north.
          const int p[8] = {at(u, v - 1),     at(u + 1, v - 1),
                            at(u + 1, v),     at(u + 1, v + 1),
                            at(u, v + 1),     at(u - 1, v + 1),
                            at(u - 1, v),     at(u - 1, v - 1)};
          int b = 0;
          int a = 0;
          for (int k = 0; k < 8; ++k) {
            b += p[k];
            if (p[k] == 0 && p[(k + 1) % 8] == 1) ++a;
          }
          if (b < 2 || b > 6 || a != 1) continue;
          const bool keep =
              pass == 0 ? (p[0] * p[2] * p[4] != 0 || p[2] * p[4] * p[6] != 0)
                        : (p[0] * p[2] * p[6] != 0 || p[0] * p[4] * p[6] != 0);
          if (!keep) remove.push_back(Index(u, v, width));
        }
      }
      for (size_t i : remove) image[i] = 0;
      changed |= !remove.empty();
    }
  }
  return image;
}

Mask Skeletonize(const Mask& mask, int width, int height) {
  // scikit-image's table (skimage/morphology/_skeletonize_cy.pyx, BSD-3):
  // per 8-neighbour code (NW 1, N 2, NE 4, E 8, SE 16, S 32, SW 64, W 128),
  // removable in the first pass (1), the second (2) or both (3).
  static const uint8_t kLut[256] = {
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
  };
  // A one-cell border of zeros, as scikit-image pads the image.
  const int w = width + 2;
  const int h = height + 2;
  Mask skeleton(static_cast<size_t>(w) * h, 0);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      skeleton[static_cast<size_t>(r + 1) * w + c + 1] =
          mask[static_cast<size_t>(r) * width + c] != 0;
    }
  }
  Mask cleaned = skeleton;
  bool removed = true;
  while (removed) {
    removed = false;
    for (int pass = 0; pass < 2; ++pass) {
      for (int r = 1; r < h - 1; ++r) {
        for (int c = 1; c < w - 1; ++c) {
          const size_t at = static_cast<size_t>(r) * w + c;
          if (!skeleton[at]) continue;
          const auto s = [&](int dr, int dc) {
            return skeleton[static_cast<size_t>(r + dr) * w + c + dc];
          };
          const uint8_t code = kLut[s(-1, -1) + 2 * s(-1, 0) + 4 * s(-1, 1) +
                                    8 * s(0, 1) + 16 * s(1, 1) + 32 * s(1, 0) +
                                    64 * s(1, -1) + 128 * s(0, -1)];
          if (code == 3 || (code == 1 && pass == 0) ||
              (code == 2 && pass == 1)) {
            cleaned[at] = 0;
            removed = true;
          }
        }
      }
      skeleton = cleaned;
    }
  }
  Mask out(static_cast<size_t>(width) * height);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      out[static_cast<size_t>(r) * width + c] =
          skeleton[static_cast<size_t>(r + 1) * w + c + 1];
    }
  }
  return out;
}

Mask FillPolygon(const std::vector<std::pair<int, int>>& vertices, int width,
                 int height) {
  Mask out(static_cast<size_t>(width) * height, 0);
  const size_t n = vertices.size();
  if (n == 0) return out;
  int ymin = vertices[0].second;
  int ymax = ymin;
  for (const auto& [x, y] : vertices) {
    ymin = std::min(ymin, y);
    ymax = std::max(ymax, y);
  }
  auto set = [&](int x, int y) {
    if (x >= 0 && y >= 0 && x < width && y < height) {
      out[static_cast<size_t>(y) * width + x] = 1;
    }
  };
  for (int y = std::max(ymin, 0); y <= std::min(ymax, height - 1); ++y) {
    std::vector<double> xs;
    for (size_t i = 0; i < n; ++i) {
      const auto [x0, y0] = vertices[i];
      const auto [x1, y1] = vertices[(i + 1) % n];
      if (y0 == y1) {  // Horizontal edge: its cells are on the boundary.
        if (y == y0) {
          for (int x = std::min(x0, x1); x <= std::max(x0, x1); ++x) set(x, y);
        }
        continue;
      }
      // Half-open in y so that shared vertices count once.
      if ((y >= std::min(y0, y1)) && (y < std::max(y0, y1))) {
        xs.push_back(x0 + static_cast<double>(y - y0) * (x1 - x0) / (y1 - y0));
      }
      // The top vertex of the edge is on the boundary too.
      if (y == std::max(y0, y1)) set(y == y0 ? x0 : x1, y);
    }
    std::sort(xs.begin(), xs.end());
    for (size_t k = 0; k + 1 < xs.size(); k += 2) {
      const int xa = static_cast<int>(std::ceil(xs[k] - 1e-9));
      const int xb = static_cast<int>(std::floor(xs[k + 1] + 1e-9));
      for (int x = xa; x <= xb; ++x) set(x, y);
    }
    // Edge crossings are boundary cells.
    for (double x : xs) set(static_cast<int>(std::lround(x)), y);
  }
  return out;
}

}  // namespace gr
