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
