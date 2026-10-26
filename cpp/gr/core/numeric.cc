#include "gr/core/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "gr/core/check.h"

namespace gr {

double Interp1(const std::vector<double>& x, const std::vector<double>& y,
               double query) {
  GR_CHECK(!x.empty() && x.size() == y.size());
  if (query <= x.front()) return y.front();
  if (query >= x.back()) return y.back();
  const size_t j = static_cast<size_t>(
      std::upper_bound(x.begin(), x.end(), query) - x.begin());
  const double slope = (y[j] - y[j - 1]) / (x[j] - x[j - 1]);
  return y[j - 1] + slope * (query - x[j - 1]);
}

double Median(std::vector<double> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const size_t n = values.size();
  const auto half = static_cast<std::ptrdiff_t>(n / 2);
  std::nth_element(values.begin(), values.begin() + half, values.end());
  const double upper = values[n / 2];
  if (n % 2 == 1) return upper;
  const double lower = *std::max_element(values.begin(), values.begin() + half);
  return 0.5 * (lower + upper);
}

double Percentile(std::vector<double> values, double q) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double position = std::clamp(q, 0.0, 100.0) / 100.0 *
                          static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(position));
  const auto lo_it = values.begin() + static_cast<std::ptrdiff_t>(lo);
  std::nth_element(values.begin(), lo_it, values.end());
  const double a = *lo_it;
  const double b =
      lo + 1 < values.size() ? *std::min_element(lo_it + 1, values.end()) : a;
  return a + (b - a) * (position - static_cast<double>(lo));
}

std::vector<double> Gradient(const std::vector<double>& f,
                             const std::vector<double>& x) {
  const size_t n = f.size();
  std::vector<double> out(n, 0.0);
  if (n < 2 || x.size() != n) return out;
  out[0] = (f[1] - f[0]) / (x[1] - x[0]);
  out[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
  for (size_t i = 1; i + 1 < n; ++i) {
    const double dx1 = x[i] - x[i - 1];
    const double dx2 = x[i + 1] - x[i];
    const double a = -dx2 / (dx1 * (dx1 + dx2));
    const double b = (dx2 - dx1) / (dx1 * dx2);
    const double c = dx1 / (dx2 * (dx1 + dx2));
    out[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
  }
  return out;
}

}  // namespace gr
