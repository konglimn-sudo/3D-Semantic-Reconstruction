// numpy-compatible numerical helpers: interpolation, order statistics and
// gradients, so that results match the Python tooling that consumes them.

#ifndef GARAGE_RECON_GR_NUMERIC_H_
#define GARAGE_RECON_GR_NUMERIC_H_

#include <vector>

namespace gr {

// numpy.interp: piecewise-linear y(x) at `query`, clamped outside
// [x.front(), x.back()]. `x` must be ascending and as long as `y`.
double Interp1(const std::vector<double>& x, const std::vector<double>& y,
               double query);

// numpy.median / numpy.percentile (linear interpolation, q in [0, 100]).
// Both return NaN for empty input.
double Median(std::vector<double> values);
double Percentile(std::vector<double> values, double q);

// numpy.gradient(f, x): second-order central differences inside, first order
// at the ends.
std::vector<double> Gradient(const std::vector<double>& f,
                             const std::vector<double>& x);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NUMERIC_H_
