// numpy-compatible numerical helpers: interpolation, order statistics and
// gradients, so that results match the Python tooling that consumes them.

#ifndef GARAGE_RECON_GR_CORE_NUMERIC_H_
#define GARAGE_RECON_GR_CORE_NUMERIC_H_

#include <utility>
#include <vector>

namespace gr {

// numpy.interp(query, sample_x, sample_y): the piecewise-linear function
// through the samples at `query`, clamped outside
// [sample_x.front(), sample_x.back()]. `sample_x` must be ascending and as
// long as `sample_y`.
double Interp1(const std::vector<double>& sample_x,
               const std::vector<double>& sample_y, double query);

// numpy.median / numpy.percentile (linear interpolation, `percent` in
// [0, 100]). Both return NaN for empty input.
double Median(std::vector<double> values);
double Percentile(std::vector<double> values, double percent);

// scipy.optimize.linear_sum_assignment (the shortest augmenting path method
// of its rectangular_lsap.cpp, with the same tie-breaking): a minimum-cost
// assignment of min(rows, columns) pairs for a finite rows x columns cost
// matrix (row-major), as (row, column) pairs with ascending rows.
std::vector<std::pair<int, int>> LinearSumAssignment(
    const std::vector<double>& cost, int rows, int columns);

// numpy.gradient(values, coordinates): second-order central differences
// inside, first order at the ends.
std::vector<double> Gradient(const std::vector<double>& values,
                             const std::vector<double>& coordinates);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_NUMERIC_H_
