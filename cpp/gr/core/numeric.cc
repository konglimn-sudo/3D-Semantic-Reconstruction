#include "gr/core/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Percentiles run from 0 to this.
constexpr double kMaxPercent = 100.0;

// A row or column without a partner, and AugmentingPath without a sink.
constexpr int kUnassigned = -1;

// The shortest augmenting path method of scipy's rectangular_lsap.cpp, for
// a cost matrix with at most as many rows as columns.
struct AssignmentState {
  AssignmentState(int row_count, int column_count)
      : row_dual(static_cast<size_t>(row_count), 0.0),
        column_dual(static_cast<size_t>(column_count), 0.0),
        column_of_row(static_cast<size_t>(row_count), kUnassigned),
        row_of_column(static_cast<size_t>(column_count), kUnassigned),
        path_row(static_cast<size_t>(column_count), kUnassigned),
        shortest_path_cost(static_cast<size_t>(column_count)),
        row_visited(static_cast<size_t>(row_count)),
        column_visited(static_cast<size_t>(column_count)),
        remaining_columns(static_cast<size_t>(column_count)) {}

  // The dual variables (scipy's u and v).
  std::vector<double> row_dual;
  std::vector<double> column_dual;
  // The assignment so far.
  std::vector<int> column_of_row;
  std::vector<int> row_of_column;
  // Per column, the row the shortest path reaches it from, and that path's
  // reduced cost.
  std::vector<int> path_row;
  std::vector<double> shortest_path_cost;
  // What the last path search visited.
  std::vector<uint8_t> row_visited;
  std::vector<uint8_t> column_visited;
  // Scratch: the columns the path search has not visited.
  std::vector<int> remaining_columns;
};

// The shortest augmenting path from `start_row` (scipy's augmenting_path):
// returns its sink column, kUnassigned if there is none, and sets *path_cost.
int AugmentingPath(const std::vector<double>& cost, int column_count,
                   int start_row, AssignmentState* state, double* path_cost) {
  std::vector<int>& remaining = state->remaining_columns;
  std::vector<double>& shortest = state->shortest_path_cost;
  double cost_so_far = 0.0;
  int remaining_count = column_count;
  for (int position = 0; position < column_count; ++position) {
    remaining[position] = column_count - position - 1;
  }
  std::fill(state->row_visited.begin(), state->row_visited.end(), 0);
  std::fill(state->column_visited.begin(), state->column_visited.end(), 0);
  std::fill(shortest.begin(), shortest.end(), kInf);
  int row = start_row;
  int sink = kUnassigned;
  while (sink == kUnassigned) {
    int closest_position = kUnassigned;
    double lowest_cost = kInf;
    state->row_visited[row] = 1;
    for (int position = 0; position < remaining_count; ++position) {
      const int column = remaining[position];
      const double reduced_cost =
          cost_so_far + cost[static_cast<size_t>(row) * column_count + column] -
          state->row_dual[row] - state->column_dual[column];
      if (reduced_cost < shortest[column]) {
        state->path_row[column] = row;
        shortest[column] = reduced_cost;
      }
      if (shortest[column] < lowest_cost ||
          (shortest[column] == lowest_cost &&
           state->row_of_column[column] == kUnassigned)) {
        lowest_cost = shortest[column];
        closest_position = position;
      }
    }
    cost_so_far = lowest_cost;
    if (cost_so_far == kInf) return kUnassigned;
    const int column = remaining[closest_position];
    if (state->row_of_column[column] == kUnassigned) {
      sink = column;
    } else {
      row = state->row_of_column[column];
    }
    state->column_visited[column] = 1;
    remaining[closest_position] = remaining[--remaining_count];
  }
  *path_cost = cost_so_far;
  return sink;
}

}  // namespace

double Interp1(const std::vector<double>& sample_x,
               const std::vector<double>& sample_y, double query) {
  GR_CHECK(!sample_x.empty() && sample_x.size() == sample_y.size());
  if (query <= sample_x.front()) return sample_y.front();
  if (query >= sample_x.back()) return sample_y.back();
  const size_t after = static_cast<size_t>(
      std::upper_bound(sample_x.begin(), sample_x.end(), query) -
      sample_x.begin());
  const size_t before = after - 1;
  const double slope = (sample_y[after] - sample_y[before]) /
                       (sample_x[after] - sample_x[before]);
  return sample_y[before] + slope * (query - sample_x[before]);
}

double Median(std::vector<double> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const size_t count = values.size();
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(count / 2);
  std::nth_element(values.begin(), middle, values.end());
  const double upper = *middle;
  if (count % 2 == 1) return upper;
  const double lower = *std::max_element(values.begin(), middle);
  return 0.5 * (lower + upper);
}

double Percentile(std::vector<double> values, double percent) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double position = std::clamp(percent, 0.0, kMaxPercent) / kMaxPercent *
                          static_cast<double>(values.size() - 1);
  const size_t below = static_cast<size_t>(std::floor(position));
  const auto below_it = values.begin() + static_cast<std::ptrdiff_t>(below);
  std::nth_element(values.begin(), below_it, values.end());
  const double lower_value = *below_it;
  const double upper_value = below + 1 < values.size()
                                 ? *std::min_element(below_it + 1, values.end())
                                 : lower_value;
  return lower_value +
         (upper_value - lower_value) * (position - static_cast<double>(below));
}

std::vector<double> Gradient(const std::vector<double>& values,
                             const std::vector<double>& coordinates) {
  // A difference needs two samples.
  constexpr size_t kMinSamples = 2;
  const size_t count = values.size();
  std::vector<double> out(count, 0.0);
  if (count < kMinSamples || coordinates.size() != count) return out;
  const size_t last = count - 1;
  out[0] = (values[1] - values[0]) / (coordinates[1] - coordinates[0]);
  out[last] = (values[last] - values[last - 1]) /
              (coordinates[last] - coordinates[last - 1]);
  for (size_t i = 1; i < last; ++i) {
    const double step_before = coordinates[i] - coordinates[i - 1];
    const double step_after = coordinates[i + 1] - coordinates[i];
    const double weight_before =
        -step_after / (step_before * (step_before + step_after));
    const double weight_here =
        (step_after - step_before) / (step_before * step_after);
    const double weight_after =
        step_before / (step_after * (step_before + step_after));
    out[i] = weight_before * values[i - 1] + weight_here * values[i] +
             weight_after * values[i + 1];
  }
  return out;
}

std::vector<std::pair<int, int>> LinearSumAssignment(
    const std::vector<double>& cost, int rows, int columns) {
  GR_CHECK(rows >= 0 && columns >= 0);
  GR_CHECK(cost.size() == static_cast<size_t>(rows) * columns);
  if (rows == 0 || columns == 0) return {};
  // Fewer rows than columns (transposed otherwise, as scipy does).
  const bool transpose = rows > columns;
  int row_count = rows;
  int column_count = columns;
  std::vector<double> oriented_cost = cost;
  if (transpose) {
    std::swap(row_count, column_count);
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        oriented_cost[static_cast<size_t>(column) * column_count + row] =
            cost[static_cast<size_t>(row) * columns + column];
      }
    }
  }
  AssignmentState state(row_count, column_count);
  for (int start_row = 0; start_row < row_count; ++start_row) {
    double path_cost = 0.0;
    const int sink = AugmentingPath(oriented_cost, column_count, start_row,
                                    &state, &path_cost);
    GR_CHECK(sink != kUnassigned);  // Finite costs are always feasible.
    // Update the duals ...
    state.row_dual[start_row] += path_cost;
    for (int row = 0; row < row_count; ++row) {
      if (state.row_visited[row] && row != start_row) {
        state.row_dual[row] +=
            path_cost - state.shortest_path_cost[state.column_of_row[row]];
      }
    }
    for (int column = 0; column < column_count; ++column) {
      if (state.column_visited[column]) {
        state.column_dual[column] -=
            path_cost - state.shortest_path_cost[column];
      }
    }
    // ... and augment the assignment along the path.
    int column = sink;
    while (true) {
      const int row = state.path_row[column];
      state.row_of_column[column] = row;
      std::swap(state.column_of_row[row], column);
      if (row == start_row) break;
    }
  }
  std::vector<std::pair<int, int>> out;
  for (int row = 0; row < row_count; ++row) {
    const int column = state.column_of_row[row];
    out.emplace_back(transpose ? column : row, transpose ? row : column);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace gr
