#include "gr/core/numeric.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Results that should be exact, and results of a few operations.
constexpr double kExactTolerance = 1e-15;
constexpr double kTolerance = 1e-12;

GR_TEST(NumericTest, InterpolationClampsLikeNumpy) {
  const std::vector<double> sample_x{0.0, 1.0, 3.0};
  const std::vector<double> sample_y{10.0, 20.0, 0.0};
  struct Case {
    double query;
    double expected;
    double tolerance;
  };
  // Below the samples, inside both pieces and above the samples.
  const Case cases[] = {{-1.0, 10.0, kExactTolerance},
                        {0.5, 15.0, kTolerance},
                        {2.0, 10.0, kTolerance},
                        {5.0, 0.0, kExactTolerance}};
  for (const Case& test_case : cases) {
    GR_EXPECT_NEAR(Interp1(sample_x, sample_y, test_case.query),
                   test_case.expected, test_case.tolerance);
  }
}

GR_TEST(NumericTest, OrderStatisticsMatchNumpy) {
  GR_EXPECT_NEAR(Median({3.0, 1.0, 2.0}), 2.0, kExactTolerance);
  GR_EXPECT_NEAR(Median({4.0, 1.0, 3.0, 2.0}), 2.5, kExactTolerance);
  GR_EXPECT_TRUE(std::isnan(Median({})));
  // numpy.percentile([1, 2, 3, 4], 95) == 3.85
  constexpr double kPercent = 95.0;
  GR_EXPECT_NEAR(Percentile({4.0, 1.0, 3.0, 2.0}, kPercent), 3.85, kTolerance);
}

GR_TEST(NumericTest, GradientIsSecondOrderInside) {
  // f = x^2 on a non-uniform grid: exact derivative 2x inside.
  const std::vector<double> coordinates{0.0, 0.5, 1.5, 2.0, 3.5};
  std::vector<double> values;
  for (double x : coordinates) values.push_back(x * x);
  const std::vector<double> gradient = Gradient(values, coordinates);
  for (size_t i = 1; i + 1 < coordinates.size(); ++i) {
    GR_EXPECT_NEAR(gradient[i], 2.0 * coordinates[i], kTolerance);
  }
  // One-sided at the ends: (0.5^2 - 0^2) / 0.5.
  GR_EXPECT_NEAR(gradient.front(), 0.5, kTolerance);
}

}  // namespace
}  // namespace gr
