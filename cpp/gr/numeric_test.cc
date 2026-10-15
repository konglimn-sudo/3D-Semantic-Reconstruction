#include "gr/numeric.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(NumericTest, InterpolationClampsLikeNumpy) {
  const std::vector<double> x{0.0, 1.0, 3.0};
  const std::vector<double> y{10.0, 20.0, 0.0};
  GR_EXPECT_NEAR(Interp1(x, y, -1.0), 10.0, 1e-15);
  GR_EXPECT_NEAR(Interp1(x, y, 0.5), 15.0, 1e-12);
  GR_EXPECT_NEAR(Interp1(x, y, 2.0), 10.0, 1e-12);
  GR_EXPECT_NEAR(Interp1(x, y, 5.0), 0.0, 1e-15);
}

GR_TEST(NumericTest, OrderStatisticsMatchNumpy) {
  GR_EXPECT_NEAR(Median({3.0, 1.0, 2.0}), 2.0, 1e-15);
  GR_EXPECT_NEAR(Median({4.0, 1.0, 3.0, 2.0}), 2.5, 1e-15);
  GR_EXPECT_TRUE(std::isnan(Median({})));
  // numpy.percentile([1, 2, 3, 4], 95) == 3.85
  GR_EXPECT_NEAR(Percentile({4.0, 1.0, 3.0, 2.0}, 95.0), 3.85, 1e-12);
}

GR_TEST(NumericTest, GradientIsSecondOrderInside) {
  // f = x^2 on a non-uniform grid: exact derivative 2x inside.
  const std::vector<double> x{0.0, 0.5, 1.5, 2.0, 3.5};
  std::vector<double> f;
  for (double v : x) f.push_back(v * v);
  const std::vector<double> g = Gradient(f, x);
  for (size_t i = 1; i + 1 < x.size(); ++i) {
    GR_EXPECT_NEAR(g[i], 2.0 * x[i], 1e-12);
  }
  GR_EXPECT_NEAR(g.front(), 0.5, 1e-12);  // One-sided at the ends.
}

}  // namespace
}  // namespace gr
