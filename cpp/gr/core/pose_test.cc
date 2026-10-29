#include "gr/core/pose.h"

#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Results of a few operations, and results that should be exact.
constexpr double kTolerance = 1e-12;
constexpr double kExactTolerance = 1e-15;

GR_TEST(PoseTest, So3ExpLogRoundTrip) {
  constexpr int kSeed = 3;
  constexpr int kSamples = 100;
  constexpr double kMaxComponent = 2.5;
  // Log(Exp(omega)) == omega only up to an angle of pi; stay below it.
  constexpr double kMaxAngle = 3.0;
  constexpr double kRoundTripTolerance = 1e-9;
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(-kMaxComponent, kMaxComponent);
  for (int i = 0; i < kSamples; ++i) {
    const Vec3 omega(uniform(rng), uniform(rng), uniform(rng));
    if (omega.norm() > kMaxAngle) continue;
    GR_EXPECT_NEAR((So3Log(So3Exp(omega)) - omega).norm(), 0.0,
                   kRoundTripTolerance);
  }
}

GR_TEST(PoseTest, RightJacobianInverseMatchesNumericDerivative) {
  // Log(Exp(phi) Exp(d)) ~= phi + Jr^-1(phi) d
  constexpr double kStep = 1e-6;
  constexpr double kDerivativeTolerance = 1e-5;
  const Vec3 phi(0.3, -0.7, 0.4);
  const Mat3 jacobian = So3RightJacobianInverse(phi);
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    Vec3 step = Vec3::Zero();
    step[axis] = kStep;
    const Vec3 numeric = (So3Log(So3Exp(phi) * So3Exp(step)) - phi) / kStep;
    GR_EXPECT_NEAR((numeric - jacobian.col(axis)).norm(), 0.0,
                   kDerivativeTolerance);
  }
}

GR_TEST(PoseTest, ComposeInvertAndInterpolate) {
  constexpr double kHalfway = 0.5;
  const Pose first(So3Exp(Vec3(0.1, -0.2, 0.3)), Vec3(1, 2, 3));
  const Pose second(So3Exp(Vec3(-0.3, 0.1, 0.2)), Vec3(-2, 0, 5));
  const Pose identity = first * first.Inverse();
  GR_EXPECT_NEAR((identity.Matrix() - Mat4::Identity()).norm(), 0.0,
                 kTolerance);
  const Vec3 point(0.5, -1.0, 2.0);
  GR_EXPECT_NEAR(((first * second) * point - first * (second * point)).norm(),
                 0.0, kTolerance);
  GR_EXPECT_NEAR(
      (Pose::FromMatrix(first.Matrix()).translation - first.translation).norm(),
      0.0, kExactTolerance);
  const Pose half = Interpolate(first, second, kHalfway);
  GR_EXPECT_NEAR(
      (half.translation - kHalfway * (first.translation + second.translation))
          .norm(),
      0.0, kTolerance);
  // Interpolating a sequence clamps outside the time range.
  const std::vector<double> times{1.0, 2.0};
  const std::vector<Pose> poses{first, second};
  const double before_start = times.front() - 1.0;
  const double halfway =
      times.front() + kHalfway * (times.back() - times.front());
  GR_EXPECT_NEAR((InterpolatePose(times, poses, before_start).translation -
                  first.translation)
                     .norm(),
                 0.0, kExactTolerance);
  GR_EXPECT_NEAR(
      (InterpolatePose(times, poses, halfway).translation - half.translation)
          .norm(),
      0.0, kTolerance);
  // Three half turns wrap to minus one.
  GR_EXPECT_NEAR(WrapAngle(3.0 * kPi), -kPi, kTolerance);
}

}  // namespace
}  // namespace gr
