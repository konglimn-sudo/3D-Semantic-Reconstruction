#include "gr/pose.h"

#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(PoseTest, So3ExpLogRoundTrip) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> uniform(-2.5, 2.5);
  for (int i = 0; i < 100; ++i) {
    const Vec3 omega(uniform(rng), uniform(rng), uniform(rng));
    if (omega.norm() > 3.0) continue;
    GR_EXPECT_NEAR((So3Log(So3Exp(omega)) - omega).norm(), 0.0, 1e-9);
  }
}

GR_TEST(PoseTest, RightJacobianInverseMatchesNumericDerivative) {
  // Log(Exp(phi) Exp(d)) ~= phi + Jr^-1(phi) d
  const Vec3 phi(0.3, -0.7, 0.4);
  const Mat3 jacobian = So3RightJacobianInverse(phi);
  const double h = 1e-6;
  for (int a = 0; a < 3; ++a) {
    Vec3 d = Vec3::Zero();
    d[a] = h;
    const Vec3 numeric = (So3Log(So3Exp(phi) * So3Exp(d)) - phi) / h;
    GR_EXPECT_NEAR((numeric - jacobian.col(a)).norm(), 0.0, 1e-5);
  }
}

GR_TEST(PoseTest, ComposeInvertAndInterpolate) {
  const Pose a(So3Exp(Vec3(0.1, -0.2, 0.3)), Vec3(1, 2, 3));
  const Pose b(So3Exp(Vec3(-0.3, 0.1, 0.2)), Vec3(-2, 0, 5));
  const Pose identity = a * a.Inverse();
  GR_EXPECT_NEAR((identity.Matrix() - Mat4::Identity()).norm(), 0.0, 1e-12);
  const Vec3 p(0.5, -1.0, 2.0);
  GR_EXPECT_NEAR(((a * b) * p - a * (b * p)).norm(), 0.0, 1e-12);
  GR_EXPECT_NEAR(
      (Pose::FromMatrix(a.Matrix()).translation - a.translation).norm(), 0.0,
      1e-15);
  const Pose half = Interpolate(a, b, 0.5);
  GR_EXPECT_NEAR(
      (half.translation - 0.5 * (a.translation + b.translation)).norm(), 0.0,
      1e-12);
  // Interpolating a sequence clamps outside the time range.
  const std::vector<double> times{1.0, 2.0};
  const std::vector<Pose> poses{a, b};
  GR_EXPECT_NEAR(
      (InterpolatePose(times, poses, 0.0).translation - a.translation).norm(),
      0.0, 1e-15);
  GR_EXPECT_NEAR(
      (InterpolatePose(times, poses, 1.5).translation - half.translation)
          .norm(),
      0.0, 1e-12);
  GR_EXPECT_NEAR(WrapAngle(3.0 * kPi), -kPi, 1e-12);
}

}  // namespace
}  // namespace gr
