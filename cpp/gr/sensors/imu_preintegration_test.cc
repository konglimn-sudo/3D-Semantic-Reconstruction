#include "gr/sensors/imu_preintegration.h"

#include <cmath>
#include <optional>
#include <random>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Body on a horizontal circle: constant yaw rate and speed, 400 Hz samples
// from t = 0 to 5 s, with the given constant biases added.
struct Circle {
  double yaw_rate = 0.4;
  double speed = 3.0;

  Pose PoseAt(double t) const {
    const double yaw = yaw_rate * t;
    const double radius = speed / yaw_rate;
    return Pose(
        So3Exp(Vec3(0.0, 0.0, yaw)),
        Vec3(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0));
  }
  Vec3 VelocityAt(double t) const {
    const double yaw = yaw_rate * t;
    return speed * Vec3(std::cos(yaw), std::sin(yaw), 0.0);
  }
  ImuSamples Samples(const Vec6& bias) const {
    ImuSamples imu;
    for (int i = 0; i <= 2000; ++i) {
      imu.stamps.push_back(0.0025 * i);
      imu.gyro.push_back(Vec3(0.0, 0.0, yaw_rate) + bias.head<3>());
      // Specific force in the body frame: centripetal acceleration (+y) and
      // the reaction to gravity (+z).
      imu.accel.push_back(Vec3(0.0, speed * yaw_rate, kGravity) +
                          bias.tail<3>());
    }
    return imu;
  }
};

GR_TEST(ImuPreintegrationTest, PredictsACircularDrive) {
  const Circle circle;
  const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
      circle.Samples(Vec6::Zero()), 1.0, 2.5, Vec6::Zero(), ImuNoise());
  GR_ASSERT_TRUE(preintegration.has_value());
  GR_EXPECT_NEAR(preintegration->dt(), 1.5, 1e-12);
  Pose predicted;
  Vec3 velocity;
  preintegration->Predict(circle.PoseAt(1.0), circle.VelocityAt(1.0),
                          Vec6::Zero(), predicted, velocity);
  const Pose truth = circle.PoseAt(2.5);
  // Piecewise-constant rates over 2.5 ms steps: sub-millimetre error.
  GR_EXPECT_NEAR((predicted.translation - truth.translation).norm(), 0.0, 1e-3);
  GR_EXPECT_NEAR(RotationAngle(predicted.rotation.transpose() * truth.rotation),
                 0.0, 1e-9);
  GR_EXPECT_NEAR((velocity - circle.VelocityAt(2.5)).norm(), 0.0, 1e-3);
  const Vec9 residual =
      preintegration->Residual(circle.PoseAt(1.0), circle.VelocityAt(1.0),
                               truth, circle.VelocityAt(2.5), Vec6::Zero());
  GR_EXPECT_LT(residual.norm(), 2e-3);
  // Covariance grows with time and stays symmetric positive definite.
  const Mat9& covariance = preintegration->covariance();
  GR_EXPECT_LT((covariance - covariance.transpose()).norm(), 1e-15);
  GR_EXPECT_GT(covariance.ldlt().vectorD().minCoeff(), 0.0);
}

GR_TEST(ImuPreintegrationTest, FirstOrderBiasCorrection) {
  const Circle circle;
  Vec6 bias;
  bias << 0.003, -0.002, 0.004, 0.05, -0.03, 0.04;
  // Integrated with a zero linearisation point; evaluated at the true bias.
  const std::optional<ImuPreintegration> preintegration =
      PreintegrateImu(circle.Samples(bias), 0.5, 1.5, Vec6::Zero(), ImuNoise());
  GR_ASSERT_TRUE(preintegration.has_value());
  const Pose pose_i = circle.PoseAt(0.5);
  const Pose pose_j = circle.PoseAt(1.5);
  const Vec3 velocity_i = circle.VelocityAt(0.5);
  const Vec3 velocity_j = circle.VelocityAt(1.5);
  const Vec9 uncorrected = preintegration->Residual(pose_i, velocity_i, pose_j,
                                                    velocity_j, Vec6::Zero());
  const Vec9 corrected =
      preintegration->Residual(pose_i, velocity_i, pose_j, velocity_j, bias);
  GR_EXPECT_GT(uncorrected.norm(), 0.01);
  GR_EXPECT_LT(corrected.norm(), 1e-3);
}

GR_TEST(ImuPreintegrationTest, JacobianMatchesFiniteDifferences) {
  std::mt19937 rng(7);
  std::normal_distribution<double> normal(0.0, 1.0);
  const auto random_vec = [&](double scale) -> Vec3 {
    return Vec3(normal(rng), normal(rng), normal(rng)) * scale;
  };
  ImuSamples imu;
  for (int i = 0; i <= 200; ++i) {
    imu.stamps.push_back(0.005 * i);
    imu.gyro.push_back(random_vec(0.3));
    imu.accel.push_back(random_vec(1.0) + Vec3(0.0, 0.0, kGravity));
  }
  Vec6 linearisation;
  linearisation << random_vec(0.01), random_vec(0.1);
  const std::optional<ImuPreintegration> preintegration =
      PreintegrateImu(imu, 0.1, 0.8, linearisation, ImuNoise());
  GR_ASSERT_TRUE(preintegration.has_value());

  const Pose pose_i(So3Exp(random_vec(0.5)), random_vec(2.0));
  const Pose pose_j(So3Exp(random_vec(0.5)), random_vec(2.0));
  const Vec3 velocity_i = random_vec(1.0);
  const Vec3 velocity_j = random_vec(1.0);
  Vec6 bias;
  bias << random_vec(0.01), random_vec(0.1);
  ImuPreintegration::Jacobian analytic;
  preintegration->Residual(pose_i, velocity_i, pose_j, velocity_j, bias,
                           &analytic);

  const double h = 1e-6;
  for (int column = 0; column < 24; ++column) {
    Vec9 plus;
    Vec9 minus;
    for (const double sign : {1.0, -1.0}) {
      Pose pi = pose_i;
      Pose pj = pose_j;
      Vec3 vi = velocity_i;
      Vec3 vj = velocity_j;
      Vec6 b = bias;
      const int block = column / 3;
      const int axis = column % 3;
      Vec3 delta = Vec3::Zero();
      delta[axis] = sign * h;
      switch (block) {
        case 0:
          pi.rotation = pi.rotation * So3Exp(delta);
          break;
        case 1:
          pi.translation += delta;
          break;
        case 2:
          vi += delta;
          break;
        case 3:
          pj.rotation = pj.rotation * So3Exp(delta);
          break;
        case 4:
          pj.translation += delta;
          break;
        case 5:
          vj += delta;
          break;
        case 6:
          b.head<3>() += delta;
          break;
        default:
          b.tail<3>() += delta;
          break;
      }
      (sign > 0 ? plus : minus) = preintegration->Residual(pi, vi, pj, vj, b);
    }
    const Vec9 numeric = (plus - minus) / (2.0 * h);
    GR_EXPECT_NEAR((numeric - analytic.col(column)).norm(), 0.0, 1e-5);
  }
}

}  // namespace
}  // namespace gr
