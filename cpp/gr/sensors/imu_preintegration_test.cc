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
  static constexpr int kSamples = 2000;
  static constexpr double kSamplePeriod = 0.0025;  // s

  double yaw_rate = 0.4;  // rad/s
  double speed = 3.0;     // m/s

  Pose PoseAt(double time) const {
    const double yaw = yaw_rate * time;
    const double radius = speed / yaw_rate;
    return Pose(
        So3Exp(Vec3(0.0, 0.0, yaw)),
        Vec3(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0));
  }
  Vec3 VelocityAt(double time) const {
    const double yaw = yaw_rate * time;
    return speed * Vec3(std::cos(yaw), std::sin(yaw), 0.0);
  }
  ImuSamples Samples(const Vec6& bias) const {
    ImuSamples imu;
    for (int i = 0; i <= kSamples; ++i) {
      imu.stamps.push_back(kSamplePeriod * i);
      imu.gyro.push_back(Vec3(0.0, 0.0, yaw_rate) + bias.head<kSpaceDim>());
      // Specific force in the body frame: centripetal acceleration (+y) and
      // the reaction to gravity (+z).
      imu.accel.push_back(Vec3(0.0, speed * yaw_rate, kGravity) +
                          bias.tail<kSpaceDim>());
    }
    return imu;
  }
};

GR_TEST(ImuPreintegrationTest, PredictsACircularDrive) {
  constexpr double kStart = 1.0;  // s
  constexpr double kEnd = 2.5;
  // Piecewise-constant rates over 2.5 ms steps: sub-millimetre error.
  constexpr double kTranslationTolerance = 1e-3;  // m
  constexpr double kVelocityTolerance = 1e-3;     // m/s
  constexpr double kRotationTolerance = 1e-9;     // rad
  constexpr double kMaxResidual = 2e-3;
  constexpr double kSymmetryTolerance = 1e-15;
  constexpr double kDurationTolerance = 1e-12;  // s
  const Circle circle;
  const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
      circle.Samples(Vec6::Zero()), kStart, kEnd, Vec6::Zero(), ImuNoise());
  GR_ASSERT_TRUE(preintegration.has_value());
  GR_EXPECT_NEAR(preintegration->dt(), kEnd - kStart, kDurationTolerance);
  Pose predicted;
  Vec3 velocity;
  preintegration->Predict(circle.PoseAt(kStart), circle.VelocityAt(kStart),
                          Vec6::Zero(), &predicted, &velocity);
  const Pose truth = circle.PoseAt(kEnd);
  GR_EXPECT_NEAR((predicted.translation - truth.translation).norm(), 0.0,
                 kTranslationTolerance);
  GR_EXPECT_NEAR(RotationAngle(predicted.rotation.transpose() * truth.rotation),
                 0.0, kRotationTolerance);
  GR_EXPECT_NEAR((velocity - circle.VelocityAt(kEnd)).norm(), 0.0,
                 kVelocityTolerance);
  const Vec9 residual =
      preintegration->Residual(circle.PoseAt(kStart), circle.VelocityAt(kStart),
                               truth, circle.VelocityAt(kEnd), Vec6::Zero());
  GR_EXPECT_LT(residual.norm(), kMaxResidual);
  // Covariance grows with time and stays symmetric positive definite.
  const Mat9& covariance = preintegration->covariance();
  GR_EXPECT_LT((covariance - covariance.transpose()).norm(),
               kSymmetryTolerance);
  GR_EXPECT_GT(covariance.ldlt().vectorD().minCoeff(), 0.0);
}

GR_TEST(ImuPreintegrationTest, FirstOrderBiasCorrection) {
  constexpr double kStart = 0.5;  // s
  constexpr double kEnd = 1.5;
  // The biases spoil the residual; the first-order correction removes it.
  constexpr double kMinUncorrected = 0.01;
  constexpr double kMaxCorrected = 1e-3;
  const Circle circle;
  Vec6 bias;
  bias << 0.003, -0.002, 0.004, 0.05, -0.03, 0.04;
  // Integrated with a zero linearisation point; evaluated at the true bias.
  const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
      circle.Samples(bias), kStart, kEnd, Vec6::Zero(), ImuNoise());
  GR_ASSERT_TRUE(preintegration.has_value());
  const Pose pose_i = circle.PoseAt(kStart);
  const Pose pose_j = circle.PoseAt(kEnd);
  const Vec3 velocity_i = circle.VelocityAt(kStart);
  const Vec3 velocity_j = circle.VelocityAt(kEnd);
  const Vec9 uncorrected = preintegration->Residual(pose_i, velocity_i, pose_j,
                                                    velocity_j, Vec6::Zero());
  const Vec9 corrected =
      preintegration->Residual(pose_i, velocity_i, pose_j, velocity_j, bias);
  GR_EXPECT_GT(uncorrected.norm(), kMinUncorrected);
  GR_EXPECT_LT(corrected.norm(), kMaxCorrected);
}

GR_TEST(ImuPreintegrationTest, JacobianMatchesFiniteDifferences) {
  // 1 s of random motion sampled at 200 Hz, preintegrated over [0.1, 0.8].
  constexpr int kSeed = 7;
  constexpr int kSamples = 200;
  constexpr double kSamplePeriod = 0.005;  // s
  constexpr double kStart = 0.1;
  constexpr double kEnd = 0.8;
  // Standard deviations of the random states.
  constexpr double kRateScale = 0.3;
  constexpr double kForceScale = 1.0;
  constexpr double kGyroBiasScale = 0.01;
  constexpr double kAccelBiasScale = 0.1;
  constexpr double kRotationScale = 0.5;
  constexpr double kPositionScale = 2.0;
  constexpr double kVelocityScale = 1.0;
  constexpr double kStep = 1e-6;
  constexpr double kDerivativeTolerance = 1e-5;
  std::mt19937 rng(kSeed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const auto random_vec = [&](double scale) -> Vec3 {
    return Vec3(normal(rng), normal(rng), normal(rng)) * scale;
  };
  ImuSamples imu;
  for (int i = 0; i <= kSamples; ++i) {
    imu.stamps.push_back(kSamplePeriod * i);
    imu.gyro.push_back(random_vec(kRateScale));
    imu.accel.push_back(random_vec(kForceScale) + Vec3(0.0, 0.0, kGravity));
  }
  Vec6 linearisation;
  linearisation << random_vec(kGyroBiasScale), random_vec(kAccelBiasScale);
  const std::optional<ImuPreintegration> preintegration =
      PreintegrateImu(imu, kStart, kEnd, linearisation, ImuNoise());
  GR_ASSERT_TRUE(preintegration.has_value());

  const Pose pose_i(So3Exp(random_vec(kRotationScale)),
                    random_vec(kPositionScale));
  const Pose pose_j(So3Exp(random_vec(kRotationScale)),
                    random_vec(kPositionScale));
  const Vec3 velocity_i = random_vec(kVelocityScale);
  const Vec3 velocity_j = random_vec(kVelocityScale);
  Vec6 bias;
  bias << random_vec(kGyroBiasScale), random_vec(kAccelBiasScale);
  ImuPreintegration::Jacobian analytic;
  preintegration->Residual(pose_i, velocity_i, pose_j, velocity_j, bias,
                           &analytic);

  for (int column = 0; column < ImuPreintegration::kJacobianCols; ++column) {
    // Residuals of the positive and the negative step.
    Vec9 plus = Vec9::Zero();
    Vec9 minus = Vec9::Zero();
    for (const double sign : {1.0, -1.0}) {
      Pose perturbed_pose_i = pose_i;
      Pose perturbed_pose_j = pose_j;
      Vec3 perturbed_velocity_i = velocity_i;
      Vec3 perturbed_velocity_j = velocity_j;
      Vec6 perturbed_bias = bias;
      const int block_start = column - column % kSpaceDim;
      Vec3 delta = Vec3::Zero();
      delta[column % kSpaceDim] = sign * kStep;
      switch (block_start) {
        case ImuPreintegration::kRotationICol:
          perturbed_pose_i.rotation = perturbed_pose_i.rotation * So3Exp(delta);
          break;
        case ImuPreintegration::kPositionICol:
          perturbed_pose_i.translation += delta;
          break;
        case ImuPreintegration::kVelocityICol:
          perturbed_velocity_i += delta;
          break;
        case ImuPreintegration::kRotationJCol:
          perturbed_pose_j.rotation = perturbed_pose_j.rotation * So3Exp(delta);
          break;
        case ImuPreintegration::kPositionJCol:
          perturbed_pose_j.translation += delta;
          break;
        case ImuPreintegration::kVelocityJCol:
          perturbed_velocity_j += delta;
          break;
        case ImuPreintegration::kGyroBiasCol:
          perturbed_bias.segment<kSpaceDim>(kGyroBiasOffset) += delta;
          break;
        default:
          perturbed_bias.segment<kSpaceDim>(kAccelBiasOffset) += delta;
          break;
      }
      (sign > 0 ? plus : minus) = preintegration->Residual(
          perturbed_pose_i, perturbed_velocity_i, perturbed_pose_j,
          perturbed_velocity_j, perturbed_bias);
    }
    const Vec9 numeric = (plus - minus) / (2.0 * kStep);
    GR_EXPECT_NEAR((numeric - analytic.col(column)).norm(), 0.0,
                   kDerivativeTolerance);
  }
}

}  // namespace
}  // namespace gr
