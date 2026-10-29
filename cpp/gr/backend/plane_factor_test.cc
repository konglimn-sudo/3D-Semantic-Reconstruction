#include "gr/backend/plane_factor.h"

#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kExactTolerance = 1e-12;
constexpr double kNumericTolerance = 1e-9;
// Weight of every point of a PlaneScene observation.
constexpr double kPointWeight = 1.5;
// Noise (m) of the noisy scenes.
constexpr double kSceneNoise = 0.01;

Pose Perturbed(const Pose& pose, const Vec6& delta) {
  return Pose(pose.rotation * So3Exp(delta.segment<kSpaceDim>(kRotationOffset)),
              pose.translation + delta.segment<kSpaceDim>(kTranslationOffset));
}

// Three scans of the plane z = 0.1 x - 0.2 y + 2 seen from different poses:
// the points of each scan, in the scan's frame, with optional noise along z.
struct PlaneScene {
  static constexpr int kSeed = 7;
  static constexpr int kPointsPerScan = 60;
  static constexpr double kHalfExtent = 2.0;  // m, in x and y

  std::vector<Pose> poses;
  std::vector<std::vector<Vec3>> local_points;

  explicit PlaneScene(double noise) {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<double> uniform(-kHalfExtent, kHalfExtent);
    std::normal_distribution<double> gauss(0.0, noise > 0.0 ? noise : 1.0);
    poses = {Pose(So3Exp(Vec3(0.1, -0.05, 0.3)), Vec3(1.0, 0.5, 0.2)),
             Pose(So3Exp(Vec3(-0.08, 0.02, -0.6)), Vec3(-0.5, 1.5, -0.1)),
             Pose(So3Exp(Vec3(0.02, 0.1, 1.2)), Vec3(0.3, -1.0, 0.4))};
    for (const Pose& pose : poses) {
      std::vector<Vec3> points;
      for (int point = 0; point < kPointsPerScan; ++point) {
        const double x = uniform(rng);
        const double y = uniform(rng);
        double z = 0.1 * x - 0.2 * y + 2.0;
        if (noise > 0.0) z += gauss(rng);
        points.push_back(pose.Inverse() * Vec3(x, y, z));
      }
      local_points.push_back(points);
    }
  }

  std::vector<PlaneObservation> Observations() const {
    std::vector<PlaneObservation> observations;
    for (size_t scan = 0; scan < local_points.size(); ++scan) {
      PlaneObservation observation;
      observation.node = static_cast<int>(scan);
      for (const Vec3& point : local_points[scan]) {
        observation.points.Add(point, kPointWeight);
      }
      observations.push_back(observation);
    }
    return observations;
  }
};

GR_TEST(PlaneFactorTest, ClusterTransformMatchesTransformedPoints) {
  constexpr double kWeight = 0.7;
  constexpr double kOuterTolerance = 1e-8;
  const PlaneScene scene(kSceneNoise);
  const Pose pose(So3Exp(Vec3(0.3, -0.2, 0.9)), Vec3(4.0, -2.0, 1.0));
  PointCluster local;
  PointCluster world;
  for (const Vec3& point : scene.local_points[0]) {
    local.Add(point, kWeight);
    world.Add(pose * point, kWeight);
  }
  const PointCluster moved = local.Transformed(pose);
  GR_EXPECT_NEAR(moved.weight(), world.weight(), kExactTolerance);
  GR_EXPECT_NEAR((moved.sum() - world.sum()).norm(), 0.0, kNumericTolerance);
  GR_EXPECT_NEAR((moved.outer() - world.outer()).norm(), 0.0, kOuterTolerance);
}

GR_TEST(PlaneFactorTest, CostIsTheSumOfSquaredDistancesToTheBestPlane) {
  constexpr double kNormalTolerance = 1e-4;
  const PlaneScene scene(kSceneNoise);
  const std::vector<PlaneObservation> observations = scene.Observations();
  const double cost = PlaneCost(observations, scene.poses);
  const ClusterPlane plane =
      FitClusterPlane(MergeObservations(observations, scene.poses));
  double explicit_cost = 0.0;
  for (size_t scan = 0; scan < scene.poses.size(); ++scan) {
    for (const Vec3& point : scene.local_points[scan]) {
      const double residual =
          plane.normal.dot(scene.poses[scan] * point) + plane.offset;
      explicit_cost += kPointWeight * residual * residual;
    }
  }
  GR_EXPECT_NEAR(cost, explicit_cost, kNumericTolerance);
  // The fitted plane is z = 0.1 x - 0.2 y + 2 up to the noise.
  const Vec3 expected = Vec3(0.1, -0.2, -1.0).normalized();
  GR_EXPECT_NEAR(std::abs(plane.normal.dot(expected)), 1.0, kNormalTolerance);
}

GR_TEST(PlaneFactorTest, GradientMatchesNumericDerivativeOfTheCost) {
  constexpr double kNoise = 0.02;  // m
  constexpr double kStep = 1e-6;
  constexpr double kMinGradient = 1e-3;
  constexpr double kRelativeTolerance = 1e-5;
  const PlaneScene scene(kNoise);
  const std::vector<PlaneObservation> observations = scene.Observations();
  const std::optional<PlaneLinearization> linearization =
      LinearizePlane(observations, scene.poses);
  GR_ASSERT_TRUE(linearization.has_value());
  GR_EXPECT_NEAR(linearization->cost, PlaneCost(observations, scene.poses),
                 kExactTolerance);
  double worst = 0.0;
  double scale = 0.0;
  for (size_t scan = 0; scan < scene.poses.size(); ++scan) {
    for (int axis = 0; axis < kPoseDim; ++axis) {
      Vec6 delta = Vec6::Zero();
      delta[axis] = kStep;
      std::vector<Pose> plus = scene.poses;
      std::vector<Pose> minus = scene.poses;
      plus[scan] = Perturbed(scene.poses[scan], delta);
      minus[scan] = Perturbed(scene.poses[scan], -delta);
      const double numeric =
          (PlaneCost(observations, plus) - PlaneCost(observations, minus)) /
          (2.0 * kStep);
      // The cost is sum r^2, so its gradient is twice J^T r.
      const double analytic =
          2.0 *
          linearization
              ->gradient[static_cast<Eigen::Index>(kPoseDim * scan) + axis];
      worst = std::max(worst, std::abs(numeric - analytic));
      scale = std::max(scale, std::abs(numeric));
    }
  }
  GR_EXPECT_GT(scale, kMinGradient);
  GR_EXPECT_LT(worst, kRelativeTolerance * std::max(1.0, scale));
}

GR_TEST(PlaneFactorTest, EliminatedHessianPredictsTheCostOfSmallMotions) {
  // Noise-free points: zero residual, so Gauss-Newton is exact to second
  // order and cost(delta) = delta^T H delta.
  constexpr int kSeed = 3;
  constexpr double kMotionSigma = 1e-4;
  constexpr int kTrials = 5;
  constexpr double kRatioTolerance = 1e-3;
  const PlaneScene scene(0.0);
  const int num_params = kPoseDim * static_cast<int>(scene.poses.size());
  const std::vector<PlaneObservation> observations = scene.Observations();
  const std::optional<PlaneLinearization> linearization =
      LinearizePlane(observations, scene.poses);
  GR_ASSERT_TRUE(linearization.has_value());
  GR_EXPECT_NEAR(linearization->cost, 0.0, kNumericTolerance);
  std::mt19937 rng(kSeed);
  std::normal_distribution<double> gauss(0.0, kMotionSigma);
  for (int trial = 0; trial < kTrials; ++trial) {
    Eigen::VectorXd delta(num_params);
    for (Eigen::Index param = 0; param < delta.size(); ++param) {
      delta[param] = gauss(rng);
    }
    std::vector<Pose> moved = scene.poses;
    for (size_t scan = 0; scan < moved.size(); ++scan) {
      moved[scan] = Perturbed(
          scene.poses[scan],
          delta.segment<kPoseDim>(static_cast<Eigen::Index>(kPoseDim * scan)));
    }
    const double predicted = delta.dot(linearization->hessian * delta);
    const double actual = PlaneCost(observations, moved);
    GR_EXPECT_GT(actual, 0.0);
    GR_EXPECT_NEAR(predicted / actual, 1.0, kRatioTolerance);
  }
  // Moving all scans together (a rigid motion of the whole scene) costs
  // nothing.
  Eigen::VectorXd rigid = Eigen::VectorXd::Zero(num_params);
  const Vec3 shift(0.01, -0.02, 0.005);
  for (int scan = 0; scan < static_cast<int>(scene.poses.size()); ++scan) {
    rigid.segment<kSpaceDim>(kPoseDim * scan + kTranslationOffset) = shift;
  }
  GR_EXPECT_NEAR(rigid.dot(linearization->hessian * rigid), 0.0,
                 kExactTolerance);
}

GR_TEST(PlaneFactorTest, LinePointsDoNotDefineAPlane) {
  constexpr int kPoints = 20;
  constexpr double kSpacing = 0.1;  // m
  constexpr double kOffset = 0.05;  // m, of the second scan's points
  PlaneObservation first;
  PlaneObservation second;
  second.node = 1;
  for (int point = 0; point < kPoints; ++point) {
    first.points.Add(Vec3(kSpacing * point, 0.0, 0.0));
    second.points.Add(Vec3(kSpacing * point + kOffset, 0.0, 0.0));
  }
  GR_EXPECT_FALSE(
      LinearizePlane({first, second}, {Pose(), Pose()}).has_value());
}

}  // namespace
}  // namespace gr
