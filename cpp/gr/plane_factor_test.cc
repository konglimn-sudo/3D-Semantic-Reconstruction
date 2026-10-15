#include "gr/plane_factor.h"

#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

Pose Perturbed(const Pose& pose, const Vec6& delta) {
  return Pose(pose.rotation * So3Exp(delta.tail<3>()),
              pose.translation + delta.head<3>());
}

// Three scans of the plane z = 0.1 x - 0.2 y + 2 seen from different poses:
// the points of each scan, in the scan's frame, with optional noise along z.
struct PlaneScene {
  std::vector<Pose> poses;
  std::vector<std::vector<Vec3>> local_points;

  explicit PlaneScene(double noise) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uniform(-2.0, 2.0);
    std::normal_distribution<double> gauss(0.0, noise > 0.0 ? noise : 1.0);
    poses = {Pose(So3Exp(Vec3(0.1, -0.05, 0.3)), Vec3(1.0, 0.5, 0.2)),
             Pose(So3Exp(Vec3(-0.08, 0.02, -0.6)), Vec3(-0.5, 1.5, -0.1)),
             Pose(So3Exp(Vec3(0.02, 0.1, 1.2)), Vec3(0.3, -1.0, 0.4))};
    for (const Pose& pose : poses) {
      std::vector<Vec3> points;
      for (int i = 0; i < 60; ++i) {
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
    std::vector<PlaneObservation> out;
    for (size_t a = 0; a < local_points.size(); ++a) {
      PlaneObservation observation;
      observation.node = static_cast<int>(a);
      for (const Vec3& p : local_points[a]) observation.points.Add(p, 1.5);
      out.push_back(observation);
    }
    return out;
  }
};

GR_TEST(PlaneFactorTest, ClusterTransformMatchesTransformedPoints) {
  const PlaneScene scene(0.01);
  const Pose pose(So3Exp(Vec3(0.3, -0.2, 0.9)), Vec3(4.0, -2.0, 1.0));
  PointCluster local;
  PointCluster world;
  for (const Vec3& p : scene.local_points[0]) {
    local.Add(p, 0.7);
    world.Add(pose * p, 0.7);
  }
  const PointCluster moved = local.Transformed(pose);
  GR_EXPECT_NEAR(moved.weight(), world.weight(), 1e-12);
  GR_EXPECT_NEAR((moved.sum() - world.sum()).norm(), 0.0, 1e-9);
  GR_EXPECT_NEAR((moved.outer() - world.outer()).norm(), 0.0, 1e-8);
}

GR_TEST(PlaneFactorTest, CostIsTheSumOfSquaredDistancesToTheBestPlane) {
  const PlaneScene scene(0.01);
  const std::vector<PlaneObservation> observations = scene.Observations();
  const double cost = PlaneCost(observations, scene.poses);
  const ClusterPlane plane =
      FitClusterPlane(MergeObservations(observations, scene.poses));
  double explicit_cost = 0.0;
  for (size_t a = 0; a < scene.poses.size(); ++a) {
    for (const Vec3& p : scene.local_points[a]) {
      const double r = plane.normal.dot(scene.poses[a] * p) + plane.offset;
      explicit_cost += 1.5 * r * r;
    }
  }
  GR_EXPECT_NEAR(cost, explicit_cost, 1e-9);
  // The fitted plane is z = 0.1 x - 0.2 y + 2 up to the noise.
  const Vec3 expected = Vec3(0.1, -0.2, -1.0).normalized();
  GR_EXPECT_NEAR(std::abs(plane.normal.dot(expected)), 1.0, 1e-4);
}

GR_TEST(PlaneFactorTest, GradientMatchesNumericDerivativeOfTheCost) {
  const PlaneScene scene(0.02);
  const std::vector<PlaneObservation> observations = scene.Observations();
  const std::optional<PlaneLinearization> linearization =
      LinearizePlane(observations, scene.poses);
  GR_ASSERT_TRUE(linearization.has_value());
  GR_EXPECT_NEAR(linearization->cost, PlaneCost(observations, scene.poses),
                 1e-12);
  const double h = 1e-6;
  double worst = 0.0;
  double scale = 0.0;
  for (size_t a = 0; a < scene.poses.size(); ++a) {
    for (int k = 0; k < 6; ++k) {
      Vec6 delta = Vec6::Zero();
      delta[k] = h;
      std::vector<Pose> plus = scene.poses;
      std::vector<Pose> minus = scene.poses;
      plus[a] = Perturbed(scene.poses[a], delta);
      minus[a] = Perturbed(scene.poses[a], -delta);
      const double numeric =
          (PlaneCost(observations, plus) - PlaneCost(observations, minus)) /
          (2.0 * h);
      // The cost is sum r^2, so its gradient is twice J^T r.
      const double analytic =
          2.0 * linearization->gradient[static_cast<Eigen::Index>(6 * a) + k];
      worst = std::max(worst, std::abs(numeric - analytic));
      scale = std::max(scale, std::abs(numeric));
    }
  }
  GR_EXPECT_GT(scale, 1e-3);
  GR_EXPECT_LT(worst, 1e-5 * std::max(1.0, scale));
}

GR_TEST(PlaneFactorTest, EliminatedHessianPredictsTheCostOfSmallMotions) {
  // Noise-free points: zero residual, so Gauss-Newton is exact to second
  // order and cost(delta) = delta^T H delta.
  const PlaneScene scene(0.0);
  const std::vector<PlaneObservation> observations = scene.Observations();
  const std::optional<PlaneLinearization> linearization =
      LinearizePlane(observations, scene.poses);
  GR_ASSERT_TRUE(linearization.has_value());
  GR_EXPECT_NEAR(linearization->cost, 0.0, 1e-9);
  std::mt19937 rng(3);
  std::normal_distribution<double> gauss(0.0, 1e-4);
  for (int trial = 0; trial < 5; ++trial) {
    Eigen::VectorXd delta(18);
    for (Eigen::Index k = 0; k < delta.size(); ++k) delta[k] = gauss(rng);
    std::vector<Pose> moved = scene.poses;
    for (size_t a = 0; a < moved.size(); ++a) {
      moved[a] = Perturbed(scene.poses[a],
                           delta.segment<6>(static_cast<Eigen::Index>(6 * a)));
    }
    const double predicted = delta.dot(linearization->hessian * delta);
    const double actual = PlaneCost(observations, moved);
    GR_EXPECT_GT(actual, 0.0);
    GR_EXPECT_NEAR(predicted / actual, 1.0, 1e-3);
  }
  // Moving all scans together (a rigid motion of the whole scene) costs
  // nothing.
  Eigen::VectorXd rigid = Eigen::VectorXd::Zero(18);
  const Vec3 shift(0.01, -0.02, 0.005);
  for (int a = 0; a < 3; ++a) rigid.segment<3>(6 * a) = shift;
  GR_EXPECT_NEAR(rigid.dot(linearization->hessian * rigid), 0.0, 1e-12);
}

GR_TEST(PlaneFactorTest, LinePointsDoNotDefineAPlane) {
  PlaneObservation a;
  PlaneObservation b;
  b.node = 1;
  for (int i = 0; i < 20; ++i) {
    a.points.Add(Vec3(0.1 * i, 0.0, 0.0));
    b.points.Add(Vec3(0.1 * i + 0.05, 0.0, 0.0));
  }
  GR_EXPECT_FALSE(LinearizePlane({a, b}, {Pose(), Pose()}).has_value());
}

}  // namespace
}  // namespace gr
