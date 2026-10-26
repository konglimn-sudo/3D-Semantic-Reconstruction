#include "gr/registration/cloud_compare.h"

#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "gr/registration/registration.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

// The test garage scanned from a walk along both aisles.
Points GarageCloud(double noise, int seed) {
  const BoxScene scene = testing::MakeScene(testing::GarageStructure());
  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, noise);
  Points cloud;
  for (double x = -4.0; x <= 34.0; x += 3.0) {
    for (double y : {5.0, 15.0}) {
      const Pose pose(Mat3::Identity(), Vec3(x, y, 1.6));
      for (const Vec3& p :
           testing::ScanFromPose(scene, pose, 25.0).points_world) {
        cloud.push_back(p + Vec3(gauss(rng), gauss(rng), gauss(rng)));
      }
    }
  }
  return cloud;
}

GR_TEST(CloudCompareTest, CoarseAndFineRegistrationRecoverTheFrame) {
  const Points source = GarageCloud(0.005, 1);
  // The reference is the same garage in another frame (another session).
  const Pose reference_from_source(So3Exp(Vec3(0.0, 0.0, DegToRad(37.0))),
                                   Vec3(120.0, -48.0, 3.1));
  const Points reference =
      TransformPoints(GarageCloud(0.005, 2), reference_from_source);
  const std::optional<CoarseAlignment> coarse =
      AlignGravityAligned(source, reference);
  GR_ASSERT_TRUE(coarse.has_value());
  const Pose error =
      reference_from_source.Inverse() * coarse->reference_from_source;
  GR_EXPECT_LT(RadToDeg(RotationAngle(error.rotation)), 1.0);
  GR_EXPECT_LT(error.translation.norm(), 0.5);
  GR_EXPECT_GT(coarse->score, 0.5);
  GR_EXPECT_LT(coarse->ambiguity, 0.95);
  const IcpResult fine =
      AlignMultiScale(source, reference, coarse->reference_from_source,
                      {1.0, 0.5, 0.2, 0.1}, 0.1);
  const Pose fine_error = reference_from_source.Inverse() * fine.pose;
  GR_EXPECT_LT(fine_error.translation.norm(), 0.01);
  GR_EXPECT_LT(RadToDeg(RotationAngle(fine_error.rotation)), 0.02);
}

GR_TEST(CloudCompareTest, SurfaceDistancesFollowTheReferenceNormal) {
  const Points reference = GarageCloud(0.002, 3);
  Points query = GarageCloud(0.0, 4);
  for (Vec3& p : query) p.z() += 0.02;     // The map sits 2 cm too high.
  query.push_back(Vec3(10.0, 5.0, 10.0));  // Far from everything.
  const SurfaceDistances distances =
      MeasureSurfaceDistance(query, reference, 0.2, 0.3);
  int floor_points = 0;
  double floor_sum = 0.0;
  for (size_t i = 0; i + 1 < query.size(); ++i) {
    if (std::isnan(distances.signed_distance[i])) continue;
    if (distances.normals[i].z() > 0.99 && query[i].z() < 0.5) {
      ++floor_points;
      floor_sum += distances.signed_distance[i];
    }
    GR_EXPECT_LE(distances.nearest[i], 0.3);
  }
  GR_ASSERT_TRUE(floor_points > 1000);
  GR_EXPECT_NEAR(floor_sum / floor_points, 0.02, 0.002);
  GR_EXPECT_TRUE(std::isnan(distances.nearest.back()));
  GR_EXPECT_TRUE(std::isnan(distances.signed_distance.back()));
}

}  // namespace
}  // namespace gr
