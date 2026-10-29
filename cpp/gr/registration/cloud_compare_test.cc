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

// The test garage scanned every 3 m of a walk along both aisles, the
// scanner 1.6 m high, out to 25 m.
Points GarageCloud(double noise, int seed) {
  constexpr double kFirstX = -4.0;  // m
  constexpr double kLastX = 34.0;
  constexpr double kSpacing = 3.0;
  constexpr double kScannerHeight = 1.6;
  constexpr double kRange = 25.0;
  const BoxScene scene = testing::MakeScene(testing::GarageStructure());
  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, noise);
  Points cloud;
  for (double x = kFirstX; x <= kLastX; x += kSpacing) {
    for (double aisle_y : {5.0, 15.0}) {
      const Pose pose(Mat3::Identity(), Vec3(x, aisle_y, kScannerHeight));
      for (const Vec3& point :
           testing::ScanFromPose(scene, pose, kRange).points_world) {
        cloud.push_back(point + Vec3(gauss(rng), gauss(rng), gauss(rng)));
      }
    }
  }
  return cloud;
}

GR_TEST(CloudCompareTest, CoarseAndFineRegistrationRecoverTheFrame) {
  constexpr double kNoise = 0.005;  // m
  constexpr int kSourceSeed = 1;
  constexpr int kReferenceSeed = 2;
  constexpr double kYawDegrees = 37.0;
  // The coarse search finds the frame to a degree and half a metre, ICP to
  // a centimetre.
  constexpr double kMaxCoarseRotation = 1.0;     // deg
  constexpr double kMaxCoarseTranslation = 0.5;  // m
  constexpr double kMinScore = 0.5;
  constexpr double kMaxAmbiguity = 0.95;
  constexpr double kFineVoxel = 0.1;            // m
  constexpr double kMaxFineTranslation = 0.01;  // m
  constexpr double kMaxFineRotation = 0.02;     // deg
  const Points source = GarageCloud(kNoise, kSourceSeed);
  // The reference is the same garage in another frame (another session).
  const Pose reference_from_source(
      So3Exp(Vec3(0.0, 0.0, DegToRad(kYawDegrees))), Vec3(120.0, -48.0, 3.1));
  const Points reference = TransformPoints(GarageCloud(kNoise, kReferenceSeed),
                                           reference_from_source);
  const std::optional<CoarseAlignment> coarse =
      AlignGravityAligned(source, reference);
  GR_ASSERT_TRUE(coarse.has_value());
  const Pose error =
      reference_from_source.Inverse() * coarse->reference_from_source;
  GR_EXPECT_LT(RadToDeg(RotationAngle(error.rotation)), kMaxCoarseRotation);
  GR_EXPECT_LT(error.translation.norm(), kMaxCoarseTranslation);
  GR_EXPECT_GT(coarse->score, kMinScore);
  GR_EXPECT_LT(coarse->ambiguity, kMaxAmbiguity);
  const IcpResult fine =
      AlignMultiScale(source, reference, coarse->reference_from_source,
                      {1.0, 0.5, 0.2, 0.1}, kFineVoxel);
  const Pose fine_error = reference_from_source.Inverse() * fine.pose;
  GR_EXPECT_LT(fine_error.translation.norm(), kMaxFineTranslation);
  GR_EXPECT_LT(RadToDeg(RotationAngle(fine_error.rotation)), kMaxFineRotation);
}

GR_TEST(CloudCompareTest, SurfaceDistancesFollowTheReferenceNormal) {
  constexpr double kReferenceNoise = 0.002;  // m
  constexpr int kReferenceSeed = 3;
  constexpr int kQuerySeed = 4;
  constexpr double kHeightError = 0.02;  // m: the map sits 2 cm too high.
  constexpr double kRadius = 0.2;        // m
  constexpr double kMaxDistance = 0.3;   // m
  // Floor points: the normal points up and the point lies low.
  constexpr double kMinFloorNormalZ = 0.99;
  constexpr double kMaxFloorHeight = 0.5;  // m
  constexpr int kMinFloorPoints = 1000;
  constexpr double kMeanTolerance = 0.002;  // m
  const Points reference = GarageCloud(kReferenceNoise, kReferenceSeed);
  Points query = GarageCloud(0.0, kQuerySeed);
  for (Vec3& point : query) point.z() += kHeightError;
  query.push_back(Vec3(10.0, 5.0, 10.0));  // Far from everything.
  const SurfaceDistances distances =
      MeasureSurfaceDistance(query, reference, kRadius, kMaxDistance);
  int floor_points = 0;
  double floor_sum = 0.0;
  for (size_t i = 0; i + 1 < query.size(); ++i) {
    if (std::isnan(distances.signed_distance[i])) continue;
    if (distances.normals[i].z() > kMinFloorNormalZ &&
        query[i].z() < kMaxFloorHeight) {
      ++floor_points;
      floor_sum += distances.signed_distance[i];
    }
    GR_EXPECT_LE(distances.nearest[i], kMaxDistance);
  }
  GR_ASSERT_TRUE(floor_points > kMinFloorPoints);
  GR_EXPECT_NEAR(floor_sum / floor_points, kHeightError, kMeanTolerance);
  GR_EXPECT_TRUE(std::isnan(distances.nearest.back()));
  GR_EXPECT_TRUE(std::isnan(distances.signed_distance.back()));
}

}  // namespace
}  // namespace gr
