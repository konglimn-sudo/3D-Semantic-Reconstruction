#include "gr/lidar_ba.h"

#include <cmath>
#include <random>
#include <vector>

#include "gr/backend.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

// Keyframes every 1.5 m along y = 5 in the test garage (LiDAR at 1.8 m),
// with exact sweeps plus 5 mm range noise.
struct Drive {
  std::vector<Pose> truth;
  KeyframeSet keyframes{{0.1}};

  explicit Drive(const std::vector<Pose>& odometry_noise = {}) {
    const BoxScene scene = testing::MakeScene(testing::GarageStructure());
    std::mt19937 rng(21);
    std::normal_distribution<double> gauss(0.0, 0.005);
    for (int k = 0; k < 14; ++k) {
      const Pose pose(So3Exp(Vec3(0.0, 0.0, 0.05 * std::sin(0.7 * k))),
                      Vec3(2.0 + 1.5 * k, 5.0, 1.8));
      truth.push_back(pose);
      Points sweep = testing::ScanFromPose(scene, pose, 30.0).points_sensor;
      for (Vec3& p : sweep) p += gauss(rng) * p.normalized();
      Keyframe keyframe;
      keyframe.frame = k;
      keyframe.stamp = 0.2 * k;
      keyframe.odom = odometry_noise.empty()
                          ? pose
                          : pose * odometry_noise[static_cast<size_t>(k)];
      keyframes.Add(keyframe, sweep);
    }
  }
};

GR_TEST(LidarBaTest, ExtractsTheSurfacesSharedByTheSweeps) {
  const Drive drive;
  LidarBaConfig config;
  const std::vector<BaPlane> planes =
      ExtractPlanes(drive.keyframes, drive.truth, config);
  GR_ASSERT_TRUE(planes.size() > 100);
  int horizontal = 0;
  int vertical = 0;
  int thin = 0;
  for (const BaPlane& plane : planes) {
    GR_EXPECT_GE(static_cast<int>(plane.observations.size()), 2);
    GR_EXPECT_LE(static_cast<int>(plane.observations.size()),
                 config.max_keyframes);
    for (size_t a = 1; a < plane.observations.size(); ++a) {
      GR_EXPECT_LT(plane.observations[a - 1].node, plane.observations[a].node);
    }
    for (const PlaneObservation& observation : plane.observations) {
      GR_EXPECT_LE(observation.points.weight(),
                   config.max_cluster_weight + 1e-9);
    }
    // Range noise only: thin planes, axis-aligned like the garage. A few
    // voxels at column / floor corners keep a little of the other surface.
    GR_EXPECT_LE(plane.rms, config.max_rms);
    if (plane.rms < 0.01) ++thin;
    const double z = std::abs(plane.normal.z());
    if (z > 0.999) ++horizontal;
    if (z < 0.01) ++vertical;
  }
  GR_EXPECT_GT(horizontal, 20);  // Floor and ceiling.
  GR_EXPECT_GT(vertical, 20);    // Walls and columns.
  GR_EXPECT_GT(horizontal + vertical, static_cast<int>(planes.size()) * 9 / 10);
  GR_EXPECT_GT(thin, static_cast<int>(planes.size()) * 98 / 100);
}

GR_TEST(LidarBaTest, BundleAdjustmentRemovesOdometryErrors) {
  // Odometry with a random 2 cm / 0.3 deg error per keyframe; an anchor on
  // the first keyframe fixes the gauge.
  std::mt19937 rng(4);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<Pose> noise;
  for (int k = 0; k < 14; ++k) {
    noise.emplace_back(
        So3Exp(DegToRad(0.3) * Vec3(uniform(rng), uniform(rng), uniform(rng))),
        0.02 * Vec3(uniform(rng), uniform(rng), uniform(rng)));
  }
  noise[0] = Pose();
  const Drive drive(noise);
  Anchor anchor;
  anchor.keyframe = 0;
  anchor.pose = drive.truth[0];
  anchor.sigma_t = 0.001;
  anchor.sigma_r = 0.01;
  const BackendConfig backend;
  LidarBaConfig config;
  config.rounds = 3;
  LidarBaSummary summary;
  const std::vector<Pose> start = drive.keyframes.OdomPoses();
  const std::vector<Pose> refined =
      RunLidarBa(drive.keyframes, start, {}, {anchor}, {}, backend, config,
                 nullptr, &summary);
  GR_ASSERT_TRUE(summary.rounds.size() == 3u);
  GR_EXPECT_LT(summary.rounds.back().rms_median_after,
               summary.rounds.front().rms_median_before);
  double before = 0.0;
  double after = 0.0;
  double rotation_after = 0.0;
  for (size_t k = 0; k < start.size(); ++k) {
    before = std::max(
        before, (start[k].translation - drive.truth[k].translation).norm());
    after = std::max(
        after, (refined[k].translation - drive.truth[k].translation).norm());
    rotation_after = std::max(
        rotation_after, RadToDeg(RotationAngle(refined[k].rotation.transpose() *
                                               drive.truth[k].rotation)));
  }
  GR_EXPECT_GT(before, 0.02);
  GR_EXPECT_LT(after, 0.003);
  GR_EXPECT_LT(rotation_after, 0.02);
}

}  // namespace
}  // namespace gr
