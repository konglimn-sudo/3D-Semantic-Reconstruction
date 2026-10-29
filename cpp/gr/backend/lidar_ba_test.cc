#include "gr/backend/lidar_ba.h"

#include <cmath>
#include <random>
#include <vector>

#include "gr/backend/backend.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::GarageStructure;
using ::gr::testing::MakeScene;
using ::gr::testing::ScanFromPose;

constexpr int kKeyframes = 14;
// Odometry errors per keyframe, and the anchor on the first keyframe.
constexpr int kNoiseSeed = 4;
constexpr double kOdometryErrorT = 0.02;  // m
constexpr double kOdometryErrorR = 0.3;   // deg
constexpr double kAnchorSigmaT = 0.001;   // m
constexpr double kAnchorSigmaR = 0.01;    // deg
constexpr int kRounds = 3;

// Keyframes every 1.5 m along y = 5 in the test garage (LiDAR at 1.8 m),
// with exact sweeps plus 5 mm range noise.
struct Drive {
  static constexpr int kSeed = 21;
  static constexpr double kSweepVoxel = 0.1;      // m
  static constexpr double kRangeNoise = 0.005;    // m
  static constexpr double kMaxRange = 30.0;       // m
  static constexpr double kFirstX = 2.0;          // m
  static constexpr double kSpacing = 1.5;         // m
  static constexpr double kRouteY = 5.0;          // m
  static constexpr double kLidarHeight = 1.8;     // m
  static constexpr double kKeyframePeriod = 0.2;  // s
  // The heading weaves by 0.05 rad, with 0.7 rad of phase per keyframe.
  static constexpr double kWeave = 0.05;
  static constexpr double kWeavePhase = 0.7;

  std::vector<Pose> truth;
  KeyframeSet keyframes{{kSweepVoxel}};

  explicit Drive(const std::vector<Pose>& odometry_noise = {}) {
    const BoxScene scene = MakeScene(GarageStructure());
    std::mt19937 rng(kSeed);
    std::normal_distribution<double> gauss(0.0, kRangeNoise);
    for (int index = 0; index < kKeyframes; ++index) {
      const Pose pose(
          So3Exp(Vec3(0.0, 0.0, kWeave * std::sin(kWeavePhase * index))),
          Vec3(kFirstX + kSpacing * index, kRouteY, kLidarHeight));
      truth.push_back(pose);
      Points sweep = ScanFromPose(scene, pose, kMaxRange).points_sensor;
      for (Vec3& point : sweep) point += gauss(rng) * point.normalized();
      Keyframe keyframe;
      keyframe.frame = index;
      keyframe.stamp = kKeyframePeriod * index;
      keyframe.odom = odometry_noise.empty()
                          ? pose
                          : pose * odometry_noise[static_cast<size_t>(index)];
      keyframes.Add(keyframe, sweep);
    }
  }
};

// Random odometry errors of up to kOdometryErrorT / kOdometryErrorR along
// every axis, none on the first keyframe.
std::vector<Pose> OdometryNoise() {
  std::mt19937 rng(kNoiseSeed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<Pose> noise;
  for (int index = 0; index < kKeyframes; ++index) {
    noise.emplace_back(
        So3Exp(DegToRad(kOdometryErrorR) *
               Vec3(uniform(rng), uniform(rng), uniform(rng))),
        kOdometryErrorT * Vec3(uniform(rng), uniform(rng), uniform(rng)));
  }
  noise[0] = Pose();
  return noise;
}

Anchor FirstKeyframeAnchor(const Drive& drive) {
  Anchor anchor;
  anchor.keyframe = 0;
  anchor.pose = drive.truth[0];
  anchor.sigma_t = kAnchorSigmaT;
  anchor.sigma_r = kAnchorSigmaR;
  return anchor;
}

// Largest position error (m) of `poses`.
double MaxPositionError(const std::vector<Pose>& poses, const Drive& drive) {
  double worst = 0.0;
  for (size_t keyframe = 0; keyframe < poses.size(); ++keyframe) {
    worst = std::max(
        worst, (poses[keyframe].translation - drive.truth[keyframe].translation)
                   .norm());
  }
  return worst;
}

GR_TEST(LidarBaTest, ExtractsTheSurfacesSharedByTheSweeps) {
  constexpr size_t kMinPlanes = 100;
  constexpr int kMinObservations = 2;
  constexpr double kWeightTolerance = 1e-9;
  constexpr double kThinRms = 0.01;  // m
  // |normal.z| of horizontal (floor, ceiling) and vertical (walls, columns)
  // planes.
  constexpr double kMinHorizontalZ = 0.999;
  constexpr double kMaxVerticalZ = 0.01;
  constexpr int kMinOrientedPlanes = 20;  // Of either orientation.
  // Percentages of the planes.
  constexpr int kPercent = 100;
  constexpr int kMinAxisAlignedPercent = 90;
  constexpr int kMinThinPercent = 98;
  const Drive drive;
  LidarBaConfig config;
  const std::vector<BaPlane> planes =
      ExtractPlanes(drive.keyframes, drive.truth, config);
  GR_ASSERT_TRUE(planes.size() > kMinPlanes);
  int horizontal = 0;
  int vertical = 0;
  int thin = 0;
  for (const BaPlane& plane : planes) {
    GR_EXPECT_GE(static_cast<int>(plane.observations.size()), kMinObservations);
    GR_EXPECT_LE(static_cast<int>(plane.observations.size()),
                 config.max_keyframes);
    for (size_t observation = 1; observation < plane.observations.size();
         ++observation) {
      GR_EXPECT_LT(plane.observations[observation - 1].node,
                   plane.observations[observation].node);
    }
    for (const PlaneObservation& observation : plane.observations) {
      GR_EXPECT_LE(observation.points.weight(),
                   config.max_cluster_weight + kWeightTolerance);
    }
    // Range noise only: thin planes, axis-aligned like the garage. A few
    // voxels at column / floor corners keep a little of the other surface.
    GR_EXPECT_LE(plane.rms, config.max_rms);
    if (plane.rms < kThinRms) ++thin;
    const double normal_z = std::abs(plane.normal.z());
    if (normal_z > kMinHorizontalZ) ++horizontal;
    if (normal_z < kMaxVerticalZ) ++vertical;
  }
  const int num_planes = static_cast<int>(planes.size());
  GR_EXPECT_GT(horizontal, kMinOrientedPlanes);  // Floor and ceiling.
  GR_EXPECT_GT(vertical, kMinOrientedPlanes);    // Walls and columns.
  GR_EXPECT_GT(horizontal + vertical,
               num_planes * kMinAxisAlignedPercent / kPercent);
  GR_EXPECT_GT(thin, num_planes * kMinThinPercent / kPercent);
}

GR_TEST(LidarBaTest, BundleAdjustmentRemovesOdometryErrors) {
  // Odometry with a random 2 cm / 0.3 deg error per keyframe; an anchor on
  // the first keyframe fixes the gauge.
  constexpr double kMinErrorBefore = 0.02;    // m
  constexpr double kMaxErrorAfter = 0.003;    // m
  constexpr double kMaxRotationAfter = 0.02;  // deg
  const Drive drive(OdometryNoise());
  const BackendConfig backend;
  LidarBaConfig config;
  config.rounds = kRounds;
  LidarBaSummary summary;
  const std::vector<Pose> start = drive.keyframes.OdomPoses();
  const std::vector<Pose> refined =
      RunLidarBa(drive.keyframes, start, {}, {FirstKeyframeAnchor(drive)}, {},
                 backend, config, nullptr, &summary);
  GR_ASSERT_TRUE(summary.rounds.size() == static_cast<size_t>(kRounds));
  GR_EXPECT_LT(summary.rounds.back().rms_median_after,
               summary.rounds.front().rms_median_before);
  double rotation_after = 0.0;
  for (size_t keyframe = 0; keyframe < start.size(); ++keyframe) {
    rotation_after =
        std::max(rotation_after,
                 RadToDeg(RotationAngle(refined[keyframe].rotation.transpose() *
                                        drive.truth[keyframe].rotation)));
  }
  GR_EXPECT_GT(MaxPositionError(start, drive), kMinErrorBefore);
  GR_EXPECT_LT(MaxPositionError(refined, drive), kMaxErrorAfter);
  GR_EXPECT_LT(rotation_after, kMaxRotationAfter);
}

GR_TEST(LidarBaTest, HierarchicalBaMatchesTheDenseOne) {
  // The same drive with windows of 4 keyframes (4 windows): close to the
  // dense BA.
  constexpr int kWindow = 4;
  constexpr double kMaxErrorAfter = 0.006;  // m
  const Drive drive(OdometryNoise());
  LidarBaConfig config;
  config.rounds = kRounds;
  config.window = kWindow;
  const std::vector<Pose> start = drive.keyframes.OdomPoses();
  const std::vector<Pose> refined = HierarchicalLidarBa(
      drive.keyframes, start, {}, {FirstKeyframeAnchor(drive)}, {},
      BackendConfig(), config);
  GR_EXPECT_LT(MaxPositionError(refined, drive), kMaxErrorAfter);
}

}  // namespace
}  // namespace gr
