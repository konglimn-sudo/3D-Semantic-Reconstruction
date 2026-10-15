// End-to-end section 4/5 chain in C++ only: two simulated drives through the
// test garage -> odometry with the IMU / wheel prior -> keyframes -> graph
// stage (loops, anchors, IMU, LiDAR bundle adjustment) -> map -> dynamic
// points -> occupancy and per-frame labels. A parked car exists only during
// drive 0.

#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"

#include "gr/backend.h"
#include "gr/exposure.h"
#include "gr/labeling.h"
#include "gr/occupancy.h"
#include "gr/odometry.h"
#include "gr/static_map.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::GarageStructure;
using ::gr::testing::LidarModel;
using ::gr::testing::LoopRoute;
using ::gr::testing::MakeScene;
using ::gr::testing::ParkedCars;
using ::gr::testing::Proprioception;
using ::gr::testing::SimBox;
using ::gr::testing::SimulateProprioception;
using ::gr::testing::SimulateSweep;
using ::gr::testing::Sweep;

const Pose& VehicleFromLidar() {
  static const Pose* const kPose =
      new Pose(Mat3::Identity(), Vec3(1.0, 0, 1.8));
  return *kPose;
}

struct Drive {
  LoopRoute route;
  std::vector<OdometryFrame> keyframes;  // With their sweeps.
  TripInputs inputs;
};

// IMU bias of each drive: [gyro (rad/s), accel (m/s^2)].
Vec6 DriveImuBias(int trip) {
  Vec6 bias;
  if (trip == 0) {
    bias << 0.002, -0.001, 0.003, 0.03, -0.02, 0.02;
  } else {
    bias << -0.001, 0.002, -0.002, -0.02, 0.04, -0.03;
  }
  return bias;
}

// Simulates one drive, runs the odometry front-end and adds its keyframes.
Drive RunDrive(int trip, double start, bool extra_car, double duration,
               KeyframeSet& keyframe_set, std::mt19937& rng) {
  Drive drive{LoopRoute(start), {}, {}};
  std::vector<SimBox> boxes = GarageStructure();
  for (const SimBox& car : ParkedCars(extra_car)) boxes.push_back(car);
  const BoxScene scene = MakeScene(boxes);
  Proprioception sensors = SimulateProprioception(
      drive.route, -0.5, duration + 0.5, DriveImuBias(trip));
  OdometryConfig config;
  config.voxel_size = 0.4;
  config.min_range = 1.5;
  OdometryPipeline pipeline(config, KeyframeConfig(), VehicleFromLidar(),
                            drive.route.PoseAt(0.0),
                            MotionPrior(sensors.imu, sensors.wheel));
  const int num_frames = static_cast<int>(duration * 10);
  for (int k = 0; k <= num_frames; ++k) {
    const double stamp = 0.1 * k;
    const Sweep sweep = SimulateSweep(scene, drive.route, VehicleFromLidar(),
                                      stamp, LidarModel(), rng);
    OdometryFrame frame = pipeline.Process(
        sweep.points, std::vector<float>(sweep.points.size(), 0.3f),
        sweep.times, stamp, k == num_frames);
    if (!frame.is_keyframe) continue;
    Keyframe keyframe;
    keyframe.trip = trip;
    keyframe.frame = k;
    keyframe.stamp = stamp;
    keyframe.odom = frame.pose;
    keyframe_set.Add(keyframe, frame.points);
    drive.keyframes.push_back(std::move(frame));
  }
  drive.inputs.vehicle_from_lidar = VehicleFromLidar();
  drive.inputs.initial_pose = drive.route.PoseAt(0.0);  // INS at the entrance.
  drive.inputs.frame_stamps = pipeline.stamps();
  drive.inputs.frame_odometry = pipeline.poses();
  drive.inputs.imu = std::move(sensors.imu);
  drive.inputs.wheel = std::move(sensors.wheel);
  return drive;
}

GR_TEST(EndToEndTest, TwoDrivesReconstructAndLabelASimulatedGarage) {
  // Longer than one lap (83 m at 4 m/s): loops within and across drives.
  const double duration = 25.0;
  std::mt19937 rng(3);
  const GraphStageConfig config;
  KeyframeSet keyframes(config.RequiredVoxels());
  std::vector<Drive> drives;
  drives.push_back(RunDrive(0, 0.0, true, duration, keyframes, rng));
  // Drive 1 starts on the other aisle; the extra car has left.
  drives.push_back(RunDrive(1, LoopRoute::kStraight + kPi * LoopRoute::kRadius,
                            false, duration, keyframes, rng));

  // ---------------------------------------------------------- graph stage
  std::vector<std::string> log;
  const GraphStageResult graph =
      RunGraphStage(keyframes, {drives[0].inputs, drives[1].inputs}, config,
                    [&](const std::string& line) { log.push_back(line); });
  GR_ASSERT_TRUE(graph.poses.size() == keyframes.size());
  // The bundle adjustment tightens the planes shared by the sweeps.
  GR_ASSERT_TRUE(graph.ba.rounds.size() ==
                 static_cast<size_t>(config.ba.rounds));
  GR_EXPECT_GT(graph.ba.rounds.front().planes, 500);
  GR_EXPECT_LT(graph.ba.rounds.back().rms_median_after,
               graph.ba.rounds.front().rms_median_before);
  GR_EXPECT_GE(graph.loops.size(), 10u);
  // IMU factors between consecutive keyframes estimate each drive's bias.
  GR_EXPECT_GT(graph.summary.imu_factors, 150);
  GR_ASSERT_TRUE(graph.inertial.valid.size() == keyframes.size());
  double gyro_bias_error = 0.0;
  double accel_bias_error = 0.0;
  for (size_t k = 0; k < keyframes.size(); ++k) {
    GR_EXPECT_TRUE(graph.inertial.valid[k] != 0);
    const Vec6 error =
        graph.inertial.biases[k] - DriveImuBias(keyframes[k].trip);
    gyro_bias_error = std::max(gyro_bias_error, error.head<3>().norm());
    accel_bias_error = std::max(accel_bias_error, error.tail<3>().norm());
  }
  std::printf(
      "       e2e: IMU bias error max: gyro %.4f deg/s, accel %.4f m/s^2\n",
      RadToDeg(gyro_bias_error), accel_bias_error);
  GR_EXPECT_LT(gyro_bias_error, 3e-4);
  GR_EXPECT_LT(accel_bias_error, 0.02);
  int across_drives = 0;
  for (const LoopConstraint& loop : graph.loops) {
    across_drives += keyframes[static_cast<size_t>(loop.i)].trip !=
                     keyframes[static_cast<size_t>(loop.j)].trip;
  }
  GR_EXPECT_GT(across_drives, 0);
  double error_max = 0.0;
  double error_sum = 0.0;
  double odometry_error_max = 0.0;
  double rotation_error_max = 0.0;
  for (size_t k = 0; k < keyframes.size(); ++k) {
    const Keyframe& keyframe = keyframes[k];
    const Pose truth = drives[static_cast<size_t>(keyframe.trip)].route.PoseAt(
                           keyframe.stamp) *
                       VehicleFromLidar();
    const double error =
        (graph.poses[k].translation - truth.translation).norm();
    error_max = std::max(error_max, error);
    error_sum += error;
    odometry_error_max =
        std::max(odometry_error_max,
                 (keyframe.odom.translation - truth.translation).norm());
    rotation_error_max = std::max(
        rotation_error_max, RadToDeg(RotationAngle(truth.rotation.transpose() *
                                                   graph.poses[k].rotation)));
  }
  const double error_mean = error_sum / static_cast<double>(keyframes.size());
  // Error after the best rigid alignment: the map's internal accuracy, as a
  // survey check against control points would see it.
  Eigen::Matrix3Xd estimated(3, keyframes.size());
  Eigen::Matrix3Xd truths(3, keyframes.size());
  for (size_t k = 0; k < keyframes.size(); ++k) {
    const Keyframe& keyframe = keyframes[k];
    estimated.col(static_cast<Eigen::Index>(k)) = graph.poses[k].translation;
    truths.col(static_cast<Eigen::Index>(k)) =
        (drives[static_cast<size_t>(keyframe.trip)].route.PoseAt(
             keyframe.stamp) *
         VehicleFromLidar())
            .translation;
  }
  const Mat4 align = Eigen::umeyama(estimated, truths, false);
  const Eigen::Matrix3Xd aligned =
      (align.block<3, 3>(0, 0) * estimated).colwise() +
      Vec3(align.block<3, 1>(0, 3));
  const double aligned_mean = (aligned - truths).colwise().norm().mean();
  std::printf(
      "       e2e: %zu keyframes, %zu loops (%d across drives), keyframe "
      "error mean %.2f cm (%.2f cm rigidly aligned), max %.2f cm (odometry "
      "only %.2f cm), rotation max %.3f deg\n",
      keyframes.size(), graph.loops.size(), across_drives, error_mean * 100.0,
      aligned_mean * 100.0, error_max * 100.0, odometry_error_max * 100.0,
      rotation_error_max);
  GR_EXPECT_LT(error_mean, 0.03);
  GR_EXPECT_LT(aligned_mean, 0.01);
  GR_EXPECT_LT(error_max, 0.08);
  GR_EXPECT_LT(rotation_error_max, 0.5);

  // ------------------------------------------------------------------ map
  // Fused map: a first pass for the nearest range per voxel, then the
  // weighted average of the observations within the range window.
  MapAccumulator accumulator(0.05, 20.0);
  for (int pass = 0; pass < 2; ++pass) {
    size_t k = 0;
    for (const Drive& drive : drives) {
      for (const OdometryFrame& frame : drive.keyframes) {
        if (pass == 0) {
          accumulator.AddRanges(graph.poses[k], frame.points);
        } else {
          accumulator.Add(static_cast<int>(k), graph.poses[k], frame.points,
                          frame.intensity);
        }
        ++k;
      }
    }
  }
  const MapAccumulator::Cloud map = accumulator.Collect();
  std::vector<Vec3> origins;
  for (const Pose& pose : graph.poses) origins.push_back(pose.translation);
  const NormalsResult normals =
      EstimateOrientedNormals(map.points, map.keyframe, origins, 0.2);
  std::vector<int> trip;
  std::vector<double> height;
  for (size_t i = 0; i < map.points.size(); ++i) {
    trip.push_back(keyframes[static_cast<size_t>(map.keyframe[i])].trip);
    height.push_back(map.points[i].z());  // Flat floor at z = 0.
  }
  DynamicDetector detector(map.points, normals.normals, trip, DynamicConfig());
  detector.SetElevationResolution(2.0);
  detector.SetHeights(height);
  size_t k = 0;
  for (size_t d = 0; d < drives.size(); ++d) {
    for (const OdometryFrame& frame : drives[d].keyframes) {
      detector.AddSweep(static_cast<int>(d), graph.poses[k++], frame.points);
    }
    std::vector<double> kf_stamps;
    std::vector<Pose> kf_odom;
    std::vector<Pose> kf_poses;
    for (int i : keyframes.TripIndices(static_cast<int>(d))) {
      kf_stamps.push_back(keyframes[static_cast<size_t>(i)].stamp);
      kf_odom.push_back(keyframes[static_cast<size_t>(i)].odom);
      kf_poses.push_back(graph.poses[static_cast<size_t>(i)]);
    }
    const std::vector<Pose> frames = CorrectFramePoses(
        kf_stamps, kf_odom, kf_poses, drives[d].inputs.frame_stamps,
        drives[d].inputs.frame_odometry);
    Points path;  // Vehicle origin on the floor.
    for (const Pose& pose : frames) {
      path.push_back((pose * VehicleFromLidar().Inverse()).translation);
    }
    detector.AddEgoPath(static_cast<int>(d), path);
  }
  const std::vector<DynamicKind> kind = detector.Classify();
  ObjectBox extra_car;
  extra_car.center = Vec3(12.0, 8.2, 0.75);
  extra_car.size = Vec3(4.6, 1.9, 1.5);
  size_t car_points = 0;
  size_t car_changed = 0;
  size_t structure_points = 0;
  size_t structure_dynamic = 0;
  for (size_t i = 0; i < map.points.size(); ++i) {
    const Vec3& p = map.points[i];
    if (p.z() < 0.2) continue;
    if (extra_car.Contains(p, 0.1)) {
      ++car_points;
      car_changed += kind[i] == DynamicKind::kChanged;
    } else if (p.z() < 2.8) {
      ++structure_points;
      structure_dynamic += kind[i] != DynamicKind::kStatic;
    }
  }
  std::printf(
      "       e2e: map %zu points; car of drive 0 only: %zu/%zu changed "
      "between drives; other points above the floor dynamic: %zu/%zu\n",
      map.points.size(), car_changed, car_points, structure_dynamic,
      structure_points);
  GR_EXPECT_GT(car_points, 200u);
  GR_EXPECT_GT(car_changed, 0.6 * car_points);
  GR_EXPECT_LT(structure_dynamic, 0.02 * structure_points);

  // ------------------------------------------- occupancy and frame labels
  Points static_points;
  std::vector<int> static_labels;
  for (size_t i = 0; i < map.points.size(); ++i) {
    if (kind[i] != DynamicKind::kStatic) continue;
    const double z = map.points[i].z();
    static_points.push_back(map.points[i]);
    // Floor / ceiling / wall or column.
    static_labels.push_back(z < 0.05 ? 1 : (z > 2.9 ? 3 : 4));
  }
  const auto camera = MakeCamera("pinhole", 320, 240, {160, 160, 160, 120});
  GR_ASSERT_TRUE(camera != nullptr);
  const Mat3 forward = (Mat3() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished();
  const Pose vehicle_from_camera(
      Eigen::AngleAxisd(DegToRad(15.0), Vec3::UnitY()).toRotationMatrix() *
          forward,
      Vec3(2.0, 0.0, 1.2));
  const OccupancyGenerator occupancy(static_points, static_labels,
                                     OccupancyConfig(), VehicleFromLidar(),
                                     {{camera, vehicle_from_camera}});
  std::vector<SlotGeometry> slots(1);
  // Entrance edge on the aisle.
  slots[0].corners = {Vec2(10.0, 3.5), Vec2(12.5, 3.5), Vec2(12.5, 0.5),
                      Vec2(10.0, 0.5)};
  const FrameLabeler labeler(static_points, static_labels, slots, {},
                             FrameLabelConfig());
  // A keyframe of drive 1 on the first aisle, a few metres before the slot.
  int best = -1;
  for (int i : keyframes.TripIndices(1)) {
    const Vec3& t = graph.poses[static_cast<size_t>(i)].translation;
    if (std::abs(t.y() - 5.0) < 0.3 && t.x() > 3.0 && t.x() < 5.0) best = i;
  }
  GR_ASSERT_TRUE(best >= 0);
  const Pose world_from_vehicle =
      graph.poses[static_cast<size_t>(best)] * VehicleFromLidar().Inverse();
  const size_t index_in_drive =
      static_cast<size_t>(best - keyframes.TripIndices(1).front());
  const OccupancyResult grid = occupancy.Generate(
      world_from_vehicle, {}, drives[1].keyframes[index_in_drive].points);
  size_t floor_voxels = 0;
  size_t wall_voxels = 0;
  size_t lidar_voxels = 0;
  size_t camera_voxels = 0;
  for (size_t i = 0; i < grid.semantics.size(); ++i) {
    floor_voxels += grid.semantics[i] == 1;
    wall_voxels += grid.semantics[i] == 4;
    lidar_voxels += grid.mask_lidar[i];
    camera_voxels += grid.mask_camera[i];
  }
  GR_EXPECT_GT(floor_voxels, 5000u);
  GR_EXPECT_GT(wall_voxels, 100u);
  GR_EXPECT_GT(lidar_voxels, 10000u);
  GR_EXPECT_GT(camera_voxels, 1000u);
  const CameraExposure exposure(world_from_vehicle * vehicle_from_camera);
  const FrameLabels frame = labeler.Label(*camera, exposure, {});
  GR_ASSERT_TRUE(frame.slots.size() == 1);
  GR_EXPECT_GT(frame.slots[0].visible_fraction, 0.3);
  size_t floor_pixels = 0;
  for (uint8_t label : frame.label_image) floor_pixels += label == 1;
  GR_EXPECT_GT(floor_pixels, 5000u);
}

}  // namespace
}  // namespace gr
