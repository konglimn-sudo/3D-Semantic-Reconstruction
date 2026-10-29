#include "gr/io/trip.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "gr/io/npy.h"
#include "gr/io/tables.h"
#include "gr/nav/trip_localizer.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

namespace fs = std::filesystem;
using ::gr::testing::GarageStructure;
using ::gr::testing::LidarModel;
using ::gr::testing::LoopRoute;
using ::gr::testing::MakeScene;
using ::gr::testing::ScanFromPose;
using ::gr::testing::SimulateProprioception;
using ::gr::testing::SimulateSweep;
using ::gr::testing::Sweep;

fs::path TempDir(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / ("gr_io_test_" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void WriteText(const fs::path& path, const std::string& text) {
  std::ofstream(path) << text;
}

GR_TEST(NpyTest, NpzRoundTripsCompressedAndStored) {
  const fs::path dir = TempDir("npz");
  NpzArchive arrays;
  arrays["xyz"] = NpyArray::FromFloats({1, 2, 3, 4, 5, 6.5f}, {2, 3});
  arrays["scale"] = NpyArray::FromDoubles({0.005}, {});
  arrays["ids"] = NpyArray::FromInts({-7, 9}, {2});
  // float16 1.0, -2.5 and 2^-14, the smallest normal half (little-endian
  // bytes).
  NpyArray half;
  half.kind = 'f';
  half.item_size = sizeof(uint16_t);
  half.shape = {3};
  half.data = {0x00, 0x3C, 0x00, 0xC1, 0x00, 0x04};
  arrays["half"] = half;
  for (bool compress : {true, false}) {
    const std::string path = (dir / "a.npz").string();
    WriteNpz(path, arrays, compress);
    const NpzArchive back = ReadNpz(path);
    GR_ASSERT_TRUE(back.size() == arrays.size());
    GR_EXPECT_EQ(back.at("xyz").rows(), 2);
    GR_EXPECT_EQ(back.at("xyz").columns(), 3);
    GR_EXPECT_NEAR(back.at("xyz").ToDoubles()[5], 6.5, 0.0);
    GR_EXPECT_TRUE(back.at("scale").shape.empty());
    GR_EXPECT_NEAR(back.at("scale").ToDoubles()[0], 0.005, 0.0);
    GR_EXPECT_EQ(back.at("ids").ToInts()[0], int64_t{-7});
    constexpr int kSmallestNormalHalfExponent = -14;
    const std::vector<double> halfs = back.at("half").ToDoubles();
    GR_EXPECT_NEAR(halfs[0], 1.0, 0.0);
    GR_EXPECT_NEAR(halfs[1], -2.5, 0.0);
    GR_EXPECT_NEAR(halfs[2], std::ldexp(1.0, kSmallestNormalHalfExponent), 0.0);
  }
}

GR_TEST(TablesTest, TumRoundTripsAndQuaternionsFollowScipy) {
  // The file keeps 9 decimals of stamps and quaternions.
  constexpr double kStampTolerance = 1e-9;  // s
  constexpr double kRotationTolerance = 1e-8;
  // Positions in an (x, y, z, w) quaternion.
  constexpr int kQuaternionZ = 2;
  constexpr int kQuaternionW = 3;
  constexpr double kYawNearPi = -3.1;  // rad
  const fs::path dir = TempDir("tum");
  Trajectory trajectory;
  trajectory.stamps = {1.5, 2.5};
  trajectory.poses = {
      Pose(So3Exp(Vec3(0.1, -0.2, 3.0)), Vec3(1, 2, 3)),
      Pose(So3Exp(Vec3(0.0, 0.0, kYawNearPi)), Vec3(-1, 0, 0.5))};
  WriteTum((dir / "a.tum").string(), trajectory);
  const Trajectory back = ReadTum((dir / "a.tum").string());
  GR_ASSERT_TRUE(back.poses.size() == trajectory.poses.size());
  for (size_t k = 0; k < trajectory.poses.size(); ++k) {
    GR_EXPECT_NEAR(back.stamps[k], trajectory.stamps[k], kStampTolerance);
    GR_EXPECT_LT((back.poses[k].rotation - trajectory.poses[k].rotation).norm(),
                 kRotationTolerance);
  }
  // scipy picks the largest of trace and the diagonal: a yaw near pi keeps
  // the sign of z.
  const Eigen::Vector4d quaternion =
      QuaternionXyzw(So3Exp(Vec3(0, 0, kYawNearPi)));
  GR_EXPECT_GT(quaternion[kQuaternionZ], 0.0);
  GR_EXPECT_LT(quaternion[kQuaternionW], 0.0);
}

// A simulated trip written in the dataset layout, read back and localized
// in a map of the garage structure from its entrance pose.
GR_TEST(TripReaderTest, ReadsATripAndLocalizesIt) {
  // 100 sweeps at 10 Hz from t = 0.1 s, with 12 s of IMU and wheel data.
  constexpr int kSweeps = 100;
  constexpr double kSweepPeriod = 0.1;      // s
  constexpr double kSensorDuration = 12.0;  // s
  constexpr int kSeed = 3;
  // The synthetic generator quantises points to 5 mm (points_q * scale).
  constexpr double kQuantum = 0.005;  // m
  constexpr double kNanosecondsPerSecond = 1e9;
  constexpr double kStampTolerance = 1e-12;  // s
  constexpr double kPoseTolerance = 1e-12;   // m
  constexpr size_t kMinSweepPoints = 1000;
  // The map holds static scans of the structure every metre of the route.
  constexpr double kMapScanSpacing = 1.0;  // m
  constexpr double kLevelWindow = 1.0;     // s of IMU to level the start.
  // Every second sweep is localized; all must track.
  constexpr int kLocalizedSweeps = kSweeps / 2;
  constexpr double kMinRegisteredShare = 0.95;
  constexpr double kMaxPositionError = 0.05;  // m
  const fs::path dir = TempDir("trip");
  const LoopRoute route;
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  WriteText(dir / "rig.yaml",
            "lidars:\n  top:\n    T_vehicle_lidar:\n    - [1, 0, 0, 1.0]\n"
            "    - [0, 1, 0, 0]\n    - [0, 0, 1, 1.8]\n    - [0, 0, 0, 1]\n"
            "    sweep_duration: 0.1\nwheel_speed_scale: 1.0\n");
  const Pose entrance = route.PoseAt(kSweepPeriod);
  std::string meta = "trip_id: sim\ninitial_pose:\n";
  for (int row = 0; row < kHomogeneousDim; ++row) {
    meta += "- [";
    for (int column = 0; column < kHomogeneousDim; ++column) {
      meta +=
          (column ? ", " : "") + std::to_string(entrance.Matrix()(row, column));
    }
    meta += "]\n";
  }
  WriteText(dir / "meta.yaml", meta);
  const testing::Proprioception sensors =
      SimulateProprioception(route, 0.0, kSensorDuration);
  std::ofstream imu(dir / "imu.csv");
  imu << "t,gx,gy,gz,ax,ay,az\n";
  for (size_t k = 0; k < sensors.imu.size(); ++k) {
    imu << sensors.imu.stamps[k];
    for (const Vec3* vector : {&sensors.imu.gyro[k], &sensors.imu.accel[k]}) {
      for (int axis = 0; axis < kSpaceDim; ++axis) {
        imu << "," << (*vector)[axis];
      }
    }
    imu << "\n";
  }
  imu.close();
  std::ofstream wheel(dir / "wheel.csv");
  wheel << "t,speed\n";
  for (size_t k = 0; k < sensors.wheel.size(); ++k) {
    wheel << sensors.wheel.stamps[k] << "," << sensors.wheel.speed[k] << "\n";
  }
  wheel.close();
  fs::create_directories(dir / "lidar" / "top");
  std::mt19937 rng(kSeed);
  const BoxScene scene = MakeScene(GarageStructure());
  std::vector<Pose> truth;
  for (int k = 1; k <= kSweeps; ++k) {
    const double stamp = kSweepPeriod * k;
    const Sweep sweep = SimulateSweep(scene, route, vehicle_from_lidar, stamp,
                                      LidarModel(), rng);
    std::vector<int32_t> quantised;
    std::vector<float> time;
    for (size_t i = 0; i < sweep.points.size(); ++i) {
      for (int axis = 0; axis < kSpaceDim; ++axis) {
        quantised.push_back(static_cast<int32_t>(
            std::lround(sweep.points[i][axis] / kQuantum)));
      }
      time.push_back(static_cast<float>(sweep.times[i]));
    }
    const int64_t point_count = static_cast<int64_t>(time.size());
    NpzArchive npz;
    npz["points_q"] = NpyArray::FromInts(quantised, {point_count, kSpaceDim});
    npz["scale"] = NpyArray::FromDoubles({kQuantum}, {});
    npz["time"] = NpyArray::FromFloats(time, {point_count});
    const std::string name =
        std::to_string(std::llround(stamp * kNanosecondsPerSecond)) + ".npz";
    WriteNpz((dir / "lidar" / "top" / name).string(), npz);
    truth.push_back(route.PoseAt(stamp));
  }

  const TripReader trip(dir.string());
  GR_EXPECT_EQ(trip.trip_id(), std::string("sim"));
  GR_ASSERT_TRUE(trip.size() == kSweeps);
  GR_EXPECT_NEAR(trip.lidar_stamp(0), kSweepPeriod, kStampTolerance);
  GR_EXPECT_LT((trip.rig().primary_lidar().vehicle_from_lidar.translation -
                vehicle_from_lidar.translation)
                   .norm(),
               kPoseTolerance);
  const LidarSweep first = trip.LoadLidar(0);
  GR_EXPECT_TRUE(first.points.size() > kMinSweepPoints);
  GR_EXPECT_EQ(first.intensity.size(), first.points.size());
  MotionPrior prior = trip.BuildMotionPrior();
  GR_EXPECT_TRUE(prior.valid());
  const std::optional<Pose> start = trip.EntrancePose(prior);
  GR_ASSERT_TRUE(start.has_value());

  // Map: static scans of the structure every metre around the loop.
  Points map;
  for (double along = 0.0; along < route.length(); along += kMapScanSpacing) {
    const Pose sensor =
        route.PoseAt(along / route.speed()) * vehicle_from_lidar;
    const Points world = ScanFromPose(scene, sensor).points_world;
    map.insert(map.end(), world.begin(), world.end());
  }
  // The odometry frame starts levelled (here: level ground).
  const Pose levelled = LevelledStart(trip.ImuSamplesInVehicle(),
                                      trip.lidar_stamp(0), kLevelWindow);
  TripLocalizer localizer(map, TripLocalizerConfig(),
                          trip.rig().primary_lidar().vehicle_from_lidar,
                          levelled, std::move(prior), start);
  for (size_t k = 0; k < trip.size(); ++k) {
    const LidarSweep sweep = trip.LoadLidar(k);
    localizer.Process(sweep.points, sweep.intensity, sweep.time, sweep.stamp,
                      static_cast<int>(k), k + 1 == trip.size(), k % 2 == 0);
  }
  const TripLocalizationSummary summary = localizer.Summary();
  GR_EXPECT_EQ(summary.sweeps, kLocalizedSweeps);
  GR_EXPECT_EQ(summary.first_tracked, 0);
  GR_EXPECT_EQ(summary.tracked, kLocalizedSweeps);
  GR_EXPECT_GE(summary.registered_share_of_tracked, kMinRegisteredShare);
  for (const SweepLocalization& sweep : localizer.sweeps()) {
    GR_EXPECT_LT((sweep.map_from_vehicle.translation -
                  truth[static_cast<size_t>(sweep.index)].translation)
                     .norm(),
                 kMaxPositionError);
  }
  fs::remove_all(dir);
}

}  // namespace
}  // namespace gr
