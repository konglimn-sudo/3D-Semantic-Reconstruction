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
  // float16 1.0, -2.5, 6.1e-5 (subnormal 2^-14 * 1.0 / 2^-24 * 1024).
  NpyArray half;
  half.kind = 'f';
  half.item_size = 2;
  half.shape = {3};
  half.data = {0x00, 0x3C, 0x00, 0xC1, 0x00, 0x04};
  arrays["half"] = half;
  for (bool compress : {true, false}) {
    const std::string path = (dir / "a.npz").string();
    WriteNpz(path, arrays, compress);
    const NpzArchive back = ReadNpz(path);
    GR_ASSERT_TRUE(back.size() == 4);
    GR_EXPECT_EQ(back.at("xyz").rows(), 2);
    GR_EXPECT_EQ(back.at("xyz").columns(), 3);
    GR_EXPECT_NEAR(back.at("xyz").ToDoubles()[5], 6.5, 0.0);
    GR_EXPECT_TRUE(back.at("scale").shape.empty());
    GR_EXPECT_NEAR(back.at("scale").ToDoubles()[0], 0.005, 0.0);
    GR_EXPECT_EQ(back.at("ids").ToInts()[0], int64_t{-7});
    const std::vector<double> h = back.at("half").ToDoubles();
    GR_EXPECT_NEAR(h[0], 1.0, 0.0);
    GR_EXPECT_NEAR(h[1], -2.5, 0.0);
    GR_EXPECT_NEAR(h[2], std::ldexp(1.0, -14), 0.0);
  }
}

GR_TEST(TablesTest, TumRoundTripsAndQuaternionsFollowScipy) {
  const fs::path dir = TempDir("tum");
  Trajectory trajectory;
  trajectory.stamps = {1.5, 2.5};
  trajectory.poses = {
      Pose(So3Exp(Vec3(0.1, -0.2, 3.0)), Vec3(1, 2, 3)),
      Pose(So3Exp(Vec3(0.0, 0.0, -3.1)), Vec3(-1, 0, 0.5))};
  WriteTum((dir / "a.tum").string(), trajectory);
  const Trajectory back = ReadTum((dir / "a.tum").string());
  GR_ASSERT_TRUE(back.poses.size() == 2);
  for (size_t k = 0; k < 2; ++k) {
    GR_EXPECT_NEAR(back.stamps[k], trajectory.stamps[k], 1e-9);
    GR_EXPECT_LT((back.poses[k].rotation - trajectory.poses[k].rotation)
                     .norm(),
                 1e-8);
  }
  // scipy picks the largest of trace and the diagonal: a yaw near pi keeps
  // the sign of z.
  const Eigen::Vector4d q = QuaternionXyzw(So3Exp(Vec3(0, 0, -3.1)));
  GR_EXPECT_GT(q[2], 0.0);
  GR_EXPECT_LT(q[3], 0.0);
}

// A simulated trip written in the dataset layout, read back and localized
// in a map of the garage structure from its entrance pose.
GR_TEST(TripReaderTest, ReadsATripAndLocalizesIt) {
  const fs::path dir = TempDir("trip");
  const LoopRoute route;
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  WriteText(dir / "rig.yaml",
            "lidars:\n  top:\n    T_vehicle_lidar:\n    - [1, 0, 0, 1.0]\n"
            "    - [0, 1, 0, 0]\n    - [0, 0, 1, 1.8]\n    - [0, 0, 0, 1]\n"
            "    sweep_duration: 0.1\nwheel_speed_scale: 1.0\n");
  const Pose entrance = route.PoseAt(0.1);
  std::string meta = "trip_id: sim\ninitial_pose:\n";
  for (int r = 0; r < 4; ++r) {
    meta += "- [";
    for (int c = 0; c < 4; ++c) {
      meta += (c ? ", " : "") + std::to_string(entrance.Matrix()(r, c));
    }
    meta += "]\n";
  }
  WriteText(dir / "meta.yaml", meta);
  const testing::Proprioception sensors =
      SimulateProprioception(route, 0.0, 12.0);
  std::ofstream imu(dir / "imu.csv");
  imu << "t,gx,gy,gz,ax,ay,az\n";
  for (size_t k = 0; k < sensors.imu.size(); ++k) {
    imu << sensors.imu.stamps[k];
    for (const Vec3* v : {&sensors.imu.gyro[k], &sensors.imu.accel[k]}) {
      for (int c = 0; c < 3; ++c) imu << "," << (*v)[c];
    }
    imu << "\n";
  }
  imu.close();
  std::ofstream wheel(dir / "wheel.csv");
  wheel << "t,speed\n";
  for (size_t k = 0; k < sensors.wheel.size(); ++k) {
    wheel << sensors.wheel.stamps[k] << "," << sensors.wheel.speed[k]
          << "\n";
  }
  wheel.close();
  fs::create_directories(dir / "lidar" / "top");
  std::mt19937 rng(3);
  const BoxScene scene = MakeScene(GarageStructure());
  std::vector<Pose> truth;
  for (int k = 1; k <= 100; ++k) {
    const double t = 0.1 * k;
    const Sweep sweep =
        SimulateSweep(scene, route, vehicle_from_lidar, t, LidarModel(), rng);
    std::vector<int32_t> q;  // Quantised as the synthetic generator does.
    std::vector<float> time;
    for (size_t i = 0; i < sweep.points.size(); ++i) {
      for (int c = 0; c < 3; ++c) {
        q.push_back(static_cast<int32_t>(
            std::lround(sweep.points[i][c] / 0.005)));
      }
      time.push_back(static_cast<float>(sweep.times[i]));
    }
    const int64_t n = static_cast<int64_t>(time.size());
    NpzArchive npz;
    npz["points_q"] = NpyArray::FromInts(q, {n, 3});
    npz["scale"] = NpyArray::FromDoubles({0.005}, {});
    npz["time"] = NpyArray::FromFloats(time, {n});
    const std::string name = std::to_string(std::llround(t * 1e9)) + ".npz";
    WriteNpz((dir / "lidar" / "top" / name).string(), npz);
    truth.push_back(route.PoseAt(t));
  }

  const TripReader trip(dir.string());
  GR_EXPECT_EQ(trip.trip_id(), std::string("sim"));
  GR_ASSERT_TRUE(trip.size() == 100);
  GR_EXPECT_NEAR(trip.lidar_stamp(0), 0.1, 1e-12);
  GR_EXPECT_LT((trip.rig().primary_lidar().vehicle_from_lidar.translation -
                vehicle_from_lidar.translation)
                   .norm(),
               1e-12);
  const LidarSweep first = trip.LoadLidar(0);
  GR_EXPECT_TRUE(first.points.size() > 1000);
  GR_EXPECT_EQ(first.intensity.size(), first.points.size());
  MotionPrior prior = trip.BuildMotionPrior();
  GR_EXPECT_TRUE(prior.valid());
  const std::optional<Pose> start = trip.EntrancePose(prior);
  GR_ASSERT_TRUE(start.has_value());

  // Map: static scans of the structure every metre around the loop.
  Points map;
  for (double s = 0.0; s < route.length(); s += 1.0) {
    const Pose sensor = route.PoseAt(s / route.speed()) * vehicle_from_lidar;
    const Points world = ScanFromPose(scene, sensor).points_world;
    map.insert(map.end(), world.begin(), world.end());
  }
  // The odometry frame starts levelled (here: level ground).
  const Pose levelled = LevelledStart(trip.ImuSamplesInVehicle(),
                                      trip.lidar_stamp(0), 1.0);
  TripLocalizer localizer(map, TripLocalizerConfig(),
                          trip.rig().primary_lidar().vehicle_from_lidar,
                          levelled, std::move(prior), start);
  for (size_t k = 0; k < trip.size(); ++k) {
    const LidarSweep sweep = trip.LoadLidar(k);
    localizer.Process(sweep.points, sweep.intensity, sweep.time, sweep.stamp,
                      static_cast<int>(k), k + 1 == trip.size(), k % 2 == 0);
  }
  const TripLocalizationSummary summary = localizer.Summary();
  GR_EXPECT_EQ(summary.sweeps, 50);
  GR_EXPECT_EQ(summary.first_tracked, 0);
  GR_EXPECT_EQ(summary.tracked, 50);
  GR_EXPECT_GE(summary.registered_share_of_tracked, 0.95);
  for (const SweepLocalization& s : localizer.sweeps()) {
    GR_EXPECT_LT((s.map_from_vehicle.translation -
                  truth[static_cast<size_t>(s.index)].translation)
                     .norm(),
                 0.05);
  }
  fs::remove_all(dir);
}

}  // namespace
}  // namespace gr
