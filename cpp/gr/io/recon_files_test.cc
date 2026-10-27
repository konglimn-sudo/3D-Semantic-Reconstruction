#include "gr/io/recon_files.h"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "gr/io/digest.h"
#include "gr/io/npy.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

namespace fs = std::filesystem;

fs::path TempDir(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / ("gr_recon_files_" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

std::string Sha(const std::string& text) {
  Sha256 sha;
  sha.Update(text);
  return sha.HexDigest();
}

GR_TEST(DigestTest, Sha256MatchesTheStandardVectors) {
  GR_EXPECT_EQ(Sha(""), std::string("e3b0c44298fc1c149afbf4c8996fb924"
                                    "27ae41e4649b934ca495991b7852b855"));
  GR_EXPECT_EQ(Sha("abc"), std::string("ba7816bf8f01cfea414140de5dae2223"
                                       "b00361a396177a9cb410ff61f20015ad"));
  GR_EXPECT_EQ(Sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
               std::string("248d6a61d20638b8e5c026930c3e6039"
                           "a33ce45964ff2167f6ecedd419db06c1"));
}

GR_TEST(DigestTest, JsonDumpsSortsKeysAsPython) {
  Value inner = Value::Map();
  inner.Set("d", Value::Number(1e-5));
  inner.Set("c", Value::String("x"));
  Value list = Value::List();
  list.Append(Value::Number(1.0));
  list.Append(Value::Integer(2));
  Value value = Value::Map();
  value.Set("b", list);
  value.Set("a", inner);
  GR_EXPECT_EQ(PythonJsonDumps(value),
               std::string(R"({"a": {"c": "x", "d": 1e-05}, "b": [1.0, 2]})"));
}

GR_TEST(DigestTest, PoseSetIdsMatchPython) {
  // Golden ids from pose_set._digest for the same parts.
  Value rig_yaml = ParseYaml(R"(lidars:
  top:
    T_vehicle_lidar: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
cameras:
  cam:
    model: pinhole
    width: 640
    height: 480
    params: [500, 500, 320, 240]
    T_vehicle_cam: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)");
  TripPoses trip;
  trip.trip_id = "t0";
  trip.knot_stamps = {0.0, 1.0};
  trip.knot_poses = {Pose(), Pose(Mat3::Identity(), Vec3(1, 2, 3))};
  trip.bias_stamps = {0.5};
  Vec6 bias;
  bias << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  trip.biases = {bias};
  trip.rig = RigDict(rig_yaml);
  Eigen::MatrixXd imu(1, 7);
  imu << 0, 1, 2, 3, 4, 5, 6;
  const PoseSetIds ids =
      ComputePoseSetIds({trip}, {imu}, {Eigen::MatrixXd(0, 0)});
  GR_EXPECT_EQ(ids.trajectory_id, std::string("979dba2b352f6892"));
  GR_EXPECT_EQ(ids.pose_set_id, std::string("32bcf890ad0cca1f"));
}

GR_TEST(ValueTest, YamlRoundTripsThroughTheParser) {
  const Value value = ParseYaml(R"(a: 1
b: 1.0e-05
c: "x: y # z"
d: [1.5, -2, "s"]
e:
  - {k: 1, l: [1, 2]}
  - []
  - {}
f: {}
g: null
h: true
)");
  const std::string yaml = ToYaml(value);
  GR_EXPECT_TRUE(yaml.find("1.0e-05") != std::string::npos);
  GR_EXPECT_EQ(ToJson(ParseYaml(yaml)), ToJson(value));
}

GR_TEST(NpyTest, HalfsRoundToNearestEvenAsNumpy) {
  const std::vector<float> values = {
      1.0f,  0.1f,          65504.0f,
      65520.0f,  // Halfway to 65536: rounds up to Inf.
      1e-8f, 5.9604645e-08f,  // 2^-24, the smallest subnormal.
      3.0517578125e-05f,      // 2^-15, subnormal.
      -2.5f, 1.0f + 0x1p-11f,  // Halfway: to the even 1.0.
      1.0f + 3 * 0x1p-11f};    // Halfway: to the even 1 + 2^-9.
  const std::vector<uint16_t> expected = {0x3C00, 0x2E66, 0x7BFF, 0x7C00,
                                          0x0000, 0x0001, 0x0200, 0xC100,
                                          0x3C00, 0x3C02};
  const NpyArray array = NpyArray::FromHalfs(
      values, {static_cast<int64_t>(values.size())});
  GR_ASSERT_TRUE(array.kind == 'f' && array.item_size == 2);
  for (size_t k = 0; k < expected.size(); ++k) {
    uint16_t bits;
    std::memcpy(&bits, array.data.data() + 2 * k, 2);
    GR_EXPECT_EQ(bits, expected[k]);
  }
  GR_EXPECT_NEAR(array.ToFloats()[1], 0.0999755859375, 0.0);
}

GR_TEST(ReconFilesTest, WorkDirectoryFilesRoundTrip) {
  const fs::path dir = TempDir("work");
  TripOdometryFile odometry;
  odometry.stamps = {0.1, 0.2, 0.3};
  for (int k = 0; k < 3; ++k) {
    odometry.lidar_poses.push_back(
        Pose(Mat3::Identity(), Vec3(k, 0.5 * k, 0.0)));
  }
  odometry.kf_frames = {0, 2};
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(0.0, 0.0, 1.5));
  WriteTripOdometry(dir.string(), odometry, vehicle_from_lidar);
  const TripOdometryFile read = ReadTripOdometry(dir.string());
  GR_EXPECT_TRUE(read.kf_frames == odometry.kf_frames);
  GR_EXPECT_NEAR(read.lidar_poses[2].translation.x(), 2.0, 0.0);
  GR_EXPECT_TRUE(fs::exists(dir / "odometry.tum"));

  const std::string sweep = SweepPath(dir.string(), "kf", 2);
  GR_EXPECT_EQ(fs::path(sweep).filename().string(), std::string("000002.npz"));
  WriteSweep(sweep, {Vec3(0.1, 0.2, 0.3)}, {0.5f});
  const SweepFile loaded = ReadSweep(sweep);
  GR_ASSERT_TRUE(loaded.points.size() == 1);
  // float32 storage, as odometry.run_trip_odometry writes it.
  GR_EXPECT_NEAR(loaded.points[0].x(), static_cast<double>(0.1f), 0.0);
  GR_EXPECT_NEAR(loaded.intensity[0], 0.5, 0.0);

  KeyframeTable table;
  table.poses = {Pose(), Pose()};
  table.odom = table.poses;
  table.trip = {0, 1};
  table.frame = {0, 2};
  table.stamp = {0.1, 0.3};
  const std::string path = (dir / "graph" / "kf_poses.npz").string();
  WriteKeyframeTable(path, table);
  const NpzArchive archive = ReadNpz(path);
  // No IMU factors: (0, 3).
  GR_EXPECT_TRUE(archive.at("velocity").shape ==
                 (std::vector<int64_t>{0, 3}));
  GR_EXPECT_EQ(archive.at("trip").item_size, 8);
  GR_EXPECT_TRUE(ReadKeyframeTable(path).frame == table.frame);

  PoseSetIds ids;
  ids.trajectory_id = "a";
  ids.pose_set_id = "b";
  RecordProvenance(dir.string(), "recon.map", ids, false);
  RecordProvenance(dir.string(), "recon.trajectory", ids, false);
  const Value provenance = ReadJsonFile((dir / "provenance.json").string());
  GR_EXPECT_EQ(provenance.size(), size_t{2});
  GR_EXPECT_EQ(provenance.members().front().first, std::string("recon.map"));
  fs::remove_all(dir);
}

}  // namespace
}  // namespace gr
