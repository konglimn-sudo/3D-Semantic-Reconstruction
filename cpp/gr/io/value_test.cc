#include "gr/io/value.h"

#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>

#include "gr/io/config.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

// What PyYAML's safe_dump and the hand-written configuration look like.
const char kYaml[] = R"(# comment
lidars:
  lidar_top:
    T_vehicle_lidar:
    - - 1.0
      - 0.0
    - - 0.5
      - 2
    sweep_duration: 0.1   # s
odometry:
  voxel_size: 0.4            # local map voxel (indoor: 0.3-0.5)
  use_lio: false             # LiDAR-inertial odometry, rigs with
                             # an IMU in degenerate geometry
  external_trajectory: null  # e.g. "{trip}/lio.tum"
  gates: [1.0, 0.3]
  lr: {means: 1.6e-5, sh0: 2.5e-3}
  word: auto  # auto | oracle
  quoted: "a: b # not a comment"
  single: 'it''s'
  plain: it's here
  exponent: 1e-5
  flag: yes
  infinite: -.inf
items:
- name: a
  value: 1
- name: b
empty:
)";

GR_TEST(ValueTest, ParsesTheYamlTheProjectWrites) {
  constexpr double kTolerance = 1e-20;
  const Value document = ParseYaml(kYaml);
  const Value& lidar = document["lidars"]["lidar_top"];
  GR_ASSERT_TRUE(lidar["T_vehicle_lidar"].size() == 2);
  GR_EXPECT_NEAR(lidar["T_vehicle_lidar"][1][0].AsDouble(), 0.5, 0.0);
  GR_EXPECT_TRUE(lidar["T_vehicle_lidar"][1][1].is_integer());
  GR_EXPECT_NEAR(lidar["sweep_duration"].AsDouble(), 0.1, 0.0);
  const Value& odometry = document["odometry"];
  GR_EXPECT_FALSE(odometry["use_lio"].AsBool());
  GR_EXPECT_TRUE(odometry["external_trajectory"].is_null());
  GR_EXPECT_TRUE(odometry.Find("external_trajectory") != nullptr);
  GR_EXPECT_EQ(odometry["gates"].AsDoubles().size(), size_t{2});
  GR_EXPECT_NEAR(odometry["lr"]["means"].AsDouble(), 1.6e-5, kTolerance);
  GR_EXPECT_EQ(odometry["word"].AsString(), std::string("auto"));
  GR_EXPECT_EQ(odometry["quoted"].AsString(),
               std::string("a: b # not a comment"));
  GR_EXPECT_EQ(odometry["single"].AsString(), std::string("it's"));
  GR_EXPECT_EQ(odometry["plain"].AsString(), std::string("it's here"));
  // YAML 1.1 (PyYAML): a float needs a dot, booleans include yes / no.
  GR_EXPECT_TRUE(odometry["exponent"].is_string());
  GR_EXPECT_TRUE(odometry["flag"].AsBool());
  GR_EXPECT_TRUE(std::isinf(odometry["infinite"].AsDouble()));
  GR_EXPECT_EQ(document["items"].size(), size_t{2});
  GR_EXPECT_EQ(document["items"][0]["value"].AsInt(), 1);
  GR_EXPECT_EQ(document["items"][1]["name"].AsString(), std::string("b"));
  GR_EXPECT_TRUE(document["empty"].is_null());
  GR_EXPECT_TRUE(document["missing"].is_null());
}

GR_TEST(ValueTest, RejectsWhatItDoesNotSupport) {
  // A block scalar, an anchor, a bad indentation and an open flow list.
  const char* const kUnsupported[] = {"a: |\n  text\n", "a: &x 1\n",
                                      "a: 1\n   b: 2\n", "a: [1, 2\n"};
  int errors = 0;
  for (const char* text : kUnsupported) {
    try {
      ParseYaml(text);
    } catch (const std::runtime_error&) {
      ++errors;
    }
  }
  GR_EXPECT_EQ(errors, static_cast<int>(std::size(kUnsupported)));
}

GR_TEST(ValueTest, JsonRoundTripsAsPythonWritesIt) {
  const std::string text =
      "{\n \"trip\": \"出口\",\n \"n\": 3,\n \"x\": [\n  0.0129,\n  1e-05,\n"
      "  -180.0,\n  1e+16\n ],\n \"none\": null,\n \"ok\": true,\n"
      " \"empty\": {}\n}";
  const Value document = ParseJson(text);
  GR_EXPECT_EQ(document["trip"].AsString(), std::string("出口"));
  GR_EXPECT_TRUE(document["n"].is_integer());
  GR_EXPECT_EQ(ToJson(document, kDefaultJsonIndent), text);
  GR_EXPECT_EQ(ParseJson("\"\\u51fa\\ud83d\\ude00\"").AsString(),
               std::string("出\xF0\x9F\x98\x80"));
  GR_EXPECT_EQ(ToJson(Value::Number(0.1 + 0.2), kJsonOneLine),
               std::string("0.30000000000000004"));
  GR_EXPECT_EQ(ToJson(Value::Number(1234.5), kJsonOneLine),
               std::string("1234.5"));
}

GR_TEST(ValueTest, DeepMergeOverridesMemberByMember) {
  const Value base = ParseYaml("a:\n  b: 1\n  c: [1, 2]\nd: 4\n");
  const Value over = ParseYaml("a:\n  c: [3]\n  e: 5\n");
  const Value merged = DeepMerge(base, over);
  GR_EXPECT_EQ(merged["a"]["b"].AsInt(), 1);
  GR_EXPECT_EQ(merged["a"]["c"].size(), size_t{1});
  GR_EXPECT_EQ(merged["a"]["e"].AsInt(), 5);
  GR_EXPECT_EQ(merged["d"].AsInt(), 4);
}

GR_TEST(ConfigTest, ImuGradeFillsWhatTheUserDoesNotSet) {
  const Value defaults = ParseYaml(
      "imu_grade: consumer\n"
      "imu_grades:\n"
      "  consumer: {backend: {gyro_bias_prior: 0.01, accel_bias_prior: 0.2}}\n"
      "  tactical:\n"
      "    backend: {gyro_bias_prior: 0.002, accel_bias_prior: 0.03}\n"
      "    odometry: {lio: {init_accel_bias_sigma: 0.03}}\n"
      "backend: {gyro_bias_prior: 0.01, accel_bias_prior: 0.2, x: 1}\n");
  const Value user =
      ParseYaml("imu_grade: tactical\nbackend: {gyro_bias_prior: 0.004}\n");
  const Value resolved = ApplyImuGrade(DeepMerge(defaults, user), user);
  GR_EXPECT_NEAR(resolved["backend"]["gyro_bias_prior"].AsDouble(), 0.004, 0.0);
  GR_EXPECT_NEAR(resolved["backend"]["accel_bias_prior"].AsDouble(), 0.03, 0.0);
  GR_EXPECT_NEAR(
      resolved["odometry"]["lio"]["init_accel_bias_sigma"].AsDouble(), 0.03,
      0.0);
  GR_EXPECT_TRUE(resolved["imu_grade_applied"].AsBool());
  // A resolved configuration loaded again keeps its values.
  Value edited = resolved;
  edited.Find("backend")->Set("accel_bias_prior", Value::Number(0.5));
  GR_EXPECT_NEAR(
      ApplyImuGrade(edited, Value::Map())["backend"]["accel_bias_prior"]
          .AsDouble(),
      0.5, 0.0);
  const LioConfig lio = OdometryConfigFrom(resolved).lio;
  GR_EXPECT_NEAR(lio.init_accel_bias_sigma, 0.03, 0.0);
}

}  // namespace
}  // namespace gr
