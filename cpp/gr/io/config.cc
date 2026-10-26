#include "gr/io/config.h"

namespace gr {
namespace {

const FieldTable<LioConfig>& LioFields() {
  static const FieldTable<LioConfig> table =
      FieldTable<LioConfig>()
          .Add("map_voxel", &LioConfig::map_voxel)
          .Add("max_points_per_voxel", &LioConfig::max_points_per_voxel)
          .Add("map_radius", &LioConfig::map_radius)
          .Add("scan_voxel", &LioConfig::scan_voxel)
          .Add("insert_voxel", &LioConfig::insert_voxel)
          .Add("max_iterations", &LioConfig::max_iterations)
          .Add("convergence", &LioConfig::convergence)
          .Add("plane_radius", &LioConfig::plane_radius)
          .Add("max_plane_radius", &LioConfig::max_plane_radius)
          .Add("min_plane_points", &LioConfig::min_plane_points)
          .Add("min_planarity", &LioConfig::min_planarity)
          .Add("max_plane_thickness", &LioConfig::max_plane_thickness)
          .Add("max_residual", &LioConfig::max_residual)
          .Add("point_sigma", &LioConfig::point_sigma)
          .Add("robust_scale", &LioConfig::robust_scale)
          .Add("point_weight", &LioConfig::point_weight)
          .Add("zupt_window", &LioConfig::zupt_window)
          .Add("zupt_max_gyro", &LioConfig::zupt_max_gyro)
          .Add("zupt_max_accel", &LioConfig::zupt_max_accel)
          .Add("zupt_sigma", &LioConfig::zupt_sigma)
          .Add("max_speed", &LioConfig::max_speed)
          .Add("speed_sigma", &LioConfig::speed_sigma)
          .Add("gyro_noise", &LioConfig::gyro_noise)
          .Add("accel_noise", &LioConfig::accel_noise)
          .Add("gyro_bias_walk", &LioConfig::gyro_bias_walk)
          .Add("accel_bias_walk", &LioConfig::accel_bias_walk)
          .Add("gravity", &LioConfig::gravity)
          .Add("init_velocity_sigma", &LioConfig::init_velocity_sigma)
          .Add("init_tilt_sigma_deg", &LioConfig::init_tilt_sigma_deg)
          .Add("init_gyro_bias_sigma", &LioConfig::init_gyro_bias_sigma)
          .Add("init_accel_bias_sigma", &LioConfig::init_accel_bias_sigma);
  return table;
}

const FieldTable<OdometryConfig>& OdometryFields() {
  static const FieldTable<OdometryConfig> table =
      FieldTable<OdometryConfig>()
          .Add("max_range", &OdometryConfig::max_range)
          .Add("min_range", &OdometryConfig::min_range)
          .Add("voxel_size", &OdometryConfig::voxel_size)
          .Add("max_points_per_voxel", &OdometryConfig::max_points_per_voxel)
          .Add("initial_threshold", &OdometryConfig::initial_threshold)
          .Add("min_motion_th", &OdometryConfig::min_motion_th)
          .Add("max_icp_iters", &OdometryConfig::max_icp_iters)
          .Add("icp_conv", &OdometryConfig::icp_conv)
          .Add("deskew", &OdometryConfig::deskew)
          .Add("use_wheel_prior", &OdometryConfig::use_wheel_prior)
          .Add("use_gyro_prior", &OdometryConfig::use_gyro_prior)
          .Add("use_lio", &OdometryConfig::use_lio);
  return table;
}

const FieldTable<KeyframeConfig>& KeyframeFields() {
  static const FieldTable<KeyframeConfig> table =
      FieldTable<KeyframeConfig>()
          .Add("distance", &KeyframeConfig::distance)
          .Add("angle", &KeyframeConfig::angle)
          .Add("voxel", &KeyframeConfig::voxel);
  return table;
}

const FieldTable<PlacementConfig>& PlacementFields() {
  static const FieldTable<PlacementConfig> table =
      FieldTable<PlacementConfig>()
          .Add("voxel", &PlacementConfig::voxel)
          .Add("icp_schedule", &PlacementConfig::icp_schedule)
          .Add("icp_iters", &PlacementConfig::icp_iters)
          .Add("min_score", &PlacementConfig::min_score)
          .Add("z_peak_ratio", &PlacementConfig::z_peak_ratio)
          .Add("max_candidates", &PlacementConfig::max_candidates)
          .Add("min_fitness", &PlacementConfig::min_fitness)
          .Add("max_rmse", &PlacementConfig::max_rmse)
          .Add("min_fitness_gap", &PlacementConfig::min_fitness_gap)
          .Add("distinct_t", &PlacementConfig::distinct_t)
          .Add("distinct_r", &PlacementConfig::distinct_r);
  return table;
}

const FieldTable<LocalizerConfig>& LocalizerFields() {
  using C = LocalizerConfig;
  static const FieldTable<C> table =
      FieldTable<C>()
          .Add("map_voxel", &C::map_voxel)
          .Add("hash_voxel", &C::hash_voxel)
          .Add("scan_voxel", &C::scan_voxel)
          .Add("min_range", &C::min_range)
          .Add("max_range", &C::max_range)
          .Add("deskew", &C::deskew)
          .Add("gates", &C::gates)
          .Add("icp_iters", &C::icp_iters)
          .Add("min_fitness", &C::min_fitness)
          .Add("max_rmse", &C::max_rmse)
          .Add("max_jump_t", &C::max_jump_t)
          .Add("max_jump_r", &C::max_jump_r)
          .Add("degenerate_ratio", &C::degenerate_ratio)
          .Add("degenerate_sigma_t", &C::degenerate_sigma_t)
          .Add("degenerate_sigma_r", &C::degenerate_sigma_r)
          .Add("max_failures", &C::max_failures)
          .Add("prior_schedule", &C::prior_schedule)
          .Add("init_voxel", &C::init_voxel)
          .Add("init_spacing", &C::init_spacing)
          .Add("init_distance", &C::init_distance)
          .Add("init_retry", &C::init_retry)
          .Add("init_max_distance", &C::init_max_distance);
  return table;
}

}  // namespace

Value LoadConfig(const std::vector<std::string>& paths) {
  Value config = Value::Map();
  for (const std::string& path : paths) {
    const Value file = ReadYamlFile(path);
    if (!file.is_null()) config = DeepMerge(config, file);
  }
  return config;
}

OdometryConfig OdometryConfigFrom(const Value& config) {
  OdometryConfig out;
  const Value& section = config["odometry"];
  OdometryFields().Fill(section, &out);
  out.lio = LioConfig();
  LioFields().Fill(section["lio"], &out.lio);
  return out;
}

KeyframeConfig KeyframeConfigFrom(const Value& config) {
  KeyframeConfig out;
  KeyframeFields().Fill(config["keyframes"], &out);
  return out;
}

LocalizerConfig LocalizerConfigFrom(const Value& config) {
  LocalizerConfig out;
  const Value& section = config["localization"];
  LocalizerFields().Fill(section, &out);
  // As localize.localizer_config: the placement starts from its own
  // defaults, not the localizer's.
  out.placement = PlacementConfig();
  PlacementFields().Fill(section["placement"], &out.placement);
  return out;
}

TripLocalizerConfig TripLocalizerConfigFrom(const Value& config) {
  TripLocalizerConfig out;
  out.odometry = OdometryConfigFrom(config);
  out.keyframes = KeyframeConfigFrom(config);
  out.localizer = LocalizerConfigFrom(config);
  const Value& window = config["localization"]["level_window"];
  if (!window.is_null()) out.level_window = window.AsDouble();
  return out;
}

}  // namespace gr
