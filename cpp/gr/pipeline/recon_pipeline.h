// The geometric stages of the offline reconstruction without Python: the
// stages odometry, graph, trajectory and map of recon/pipeline.py, on the
// same work directory layout (gr/io/recon_files.h), so either side can
// resume the other's work.
//
//   odometry    per trip: OdometryPipeline over the sweeps (wheel / gyro /
//               LIO prior), keyframe sweeps kept as float32 / float16;
//   graph       anchors, IMU factors, loops (+ externally proposed pairs),
//               pose graph and LiDAR BA (hierarchical above
//               ba.hierarchical_above keyframes); frame poses;
//   trajectory  the pose set (continuous-time trajectories, rig) and the
//               keyframe sweeps deskewed again with it;
//   map         fused point map, normals and sigmas, levels and ramps,
//               dynamic points.
//
// The stages that need learned models (appearance / text loop proposals,
// camera calibration, semantics) stay on the Python side; their loop pairs
// can be passed in (set_loop_pairs).

#ifndef GARAGE_RECON_GR_PIPELINE_RECON_PIPELINE_H_
#define GARAGE_RECON_GR_PIPELINE_RECON_PIPELINE_H_

#include <string>
#include <utility>
#include <vector>

#include "gr/core/logging.h"
#include "gr/io/trip.h"
#include "gr/io/value.h"

namespace gr {

class ReconPipeline {
 public:
  // `config`: the merged configuration (LoadConfig), saved as
  // <work_dir>/config.used.yaml. Throws std::runtime_error for a directory
  // that is not a trip.
  ReconPipeline(const std::vector<std::string>& trip_dirs,
                std::string work_dir, Value config, Logger log);

  // odometry, graph, trajectory, map.
  static const std::vector<std::string>& StageNames();

  // Runs the stages in order, logging the time of each; throws
  // std::runtime_error for an unknown stage.
  void Run(const std::vector<std::string>& stages);

  void StageOdometry();
  void StageGraph();
  void StageTrajectory();
  void StageMap();

  // Keyframe pairs (indices into the keyframes of all trips, in trip
  // order) proposed by image retrieval or text, verified in the graph
  // stage like appearance loops.
  void set_loop_pairs(std::vector<std::pair<int, int>> pairs) {
    loop_pairs_ = std::move(pairs);
  }

 private:
  struct KeyframeEntry {
    int trip = 0;
    int64_t frame = 0;
    double stamp = 0.0;
    Pose odom;
    std::string path;  // The sweep file.
  };

  std::string TripDir(size_t trip) const;
  // Keyframes of all trips; `final` selects the kf_final/ sweeps when the
  // trajectory stage wrote them.
  std::vector<KeyframeEntry> Keyframes(bool final) const;

  std::vector<TripReader> trips_;
  std::string work_;
  Value config_;
  Logger log_;
  std::vector<std::pair<int, int>> loop_pairs_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_PIPELINE_RECON_PIPELINE_H_
