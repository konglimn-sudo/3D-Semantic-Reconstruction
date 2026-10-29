#include "gr/backend/loop_closure.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <utility>

#include "gr/backend/scan_context.h"
#include "gr/core/check.h"
#include "gr/registration/cloud_compare.h"
#include "gr/registration/registration.h"

namespace gr {
namespace {

// The coarse search estimates normals over this many coarse voxels.
constexpr double kCoarseNormalRadiusVoxels = 2.5;
// Hypotheses whose yaws differ by less (deg) share a yaw.
constexpr double kDegreesPerTurn = 2.0 * kDegreesPerHalfTurn;
constexpr double kSameYawTolerance = 1.0;
// Weight (m/rad) of a cycle's rotation error against its translation.
constexpr double kCycleMetresPerRadian = 5.0;

// Candidates of the query keyframe ranked by Scan Context distance: earlier
// keyframes near the current estimate on the same level. `had_candidates`
// says whether any keyframe passed the position gates.
std::vector<std::pair<double, int>> RankCandidates(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<Eigen::MatrixXd>& descriptors, int query_index,
    const LoopConfig& config, bool* had_candidates) {
  const Keyframe& query = keyframes[static_cast<size_t>(query_index)];
  const Vec3& query_position =
      poses[static_cast<size_t>(query_index)].translation;
  std::vector<std::pair<double, int>> scored;
  *had_candidates = false;
  for (int candidate = 0; candidate < query_index; ++candidate) {
    const Vec3& position = poses[static_cast<size_t>(candidate)].translation;
    if ((position.head<kPlaneDim>() - query_position.head<kPlaneDim>()).norm() >
        config.search_radius) {
      continue;
    }
    if (std::abs(position.z() - query_position.z()) > config.level_gate) {
      continue;
    }
    if (keyframes[static_cast<size_t>(candidate)].trip == query.trip &&
        candidate > query_index - config.min_index_gap) {
      continue;
    }
    *had_candidates = true;
    const ScanContextMatch match =
        MatchScanContexts(descriptors[static_cast<size_t>(query_index)],
                          descriptors[static_cast<size_t>(candidate)]);
    if (match.distance <= config.sc_threshold) {
      scored.emplace_back(match.distance, candidate);
    }
  }
  std::sort(scored.begin(), scored.end());
  if (static_cast<int>(scored.size()) > config.top_k) {
    scored.resize(static_cast<size_t>(std::max(0, config.top_k)));
  }
  return scored;
}

// Verifies the ranked candidates of the query keyframe by ICP, counting the
// runs in `tested`; the first accepted one is the loop.
std::optional<LoopConstraint> VerifyCandidates(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    int query_index, const std::vector<std::pair<double, int>>& candidates,
    const LoopConfig& config, int* tested) {
  const Points& source =
      keyframes.Cloud(static_cast<size_t>(query_index), config.icp_voxel);
  for (const auto& [distance, candidate] : candidates) {
    ++*tested;
    const Pose initial = poses[static_cast<size_t>(candidate)].Inverse() *
                         poses[static_cast<size_t>(query_index)];
    const Points target =
        keyframes.Submap(static_cast<size_t>(candidate),
                         config.submap_half_window, config.icp_voxel, poses);
    const IcpResult icp =
        AlignMultiScale(source, target, initial, config.icp_schedule,
                        config.icp_voxel, config.icp_iters, /*parallel=*/false);
    const double correction = (initial.Inverse() * icp.pose).translation.norm();
    if (icp.fitness >= config.min_fitness && icp.rmse <= config.max_rmse &&
        correction <= config.max_correction) {
      LoopConstraint loop;
      loop.i = query_index;
      loop.j = candidate;
      loop.j_from_i = icp.pose;
      loop.fitness = icp.fitness;
      loop.rmse = icp.rmse;
      loop.sc_distance = distance;
      return loop;
    }
  }
  return std::nullopt;
}

}  // namespace

std::vector<LoopConstraint> DetectLoops(const KeyframeSet& keyframes,
                                        const std::vector<Pose>& poses,
                                        const LoopConfig& config,
                                        LoopStats* stats) {
  std::vector<LoopConstraint> loops;
  const int num_keyframes = static_cast<int>(keyframes.size());
  if (stats != nullptr) *stats = LoopStats();
  if (!config.enabled || num_keyframes < kMinLoopKeyframes) return loops;
  GR_CHECK(static_cast<int>(poses.size()) == num_keyframes);

  ScanContextConfig sc_config;
  sc_config.rings = config.sc_rings;
  sc_config.sectors = config.sc_sectors;
  sc_config.max_radius = config.sc_max_radius;
  std::vector<Eigen::MatrixXd> descriptors(static_cast<size_t>(num_keyframes));
#ifdef _OPENMP
  constexpr int kDescriptorsPerChunk = 8;
#pragma omp parallel for schedule(dynamic, kDescriptorsPerChunk)
#endif
  for (int keyframe = 0; keyframe < num_keyframes; ++keyframe) {
    descriptors[static_cast<size_t>(keyframe)] = ComputeScanContext(
        keyframes.Cloud(static_cast<size_t>(keyframe), config.sc_voxel),
        sc_config);
  }

  std::vector<int> queries;
  for (int keyframe = 0; keyframe < num_keyframes;
       keyframe += std::max(1, config.stride)) {
    queries.push_back(keyframe);
  }
  const int num_queries = static_cast<int>(queries.size());
  std::vector<std::optional<LoopConstraint>> found(
      static_cast<size_t>(num_queries));
  std::vector<int> tested(static_cast<size_t>(num_queries), 0);
  std::vector<int> queried(static_cast<size_t>(num_queries), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int query = 0; query < num_queries; ++query) {
    const size_t slot = static_cast<size_t>(query);
    const int query_index = queries[slot];
    bool had_candidates = false;
    const std::vector<std::pair<double, int>> candidates = RankCandidates(
        keyframes, poses, descriptors, query_index, config, &had_candidates);
    queried[slot] = had_candidates ? 1 : 0;
    found[slot] = VerifyCandidates(keyframes, poses, query_index, candidates,
                                   config, &tested[slot]);
  }
  for (size_t query = 0; query < static_cast<size_t>(num_queries); ++query) {
    if (found[query]) loops.push_back(*found[query]);
    if (stats != nullptr) {
      stats->tested += tested[query];
      stats->queries += queried[query];
    }
  }
  return loops;
}

std::vector<LoopConstraint> VerifyAppearanceLoops(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<std::pair<int, int>>& candidates,
    const LoopConfig& config, const AppearanceLoopConfig& appearance,
    int* tested) {
  const int num_candidates = static_cast<int>(candidates.size());
  std::vector<std::optional<LoopConstraint>> found(
      static_cast<size_t>(num_candidates));
  std::vector<int> runs(static_cast<size_t>(num_candidates), 0);
  CoarseAlignConfig coarse;
  coarse.voxel = appearance.coarse_voxel;
  coarse.normal_radius = kCoarseNormalRadiusVoxels * appearance.coarse_voxel;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int candidate = 0; candidate < num_candidates; ++candidate) {
    const size_t slot = static_cast<size_t>(candidate);
    const auto [source_index, target_index] = candidates[slot];
    const int num_keyframes = static_cast<int>(keyframes.size());
    if (source_index < 0 || target_index < 0 || source_index >= num_keyframes ||
        target_index >= num_keyframes || source_index == target_index) {
      continue;
    }
    const size_t source_keyframe = static_cast<size_t>(source_index);
    const size_t target_keyframe = static_cast<size_t>(target_index);
    const Points source_world = TransformPoints(
        keyframes.Submap(source_keyframe, config.submap_half_window,
                         appearance.coarse_voxel, poses),
        poses[source_keyframe]);
    const Points target_world = TransformPoints(
        keyframes.Submap(target_keyframe, config.submap_half_window,
                         appearance.coarse_voxel, poses),
        poses[target_keyframe]);
    const std::vector<CoarseAlignment> alignments =
        AlignGravityAlignedCandidates(source_world, target_world, coarse,
                                      appearance.z_peak_ratio);
    // Every yaw first (its best height), then the other heights by score: in
    // a symmetric layout the right yaw can rank below the mirrored one, and
    // both must be tested for the ambiguity gap to see them.
    std::vector<CoarseAlignment> hypotheses;
    std::vector<char> taken(alignments.size(), 0);
    std::vector<double> yaws;  // deg
    for (size_t alignment = 0; alignment < alignments.size(); ++alignment) {
      const double yaw =
          RadToDeg(YawOf(alignments[alignment].reference_from_source.rotation));
      bool seen = false;
      for (double seen_yaw : yaws) {
        seen =
            seen || std::abs(std::remainder(yaw - seen_yaw, kDegreesPerTurn)) <
                        kSameYawTolerance;
      }
      if (seen) continue;
      yaws.push_back(yaw);
      hypotheses.push_back(alignments[alignment]);
      taken[alignment] = 1;
    }
    for (size_t alignment = 0; alignment < alignments.size(); ++alignment) {
      if (!taken[alignment]) hypotheses.push_back(alignments[alignment]);
    }
    if (static_cast<int>(hypotheses.size()) > appearance.max_hypotheses) {
      hypotheses.resize(static_cast<size_t>(appearance.max_hypotheses));
    }
    const Points& source = keyframes.Cloud(source_keyframe, config.icp_voxel);
    const Points target = keyframes.Submap(
        target_keyframe, config.submap_half_window, config.icp_voxel, poses);
    std::vector<IcpResult> results;
    for (const CoarseAlignment& hypothesis : hypotheses) {
      const Pose initial = poses[target_keyframe].Inverse() *
                           hypothesis.reference_from_source *
                           poses[source_keyframe];
      results.push_back(AlignMultiScale(source, target, initial,
                                        config.icp_schedule, config.icp_voxel,
                                        config.icp_iters, /*parallel=*/false));
      ++runs[slot];
    }
    if (results.empty()) continue;
    size_t best = 0;
    for (size_t result = 1; result < results.size(); ++result) {
      if (results[result].fitness > results[best].fitness) best = result;
    }
    // The best distinct solution (repeated layouts give a second peak).
    double runner_up = 0.0;
    for (size_t result = 0; result < results.size(); ++result) {
      const Pose best_from_result =
          results[best].pose.Inverse() * results[result].pose;
      if (best_from_result.translation.norm() > appearance.distinct_t ||
          RadToDeg(RotationAngle(best_from_result.rotation)) >
              appearance.distinct_r) {
        runner_up = std::max(runner_up, results[result].fitness);
      }
    }
    const IcpResult& icp = results[best];
    if (icp.fitness < config.min_fitness || icp.rmse > config.max_rmse ||
        icp.fitness - runner_up < appearance.min_fitness_gap) {
      continue;
    }
    LoopConstraint loop;
    loop.i = source_index;
    loop.j = target_index;
    loop.j_from_i = icp.pose;
    loop.fitness = icp.fitness;
    loop.rmse = icp.rmse;
    found[slot] = loop;
  }
  std::vector<LoopConstraint> loops;
  for (size_t slot = 0; slot < found.size(); ++slot) {
    if (found[slot]) loops.push_back(*found[slot]);
    if (tested != nullptr) *tested += runs[slot];
  }
  return loops;
}

std::vector<LoopConstraint> SelectConsistentLoops(
    const std::vector<LoopConstraint>& loops, const KeyframeSet& keyframes,
    const std::vector<Pose>& poses, const LoopConfig& config) {
  const size_t num_loops = loops.size();
  if (num_loops < kMinConsistencyLoops) return loops;
  const std::vector<double> path_distance = keyframes.PathDistance(poses);
  const auto trip = [&](int keyframe) {
    return keyframes[static_cast<size_t>(keyframe)].trip;
  };
  const auto pose = [&](int keyframe) {
    return poses[static_cast<size_t>(keyframe)];
  };
  const auto distance = [&](int keyframe) {
    return path_distance[static_cast<size_t>(keyframe)];
  };
  std::vector<std::vector<char>> consistent(num_loops,
                                            std::vector<char>(num_loops, 0));
  for (size_t a = 0; a < num_loops; ++a) {
    consistent[a][a] = 1;
    for (size_t b = a + 1; b < num_loops; ++b) {
      const LoopConstraint& loop_a = loops[a];
      const LoopConstraint& loop_b = loops[b];
      if (trip(loop_a.i) != trip(loop_b.i) ||
          trip(loop_a.j) != trip(loop_b.j)) {
        // Cannot be checked with odometry alone.
        consistent[a][b] = consistent[b][a] = 1;
        continue;
      }
      const Pose odometry_i = pose(loop_a.i).Inverse() * pose(loop_b.i);
      const Pose odometry_j = pose(loop_a.j).Inverse() * pose(loop_b.j);
      // T_ia^-1 T_ib = Z_a^-1 O_j Z_b must match O_i.
      const Pose cycle = odometry_i.Inverse() * loop_a.j_from_i.Inverse() *
                         odometry_j * loop_b.j_from_i;
      const double angle = std::acos(
          std::clamp((cycle.rotation.trace() - 1.0) / 2.0, -1.0, 1.0));
      const double error =
          cycle.translation.norm() + kCycleMetresPerRadian * angle;
      const double travelled =
          std::abs(distance(loop_a.i) - distance(loop_b.i)) +
          std::abs(distance(loop_a.j) - distance(loop_b.j));
      consistent[a][b] = consistent[b][a] =
          error <= config.pcm_threshold + config.pcm_drift_per_m * travelled;
    }
  }
  // Greedy clique: repeatedly take the loop consistent with most of the
  // remaining ones (ties: lowest index) and keep it if it agrees with all
  // loops kept so far.
  std::vector<size_t> selected;
  std::set<size_t> remaining;
  for (size_t loop = 0; loop < num_loops; ++loop) remaining.insert(loop);
  while (!remaining.empty()) {
    size_t best = *remaining.begin();
    int best_degree = -1;
    for (size_t loop : remaining) {
      int degree = 0;
      for (size_t other : remaining) degree += consistent[loop][other];
      if (degree > best_degree) {
        best_degree = degree;
        best = loop;
      }
    }
    bool agrees = true;
    for (size_t kept_loop : selected) {
      agrees = agrees && consistent[best][kept_loop];
    }
    if (agrees) selected.push_back(best);
    remaining.erase(best);
  }
  std::sort(selected.begin(), selected.end());
  std::vector<LoopConstraint> kept;
  kept.reserve(selected.size());
  for (size_t loop : selected) kept.push_back(loops[loop]);
  return kept;
}

}  // namespace gr
