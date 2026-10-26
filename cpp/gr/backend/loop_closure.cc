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

// Candidates of query keyframe i ranked by Scan Context distance: earlier
// keyframes near the current estimate on the same level.
std::vector<std::pair<double, int>> RankCandidates(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<Eigen::MatrixXd>& descriptors, int i,
    const LoopConfig& config, bool& had_candidates) {
  const Keyframe& query = keyframes[static_cast<size_t>(i)];
  const Vec3& pi = poses[static_cast<size_t>(i)].translation;
  std::vector<std::pair<double, int>> scored;
  had_candidates = false;
  for (int j = 0; j < i; ++j) {
    const Vec3& pj = poses[static_cast<size_t>(j)].translation;
    if ((pj.head<2>() - pi.head<2>()).norm() > config.search_radius) continue;
    if (std::abs(pj.z() - pi.z()) > config.level_gate) continue;
    if (keyframes[static_cast<size_t>(j)].trip == query.trip &&
        j > i - config.min_index_gap) {
      continue;
    }
    had_candidates = true;
    const ScanContextMatch match =
        MatchScanContexts(descriptors[static_cast<size_t>(i)],
                          descriptors[static_cast<size_t>(j)]);
    if (match.distance <= config.sc_threshold) {
      scored.emplace_back(match.distance, j);
    }
  }
  std::sort(scored.begin(), scored.end());
  if (static_cast<int>(scored.size()) > config.top_k) {
    scored.resize(static_cast<size_t>(std::max(0, config.top_k)));
  }
  return scored;
}

// Verifies the ranked candidates of keyframe i by ICP; the first accepted
// one is the loop.
std::optional<LoopConstraint> VerifyCandidates(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses, int i,
    const std::vector<std::pair<double, int>>& candidates,
    const LoopConfig& config, int& tested) {
  const Points& source =
      keyframes.Cloud(static_cast<size_t>(i), config.icp_voxel);
  for (const auto& [distance, j] : candidates) {
    ++tested;
    const Pose initial =
        poses[static_cast<size_t>(j)].Inverse() * poses[static_cast<size_t>(i)];
    const Points target =
        keyframes.Submap(static_cast<size_t>(j), config.submap_half_window,
                         config.icp_voxel, poses);
    const IcpResult icp =
        AlignMultiScale(source, target, initial, config.icp_schedule,
                        config.icp_voxel, config.icp_iters, false);
    const double correction = (initial.Inverse() * icp.pose).translation.norm();
    if (icp.fitness >= config.min_fitness && icp.rmse <= config.max_rmse &&
        correction <= config.max_correction) {
      LoopConstraint loop;
      loop.i = i;
      loop.j = j;
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
  if (!config.enabled || num_keyframes < 2) return loops;
  GR_CHECK(static_cast<int>(poses.size()) == num_keyframes);

  ScanContextConfig sc_config;
  sc_config.rings = config.sc_rings;
  sc_config.sectors = config.sc_sectors;
  sc_config.max_radius = config.sc_max_radius;
  std::vector<Eigen::MatrixXd> descriptors(static_cast<size_t>(num_keyframes));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
  for (int k = 0; k < num_keyframes; ++k) {
    descriptors[static_cast<size_t>(k)] = ComputeScanContext(
        keyframes.Cloud(static_cast<size_t>(k), config.sc_voxel), sc_config);
  }

  std::vector<int> queries;
  for (int i = 0; i < num_keyframes; i += std::max(1, config.stride)) {
    queries.push_back(i);
  }
  const int num_queries = static_cast<int>(queries.size());
  std::vector<std::optional<LoopConstraint>> found(
      static_cast<size_t>(num_queries));
  std::vector<int> tested(static_cast<size_t>(num_queries), 0);
  std::vector<int> had(static_cast<size_t>(num_queries), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int q = 0; q < num_queries; ++q) {
    const size_t slot = static_cast<size_t>(q);
    const int i = queries[slot];
    bool had_candidates = false;
    const std::vector<std::pair<double, int>> candidates = RankCandidates(
        keyframes, poses, descriptors, i, config, had_candidates);
    had[slot] = had_candidates ? 1 : 0;
    found[slot] =
        VerifyCandidates(keyframes, poses, i, candidates, config, tested[slot]);
  }
  for (size_t q = 0; q < static_cast<size_t>(num_queries); ++q) {
    if (found[q]) loops.push_back(*found[q]);
    if (stats != nullptr) {
      stats->tested += tested[q];
      stats->queries += had[q];
    }
  }
  return loops;
}

std::vector<LoopConstraint> VerifyAppearanceLoops(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<std::pair<int, int>>& candidates,
    const LoopConfig& config, const AppearanceLoopConfig& appearance,
    int* tested) {
  const int n = static_cast<int>(candidates.size());
  std::vector<std::optional<LoopConstraint>> found(static_cast<size_t>(n));
  std::vector<int> runs(static_cast<size_t>(n), 0);
  CoarseAlignConfig coarse;
  coarse.voxel = appearance.coarse_voxel;
  coarse.normal_radius = 2.5 * appearance.coarse_voxel;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int c = 0; c < n; ++c) {
    const auto [i, j] = candidates[static_cast<size_t>(c)];
    if (i < 0 || j < 0 || i >= static_cast<int>(keyframes.size()) ||
        j >= static_cast<int>(keyframes.size()) || i == j) {
      continue;
    }
    const size_t ii = static_cast<size_t>(i), jj = static_cast<size_t>(j);
    const Points source_world = TransformPoints(
        keyframes.Submap(ii, config.submap_half_window,
                         appearance.coarse_voxel, poses),
        poses[ii]);
    const Points target_world = TransformPoints(
        keyframes.Submap(jj, config.submap_half_window,
                         appearance.coarse_voxel, poses),
        poses[jj]);
    const std::vector<CoarseAlignment> all = AlignGravityAlignedCandidates(
        source_world, target_world, coarse, appearance.z_peak_ratio);
    // Every yaw first (its best height), then the other heights by score: in
    // a symmetric layout the right yaw can rank below the mirrored one, and
    // both must be tested for the ambiguity gap to see them.
    std::vector<CoarseAlignment> hypotheses;
    std::vector<char> taken(all.size(), 0);
    std::vector<double> yaws;
    for (size_t h = 0; h < all.size(); ++h) {
      const Mat3& r = all[h].reference_from_source.rotation;
      const double yaw = RadToDeg(std::atan2(r(1, 0), r(0, 0)));
      bool seen = false;
      for (double y : yaws) {
        seen = seen || std::abs(std::remainder(yaw - y, 360.0)) < 1.0;
      }
      if (seen) continue;
      yaws.push_back(yaw);
      hypotheses.push_back(all[h]);
      taken[h] = 1;
    }
    for (size_t h = 0; h < all.size(); ++h) {
      if (!taken[h]) hypotheses.push_back(all[h]);
    }
    if (static_cast<int>(hypotheses.size()) > appearance.max_hypotheses) {
      hypotheses.resize(static_cast<size_t>(appearance.max_hypotheses));
    }
    const Points& source = keyframes.Cloud(ii, config.icp_voxel);
    const Points target =
        keyframes.Submap(jj, config.submap_half_window, config.icp_voxel,
                         poses);
    std::vector<IcpResult> results;
    for (const CoarseAlignment& h : hypotheses) {
      const Pose initial =
          poses[jj].Inverse() * h.reference_from_source * poses[ii];
      results.push_back(AlignMultiScale(source, target, initial,
                                        config.icp_schedule, config.icp_voxel,
                                        config.icp_iters, false));
      ++runs[static_cast<size_t>(c)];
    }
    if (results.empty()) continue;
    size_t best = 0;
    for (size_t r = 1; r < results.size(); ++r) {
      if (results[r].fitness > results[best].fitness) best = r;
    }
    // The best distinct solution (repeated layouts give a second peak).
    double runner_up = 0.0;
    for (size_t r = 0; r < results.size(); ++r) {
      const Pose d = results[best].pose.Inverse() * results[r].pose;
      if (d.translation.norm() > appearance.distinct_t ||
          RadToDeg(RotationAngle(d.rotation)) > appearance.distinct_r) {
        runner_up = std::max(runner_up, results[r].fitness);
      }
    }
    const IcpResult& icp = results[best];
    if (icp.fitness < config.min_fitness || icp.rmse > config.max_rmse ||
        icp.fitness - runner_up < appearance.min_fitness_gap) {
      continue;
    }
    LoopConstraint loop;
    loop.i = i;
    loop.j = j;
    loop.j_from_i = icp.pose;
    loop.fitness = icp.fitness;
    loop.rmse = icp.rmse;
    found[static_cast<size_t>(c)] = loop;
  }
  std::vector<LoopConstraint> loops;
  for (int c = 0; c < n; ++c) {
    if (found[static_cast<size_t>(c)]) loops.push_back(*found[static_cast<size_t>(c)]);
    if (tested != nullptr) *tested += runs[static_cast<size_t>(c)];
  }
  return loops;
}

std::vector<LoopConstraint> SelectConsistentLoops(
    const std::vector<LoopConstraint>& loops, const KeyframeSet& keyframes,
    const std::vector<Pose>& poses, const LoopConfig& config) {
  const size_t n = loops.size();
  if (n < 3) return loops;
  const std::vector<double> distance = keyframes.PathDistance(poses);
  std::vector<std::vector<char>> consistent(n, std::vector<char>(n, 0));
  for (size_t a = 0; a < n; ++a) {
    consistent[a][a] = 1;
    for (size_t b = a + 1; b < n; ++b) {
      const LoopConstraint& la = loops[a];
      const LoopConstraint& lb = loops[b];
      const auto trip = [&](int k) {
        return keyframes[static_cast<size_t>(k)].trip;
      };
      if (trip(la.i) != trip(lb.i) || trip(la.j) != trip(lb.j)) {
        // Cannot be checked with odometry alone.
        consistent[a][b] = consistent[b][a] = 1;
        continue;
      }
      const auto pose = [&](int k) { return poses[static_cast<size_t>(k)]; };
      const Pose odometry_i = pose(la.i).Inverse() * pose(lb.i);
      const Pose odometry_j = pose(la.j).Inverse() * pose(lb.j);
      // T_ia^-1 T_ib = Z_a^-1 O_j Z_b must match O_i.
      const Pose cycle = odometry_i.Inverse() * la.j_from_i.Inverse() *
                         odometry_j * lb.j_from_i;
      const double angle = std::acos(
          std::clamp((cycle.rotation.trace() - 1.0) / 2.0, -1.0, 1.0));
      const double error = cycle.translation.norm() + 5.0 * angle;
      const auto dist = [&](int k) { return distance[static_cast<size_t>(k)]; };
      const double travelled =
          std::abs(dist(la.i) - dist(lb.i)) + std::abs(dist(la.j) - dist(lb.j));
      consistent[a][b] = consistent[b][a] =
          error <= config.pcm_threshold + config.pcm_drift_per_m * travelled;
    }
  }
  // Greedy clique: repeatedly take the loop consistent with most of the
  // remaining ones (ties: lowest index) and keep it if it agrees with all
  // loops kept so far.
  std::vector<size_t> selected;
  std::set<size_t> remaining;
  for (size_t v = 0; v < n; ++v) remaining.insert(v);
  while (!remaining.empty()) {
    size_t best = *remaining.begin();
    int best_degree = -1;
    for (size_t v : remaining) {
      int degree = 0;
      for (size_t u : remaining) degree += consistent[v][u];
      if (degree > best_degree) {
        best_degree = degree;
        best = v;
      }
    }
    bool agrees = true;
    for (size_t s : selected) agrees = agrees && consistent[best][s];
    if (agrees) selected.push_back(best);
    remaining.erase(best);
  }
  std::sort(selected.begin(), selected.end());
  std::vector<LoopConstraint> kept;
  kept.reserve(selected.size());
  for (size_t s : selected) kept.push_back(loops[s]);
  return kept;
}

}  // namespace gr
