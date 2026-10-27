#include "gr/semantics/markings.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Eigenvalues>

#include "gr/core/check.h"
#include "gr/core/morphology.h"

namespace gr {
namespace {

Vec2 Unit(const Vec2& v) { return v / std::max(v.norm(), 1e-9); }

}  // namespace

std::vector<MarkingSegment> MergeCollinear(std::vector<MarkingSegment> segs,
                                           const MarkingConfig& config) {
  const size_t n = segs.size();
  if (n < 2) return segs;
  std::vector<uint8_t> alive(n, 1);
  const double cos_max = std::cos(DegToRad(config.merge_angle));
  for (size_t a = 0; a < n; ++a) {
    while (alive[a]) {
      const Vec2 da = Unit(segs[a].p1 - segs[a].p0);
      const Vec2 nrm(-da.y(), da.x());
      const double la = (segs[a].p1 - segs[a].p0).norm();
      size_t partner = n;
      double t0 = 0.0;
      double t1 = 0.0;
      for (size_t k = 0; k < n; ++k) {
        if (k == a || !alive[k]) continue;
        const MarkingSegment& b = segs[k];
        const Vec2 db = Unit(b.p1 - b.p0);
        if (std::abs(db.dot(da)) < cos_max) continue;
        if (std::max(std::abs((b.p0 - segs[a].p0).dot(nrm)),
                     std::abs((b.p1 - segs[a].p0).dot(nrm))) >
            config.merge_offset) {
          continue;
        }
        const double s0 = (b.p0 - segs[a].p0).dot(da);
        const double s1 = (b.p1 - segs[a].p0).dot(da);
        const double gap =
            std::max(std::min(s0, s1) - la, -std::max(s0, s1));
        if (gap > config.merge_gap) continue;
        partner = k;
        t0 = s0;
        t1 = s1;
        break;
      }
      if (partner == n) break;
      const MarkingSegment& A = segs[a];
      const MarkingSegment& B = segs[partner];
      const double lo = std::min({0.0, t0, t1});
      const double hi = std::max({la, t0, t1});
      const double lb = B.length();
      const double total = std::max(la + lb, 1e-9);
      MarkingSegment merged;
      merged.level = A.level;
      merged.p0 = A.p0 + lo * da;
      merged.p1 = A.p0 + hi * da;
      merged.width = (A.width * la + B.width * lb) / total;
      merged.lane = A.lane;
      merged.confidence =
          std::min(1.0, (A.confidence * la + B.confidence * lb) / total);
      segs[a] = merged;
      alive[partner] = 0;
    }
  }
  std::vector<MarkingSegment> out;
  for (size_t i = 0; i < n; ++i) {
    if (alive[i]) out.push_back(segs[i]);
  }
  return out;
}

std::vector<MarkingSegment> VectorizeMarkings(
    const std::vector<uint8_t>& mask, const RasterFrame& frame,
    const MarkingConfig& config, int level,
    const std::vector<uint8_t>& given_skeleton) {
  const int width = frame.width;
  const int height = frame.height;
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  GR_CHECK(given_skeleton.empty() || given_skeleton.size() == mask.size());
  const double res = frame.res;
  const Mask skeleton = given_skeleton.empty()
                            ? Skeletonize(mask, width, height)
                            : given_skeleton;
  // Junctions: skeleton cells with three or more skeleton neighbours; the
  // branches are what is left away from them.
  Mask junction(skeleton.size(), 0);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      if (!skeleton[static_cast<size_t>(j * width + i)]) continue;
      int neighbours = 0;
      for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          if (di == 0 && dj == 0) continue;
          const int ni = i + di;
          const int nj = j + dj;
          neighbours += ni >= 0 && nj >= 0 && ni < width && nj < height &&
                        skeleton[static_cast<size_t>(nj * width + ni)];
        }
      }
      junction[static_cast<size_t>(j * width + i)] = neighbours >= 3;
    }
  }
  const Mask near_junction = DilateCross(junction, width, height);
  Mask branches(skeleton.size(), 0);
  for (size_t c = 0; c < branches.size(); ++c) {
    branches[c] = skeleton[c] && !near_junction[c];
  }
  Mask background(mask.size());
  for (size_t c = 0; c < mask.size(); ++c) background[c] = mask[c] == 0;
  const std::vector<float> dist = DistanceTransform(background, width, height);
  std::vector<MarkingSegment> segs;
  for (const std::vector<int>& cells : Label8(branches, width, height).cells) {
    if (static_cast<double>(cells.size()) * res < config.min_length * 0.5) {
      continue;
    }
    Vec2 mean = Vec2::Zero();
    for (int c : cells) mean += Vec2(c % width + 0.5, c / width + 0.5);
    mean /= static_cast<double>(cells.size());
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (int c : cells) {
      const Vec2 d = Vec2(c % width + 0.5, c / width + 0.5) - mean;
      scatter += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(scatter);
    Vec2 d = solver.eigenvectors().col(1);
    // A deterministic orientation: the larger component positive.
    if ((std::abs(d.x()) >= std::abs(d.y()) ? d.x() : d.y()) < 0.0) d = -d;
    const Vec2 nrm(-d.y(), d.x());
    double t_min = std::numeric_limits<double>::infinity();
    double t_max = -t_min;
    double resid = 0.0;
    double depth = 0.0;
    for (int c : cells) {
      const Vec2 p = Vec2(c % width + 0.5, c / width + 0.5) - mean;
      const double t = p.dot(d);
      t_min = std::min(t_min, t);
      t_max = std::max(t_max, t);
      resid += std::abs(p.dot(nrm));
      depth += dist[static_cast<size_t>(c)];
    }
    if (resid / static_cast<double>(cells.size()) > 2.0) {
      continue;  // A curved branch (arrow head, text).
    }
    MarkingSegment seg;
    seg.level = level;
    const Vec2 origin(frame.x0, frame.y0);
    seg.p0 = origin + res * (mean + t_min * d);
    seg.p1 = origin + res * (mean + t_max * d);
    seg.width = 2.0 * depth / static_cast<double>(cells.size()) * res;
    segs.push_back(seg);
  }
  segs = MergeCollinear(std::move(segs), config);
  std::vector<MarkingSegment> out;
  for (const MarkingSegment& s : segs) {
    if (s.length() >= config.min_length && s.width >= config.min_width &&
        s.width <= config.max_width) {
      out.push_back(s);
    }
  }
  return out;
}

bool BothSidesObserved(const MarkingSegment& seg,
                       const std::vector<uint8_t>& floor,
                       const RasterFrame& frame, double offset,
                       double min_fraction) {
  const Vec2 d = Unit(seg.p1 - seg.p0);
  const Vec2 n = Vec2(-d.y(), d.x()) * (seg.width / 2.0 + offset);
  for (double side : {1.0, -1.0}) {
    int inside = 0;
    int observed = 0;
    for (int k = 0; k < 20; ++k) {
      const double t = 0.1 + 0.8 * k / 19.0;
      const Vec2 p = seg.p0 + t * (seg.p1 - seg.p0) + side * n;
      const int ci =
          static_cast<int>(std::floor((p.x() - frame.x0) / frame.res));
      const int cj =
          static_cast<int>(std::floor((p.y() - frame.y0) / frame.res));
      if (ci < 0 || cj < 0 || ci >= frame.width || cj >= frame.height) continue;
      ++inside;
      observed += floor[static_cast<size_t>(cj * frame.width + ci)] != 0;
    }
    const double fraction =
        inside > 0 ? static_cast<double>(observed) / inside : 0.0;
    if (fraction < min_fraction) return false;
  }
  return true;
}

bool GridAligned(const MarkingSegment& seg, double yaw,
                 double max_angle_deg) {
  const Vec2 d = seg.p1 - seg.p0;
  const double quarter = kPi / 2.0;
  double a = std::atan2(d.y(), d.x()) - yaw;
  a -= std::floor(a / quarter) * quarter;  // Python's modulo: [0, pi / 2).
  return std::min(a, quarter - a) <= DegToRad(max_angle_deg);
}

int RefineSegments(std::vector<MarkingSegment>& segs,
                   const std::vector<const FloorImage*>& images,
                   const LineFitConfig& config) {
  int refined = 0;
  for (MarkingSegment& seg : segs) {
    std::optional<LineFit> fit;
    for (const FloorImage* image : images) {
      fit = FitLine(seg.p0, seg.p1, *image, config);
      if (fit) break;
    }
    if (!fit) continue;
    seg.p0 = fit->p0;
    seg.p1 = fit->p1;
    seg.width = fit->width;
    seg.end_ok[0] = fit->end_ok[0];
    seg.end_ok[1] = fit->end_ok[1];
    ++refined;
  }
  return refined;
}

void ClassifyMarkings(std::vector<MarkingSegment>& segs,
                      const std::map<int, std::vector<Vec2>>& paths,
                      const MarkingConfig& config) {
  const double cos_max = std::cos(DegToRad(10.0));
  for (MarkingSegment& s : segs) {
    s.lane = false;
    const auto found = paths.find(s.level);
    if (found == paths.end() || found->second.size() < 3 ||
        s.length() < config.lane_min_length) {
      continue;
    }
    const std::vector<Vec2>& path = found->second;
    const Vec2 mid = 0.5 * (s.p0 + s.p1);
    size_t k = 0;
    for (size_t i = 1; i < path.size(); ++i) {
      if ((path[i] - mid).norm() < (path[k] - mid).norm()) k = i;
    }
    const Vec2 t = path[std::min(path.size() - 1, k + 5)] -
                   path[k >= 5 ? k - 5 : 0];
    const Vec2 d = s.p1 - s.p0;
    if (t.norm() > 0.5 &&
        std::abs(t.dot(d)) / (t.norm() * d.norm()) > cos_max) {
      s.lane = true;
    }
  }
}

}  // namespace gr
