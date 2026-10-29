#include "gr/semantics/identity.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <utility>

#include "gr/core/check.h"

namespace gr {
namespace {

// The carrier z-buffer: splats of kMinSplatRadius to kMaxSplatRadius pixels,
// points from kMinRenderRange (m) to the range plus kRangeSlack (m).
constexpr int kMinSplatRadius = 1;
constexpr int kMaxSplatRadius = 3;
constexpr double kMinRenderRange = 0.1;
constexpr double kRangeSlack = 1e-6;
// Directions of shorter vectors have no meaning.
constexpr double kMinNorm = 1e-9;
// Points facing the camera at most this cosine are facing away or edge-on;
// the grazing tolerance divides by at least kMinToleranceCos.
constexpr double kMinCosIncidence = 0.05;
constexpr double kMinToleranceCos = 0.15;
// Every covered point casts one vote.
constexpr float kVoteWeight = 1.0f;

}  // namespace

IdentityAssociator::IdentityAssociator(Points points, Points normals,
                                       const IdentityConfig& config)
    : config_(config), index_(std::move(points)), normals_(std::move(normals)) {
  GR_CHECK(normals_.empty() || normals_.size() == index_.size());
  votes_.resize(index_.size());
}

int IdentityAssociator::Find(int identity) const {
  int root = identity;
  while (parent_[static_cast<size_t>(root)] != root) {
    root = parent_[static_cast<size_t>(root)];
  }
  while (parent_[static_cast<size_t>(identity)] != root) {  // Compress.
    const int next = parent_[static_cast<size_t>(identity)];
    parent_[static_cast<size_t>(identity)] = root;
    identity = next;
  }
  return root;
}

int IdentityAssociator::NewIdentity() {
  const int identity = static_cast<int>(parent_.size());
  parent_.push_back(identity);
  cannot_link_.emplace_back();
  return identity;
}

int IdentityAssociator::Merge(const std::vector<int>& roots) {
  GR_CHECK(!roots.empty());
  const int target = *std::min_element(roots.begin(), roots.end());
  for (int root : roots) {
    if (root == target) continue;
    parent_[static_cast<size_t>(root)] = target;
    std::vector<int>& links = cannot_link_[static_cast<size_t>(target)];
    std::vector<int>& absorbed = cannot_link_[static_cast<size_t>(root)];
    links.insert(links.end(), absorbed.begin(), absorbed.end());
    absorbed.clear();
  }
  return target;
}

bool IdentityAssociator::CannotLink(int first, int second) const {
  first = Find(first);
  second = Find(second);
  for (int other : cannot_link_[static_cast<size_t>(first)]) {
    if (Find(other) == second) return true;
  }
  return false;
}

int IdentityAssociator::CurrentIdentity(size_t point) const {
  int best = kNoIdentity;
  float best_votes = 0.0f;
  const std::vector<std::pair<int, float>>& votes = votes_[point];
  for (const std::pair<int, float>& candidate : votes) {
    const int root = Find(candidate.first);
    float total = 0.0f;
    for (const std::pair<int, float>& entry : votes) {
      if (Find(entry.first) == root) total += entry.second;
    }
    if (total > best_votes || (total == best_votes && root < best)) {
      best = root;
      best_votes = total;
    }
  }
  return best;
}

void IdentityAssociator::Vote(size_t point, int identity) {
  std::vector<std::pair<int, float>>& votes = votes_[point];
  for (std::pair<int, float>& entry : votes) {
    if (Find(entry.first) == identity) {
      entry.second += kVoteWeight;
      return;
    }
  }
  votes.emplace_back(identity, kVoteWeight);
}

std::vector<MaskRecord> IdentityAssociator::AddView(
    const CameraModel& camera, const CameraExposure& exposure,
    const int32_t* masks) {
  const int view = num_views_++;
  const Vec3& center = exposure.center();
  const std::vector<int> candidates =
      index_.RadiusSearch(center, config_.max_range + kRangeSlack);
  ZBuffer buffer(camera, exposure, config_.point_spacing, kMinSplatRadius,
                 kMaxSplatRadius, kMinRenderRange, config_.max_range);
  for (int candidate : candidates) {
    buffer.Splat(index_.points()[static_cast<size_t>(candidate)],
                 RenderResult::kEmpty, candidate);
  }
  const RenderResult& render = buffer.result();
  const int height = render.height;
  const int width = render.width;
  const auto mask_at = [&](int y, int x) {
    y = std::clamp(y, 0, height - 1);
    x = std::clamp(x, 0, width - 1);
    return masks[static_cast<size_t>(y) * static_cast<size_t>(width) +
                 static_cast<size_t>(x)];
  };

  std::map<int, MaskRecord> by_mask;  // Ordered: deterministic decisions.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int mask = mask_at(y, x);
      if (mask < 0) continue;
      MaskRecord& record = by_mask[mask];
      ++record.pixels;
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        record.touches_border = true;
      }
    }
  }

  // Every visible carrier point is covered by the mask at its pixel (away
  // from mask boundaries). Visible points are also counted per identity.
  std::vector<std::pair<int, int>> pairs;
  std::map<int, int> visible_by_identity;
  for (int candidate : candidates) {
    const Vec3& point = index_.points()[static_cast<size_t>(candidate)];
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(camera, point);
    if (!projection || !camera.InImage(projection->uv)) continue;
    const int x = static_cast<int>(projection->uv.x());
    const int y = static_cast<int>(projection->uv.y());
    const float depth =
        render.depth[static_cast<size_t>(y) * static_cast<size_t>(width) +
                     static_cast<size_t>(x)];
    const double range = projection->point_camera.norm();
    double tolerance = config_.depth_abs_tol;
    if (!normals_.empty()) {
      const Vec3 to_camera = center - point;
      const double cos_incidence =
          normals_[static_cast<size_t>(candidate)].dot(to_camera) /
          std::max(to_camera.norm(), kMinNorm);
      if (cos_incidence <= kMinCosIncidence) continue;
      tolerance += config_.grazing_pixels * range /
                   (camera.focal() * std::max(cos_incidence, kMinToleranceCos));
    }
    if (std::isfinite(depth) &&
        range > depth * (1.0 + config_.depth_rel_tol) + tolerance) {
      continue;  // Occluded.
    }
    const int owner = CurrentIdentity(static_cast<size_t>(candidate));
    if (owner != kNoIdentity) ++visible_by_identity[owner];
    const int mask = mask_at(y, x);
    if (mask < 0) continue;
    bool uniform = true;
    for (int row_offset = -config_.boundary;
         row_offset <= config_.boundary && uniform; ++row_offset) {
      for (int column_offset = -config_.boundary;
           column_offset <= config_.boundary && uniform; ++column_offset) {
        uniform = mask_at(y + row_offset, x + column_offset) == mask;
      }
    }
    if (uniform) pairs.emplace_back(mask, candidate);
  }
  std::sort(pairs.begin(), pairs.end());

  // Decide every mask's identity before any of them votes.
  struct Decision {
    int mask = 0;
    int identity = kNoIdentity;
    size_t first = 0;  // Range of the mask's points in `pairs`.
    size_t last = 0;
  };
  std::vector<Decision> decisions;
  size_t begin = 0;
  while (begin < pairs.size()) {
    const int mask = pairs[begin].first;
    size_t end = begin;
    double range_sum = 0.0;
    std::map<int, int> owners;  // Root identity -> identified points.
    int known = 0;
    while (end < pairs.size() && pairs[end].first == mask) {
      const size_t point = static_cast<size_t>(pairs[end].second);
      range_sum += (index_.points()[point] - center).norm();
      const int owner = CurrentIdentity(point);
      if (owner != kNoIdentity) {
        ++owners[owner];
        ++known;
      }
      ++end;
    }
    MaskRecord& record = by_mask[mask];
    record.points = static_cast<int>(end - begin);
    record.mean_range = range_sum / static_cast<double>(end - begin);
    if (record.points >= config_.min_mask_points) {
      int identity = kNoIdentity;
      const double needed =
          std::max<double>(config_.min_common_points,
                           config_.min_known_fraction * record.points);
      if (known >= needed) {
        std::vector<std::pair<int, int>> ranked;  // (-count, root)
        const double min_count = std::max<double>(config_.min_common_points,
                                                  config_.merge_share * known);
        for (const auto& [owner, count] : owners) {
          if (count >= min_count) ranked.emplace_back(-count, owner);
        }
        std::sort(ranked.begin(), ranked.end());
        std::vector<int> group;
        int group_count = 0;
        for (const auto& [negative_count, owner] : ranked) {
          bool linked = false;
          for (int member : group) linked = linked || CannotLink(owner, member);
          if (linked) continue;
          if (!group.empty()) {  // Merging needs most of the visible object.
            const auto visible = visible_by_identity.find(owner);
            const int seen =
                visible == visible_by_identity.end() ? 0 : visible->second;
            if (-negative_count < config_.merge_coverage * seen) continue;
          }
          group.push_back(owner);
          group_count -= negative_count;
        }
        if (!group.empty() && group_count >= config_.min_share * known) {
          identity = group.size() > 1 ? Merge(group) : group.front();
        }
      }
      if (identity == kNoIdentity) identity = NewIdentity();
      decisions.push_back({mask, identity, begin, end});
    }
    begin = end;
  }

  // Masks of one view are different objects unless they share an identity.
  std::vector<int> roots;
  for (const Decision& decision : decisions) {
    roots.push_back(Find(decision.identity));
  }
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  for (size_t first = 0; first < roots.size(); ++first) {
    for (size_t second = first + 1; second < roots.size(); ++second) {
      if (CannotLink(roots[first], roots[second])) continue;
      cannot_link_[static_cast<size_t>(roots[first])].push_back(roots[second]);
      cannot_link_[static_cast<size_t>(roots[second])].push_back(roots[first]);
    }
  }
  for (const Decision& decision : decisions) {
    const int root = Find(decision.identity);
    by_mask[decision.mask].identity = root;
    for (size_t pair = decision.first; pair < decision.last; ++pair) {
      Vote(static_cast<size_t>(pairs[pair].second), root);
    }
  }

  std::vector<MaskRecord> records;
  records.reserve(by_mask.size());
  for (auto& [mask, record] : by_mask) {
    record.view = view;
    record.mask = mask;
    records.push_back(record);
  }
  records_.insert(records_.end(), records.begin(), records.end());
  return records;
}

IdentityAssociator::Result IdentityAssociator::Finish() const {
  // Final roots, renumbered 0..K-1 in ascending root order.
  std::map<int, int> renumber;
  std::vector<int> point_root(votes_.size(), kNoIdentity);
  for (size_t point = 0; point < votes_.size(); ++point) {
    point_root[point] = CurrentIdentity(point);
    if (point_root[point] != kNoIdentity) {
      renumber.emplace(point_root[point], 0);
    }
  }
  for (const MaskRecord& record : records_) {
    if (record.identity != kNoIdentity) {
      renumber.emplace(Find(record.identity), 0);
    }
  }
  int next = 0;
  for (auto& [root, number] : renumber) number = next++;

  Result result;
  result.num_identities = next;
  result.point_identity.assign(votes_.size(), kNoIdentity);
  result.point_confidence.assign(votes_.size(), 0.0f);
  for (size_t point = 0; point < votes_.size(); ++point) {
    const int root = point_root[point];
    if (root == kNoIdentity) continue;
    float total = 0.0f;
    float winner = 0.0f;
    for (const std::pair<int, float>& entry : votes_[point]) {
      total += entry.second;
      if (Find(entry.first) == root) winner += entry.second;
    }
    result.point_identity[point] = renumber.find(root)->second;
    result.point_confidence[point] = total > 0.0f ? winner / total : 0.0f;
  }
  result.masks = records_;
  for (MaskRecord& record : result.masks) {
    if (record.identity != kNoIdentity) {
      record.identity = renumber.find(Find(record.identity))->second;
    }
  }
  return result;
}

}  // namespace gr
