#include "gr/identity.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <utility>

#include "gr/check.h"

namespace gr {

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

bool IdentityAssociator::CannotLink(int a, int b) const {
  a = Find(a);
  b = Find(b);
  for (int other : cannot_link_[static_cast<size_t>(a)]) {
    if (Find(other) == b) return true;
  }
  return false;
}

int IdentityAssociator::CurrentIdentity(size_t point) const {
  int best = -1;
  float best_votes = 0.0f;
  const std::vector<std::pair<int, float>>& votes = votes_[point];
  for (size_t a = 0; a < votes.size(); ++a) {
    const int root = Find(votes[a].first);
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
      entry.second += 1.0f;
      return;
    }
  }
  votes.emplace_back(identity, 1.0f);
}

std::vector<MaskRecord> IdentityAssociator::AddView(
    const CameraModel& camera, const CameraExposure& exposure,
    const int32_t* masks) {
  const int view = num_views_++;
  const Vec3& center = exposure.center();
  const std::vector<int> candidates =
      index_.RadiusSearch(center, config_.max_range + 1e-6);
  ZBuffer buffer(camera, exposure, config_.point_spacing, 1, 3, 0.1,
                 config_.max_range);
  for (int i : candidates) {
    buffer.Splat(index_.points()[static_cast<size_t>(i)], -1, i);
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
  for (int i : candidates) {
    const Vec3& point = index_.points()[static_cast<size_t>(i)];
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
          normals_[static_cast<size_t>(i)].dot(to_camera) /
          std::max(to_camera.norm(), 1e-9);
      if (cos_incidence <= 0.05) continue;  // Facing away or edge-on.
      tolerance += config_.grazing_pixels * range /
                   (camera.focal() * std::max(cos_incidence, 0.15));
    }
    if (std::isfinite(depth) &&
        range > depth * (1.0 + config_.depth_rel_tol) + tolerance) {
      continue;  // Occluded.
    }
    const int owner = CurrentIdentity(static_cast<size_t>(i));
    if (owner >= 0) ++visible_by_identity[owner];
    const int mask = mask_at(y, x);
    if (mask < 0) continue;
    bool uniform = true;
    for (int dy = -config_.boundary; dy <= config_.boundary && uniform; ++dy) {
      for (int dx = -config_.boundary; dx <= config_.boundary && uniform;
           ++dx) {
        uniform = mask_at(y + dy, x + dx) == mask;
      }
    }
    if (uniform) pairs.emplace_back(mask, i);
  }
  std::sort(pairs.begin(), pairs.end());

  // Decide every mask's identity before any of them votes.
  struct Decision {
    int mask = 0;
    int identity = -1;
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
      if (owner >= 0) {
        ++owners[owner];
        ++known;
      }
      ++end;
    }
    MaskRecord& record = by_mask[mask];
    record.points = static_cast<int>(end - begin);
    record.mean_range = range_sum / static_cast<double>(end - begin);
    if (record.points >= config_.min_mask_points) {
      int identity = -1;
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
      if (identity < 0) identity = NewIdentity();
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
  for (size_t a = 0; a < roots.size(); ++a) {
    for (size_t b = a + 1; b < roots.size(); ++b) {
      if (CannotLink(roots[a], roots[b])) continue;
      cannot_link_[static_cast<size_t>(roots[a])].push_back(roots[b]);
      cannot_link_[static_cast<size_t>(roots[b])].push_back(roots[a]);
    }
  }
  for (const Decision& decision : decisions) {
    const int root = Find(decision.identity);
    by_mask[decision.mask].identity = root;
    for (size_t p = decision.first; p < decision.last; ++p) {
      Vote(static_cast<size_t>(pairs[p].second), root);
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
  std::vector<int> point_root(votes_.size(), -1);
  for (size_t i = 0; i < votes_.size(); ++i) {
    point_root[i] = CurrentIdentity(i);
    if (point_root[i] >= 0) renumber.emplace(point_root[i], 0);
  }
  for (const MaskRecord& record : records_) {
    if (record.identity >= 0) renumber.emplace(Find(record.identity), 0);
  }
  int next = 0;
  for (auto& [root, number] : renumber) number = next++;

  Result result;
  result.num_identities = next;
  result.point_identity.assign(votes_.size(), -1);
  result.point_confidence.assign(votes_.size(), 0.0f);
  for (size_t i = 0; i < votes_.size(); ++i) {
    const int root = point_root[i];
    if (root < 0) continue;
    float total = 0.0f;
    float winner = 0.0f;
    for (const std::pair<int, float>& entry : votes_[i]) {
      total += entry.second;
      if (Find(entry.first) == root) winner += entry.second;
    }
    result.point_identity[i] = renumber.find(root)->second;
    result.point_confidence[i] = total > 0.0f ? winner / total : 0.0f;
  }
  result.masks = records_;
  for (MaskRecord& record : result.masks) {
    if (record.identity >= 0) {
      record.identity = renumber.find(Find(record.identity))->second;
    }
  }
  return result;
}

}  // namespace gr
