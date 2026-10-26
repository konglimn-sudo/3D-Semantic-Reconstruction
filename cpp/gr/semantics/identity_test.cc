#include "gr/semantics/identity.h"

#include <algorithm>
#include <map>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Floor (segment 0), a wall (1) and two boxes (2, 3) as carrier points.
void Scene(Points& points, std::vector<int>& segments) {
  for (double x = -2.0; x <= 22.0; x += 0.05) {
    for (double y = 0.0; y <= 8.0; y += 0.05) {
      points.emplace_back(x, y, 0.0);
      segments.push_back(0);
    }
    for (double z = 0.0; z <= 3.0; z += 0.05) {
      points.emplace_back(x, 8.0, z);
      segments.push_back(1);
    }
  }
  const auto add_box = [&](const Vec3& lo, const Vec3& hi, int segment) {
    for (double x = lo.x(); x <= hi.x(); x += 0.05) {
      for (double y = lo.y(); y <= hi.y(); y += 0.05) {
        points.emplace_back(x, y, hi.z());
        segments.push_back(segment);
      }
      for (double z = lo.z(); z <= hi.z(); z += 0.05) {
        points.emplace_back(x, lo.y(), z);
        segments.push_back(segment);
      }
    }
  };
  add_box(Vec3(5.0, 4.0, 0.0), Vec3(9.5, 5.8, 1.5), 2);
  add_box(Vec3(12.0, 4.5, 0.0), Vec3(16.0, 6.3, 1.4), 3);
}

GR_TEST(IdentityAssociatorTest, LinksMasksAcrossViewsThroughGeometry) {
  Points points;
  std::vector<int> segments;
  Scene(points, segments);
  const auto camera = MakeCamera("pinhole", 320, 240, {200, 200, 160, 120});
  GR_ASSERT_TRUE(camera != nullptr);
  // Camera at 1 m height looking +y (left of a vehicle driving along x).
  Mat3 world_from_optical;
  world_from_optical << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  IdentityAssociator associator(points, {}, IdentityConfig());
  std::mt19937 rng(3);
  int view = 0;
  for (double x = 0.0; x <= 20.0; x += 0.5, ++view) {
    const CameraExposure exposure(
        Pose(So3Exp(Vec3(-0.25, 0.0, 0.0)) * world_from_optical,
             Vec3(x, -1.0, 1.2)));
    // A perfect segmenter with arbitrary mask ids per view; every third
    // view over-segments the first box into left and right parts.
    const RenderResult render = RenderPoints(*camera, exposure, points,
                                             segments, 0.05, 1, 3, 0.1, 15.0);
    std::vector<int> ids{0, 1, 2, 3, 4};
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int32_t> masks(render.label.size(), -1);
    for (size_t p = 0; p < masks.size(); ++p) {
      const int segment = render.label[p];
      if (segment < 0) continue;
      masks[p] = 10 * ids[static_cast<size_t>(segment)];
      if (segment == 2 && view % 3 == 0 && p % 320 >= 160) {
        masks[p] = 10 * ids[4] + 1;
      }
    }
    associator.AddView(*camera, exposure, masks.data());
  }
  const IdentityAssociator::Result result = associator.Finish();
  GR_EXPECT_EQ(associator.num_views(), view);
  GR_EXPECT_FALSE(result.masks.empty());
  // Every segment ends with one dominant identity, all different.
  std::map<int, int> dominant;
  for (int segment = 0; segment < 4; ++segment) {
    std::map<int, int> counts;
    int covered = 0;
    for (size_t i = 0; i < points.size(); ++i) {
      if (segments[i] != segment || result.point_identity[i] < 0) continue;
      ++counts[result.point_identity[i]];
      ++covered;
    }
    GR_ASSERT_TRUE(covered > 100);
    int best = -1;
    int best_count = 0;
    for (const auto& [identity, count] : counts) {
      if (count > best_count) {
        best = identity;
        best_count = count;
      }
    }
    GR_EXPECT_GT(static_cast<double>(best_count) / covered, 0.95);
    dominant[best] = segment;
  }
  GR_EXPECT_EQ(dominant.size(), 4u);
}

GR_TEST(IdentityAssociatorTest, NormalsCullTheOtherFaceOfASlab) {
  // A ceiling (normal down) at z = 3 and the floor of the level above on the
  // same 0.3 m slab (normal up). Seen from below, only the ceiling is visible.
  Points points;
  Points normals;
  for (double x = -3.0; x <= 3.0; x += 0.1) {
    for (double y = -3.0; y <= 3.0; y += 0.1) {
      points.emplace_back(x, y, 3.0);
      normals.push_back(-Vec3::UnitZ());
      points.emplace_back(x, y, 3.3);
      normals.push_back(Vec3::UnitZ());
    }
  }
  const auto camera = MakeCamera("pinhole", 160, 120, {80, 80, 80, 60});
  GR_ASSERT_TRUE(camera != nullptr);
  IdentityConfig config;
  config.depth_abs_tol = 0.6;  // Loose enough to let the upper floor through.
  for (const bool with_normals : {false, true}) {
    IdentityAssociator associator(points, with_normals ? normals : Points(),
                                  config);
    for (double x = -1.0; x <= 1.0; x += 0.5) {
      // Identity rotation: the optical axis (camera z) points up.
      const CameraExposure exposure(Pose(Mat3::Identity(), Vec3(x, 0, 1.0)));
      const std::vector<int32_t> masks(160 * 120, 0);  // One ceiling mask.
      associator.AddView(*camera, exposure, masks.data());
    }
    const IdentityAssociator::Result result = associator.Finish();
    int upper = 0;
    int ceiling = 0;
    for (size_t i = 0; i < points.size(); ++i) {
      if (result.point_identity[i] < 0) continue;
      (points[i].z() > 3.1 ? upper : ceiling) += 1;
    }
    GR_EXPECT_GT(ceiling, 500);
    if (with_normals) {
      GR_EXPECT_EQ(upper, 0);
    } else {
      GR_EXPECT_GT(upper, 0);  // Without normals the slab leaks through.
    }
  }
}

}  // namespace
}  // namespace gr
