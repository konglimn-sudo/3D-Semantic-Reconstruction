#include "gr/semantics/identity.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kPointStep = 0.05;  // m between carrier points.
// Segments of the scene.
constexpr int kFloorSegment = 0;
constexpr int kWallSegment = 1;
constexpr int kFirstBoxSegment = 2;
constexpr int kSecondBoxSegment = 3;
constexpr int kSegments = 4;

// Floor (kFloorSegment), a wall at y = 8 (kWallSegment) and two boxes as
// carrier points.
void Scene(Points& points, std::vector<int>& segments) {
  constexpr double kWallY = 8.0;   // m
  constexpr double kWallTop = 3.0;  // m
  for (double x = -2.0; x <= 22.0; x += kPointStep) {
    for (double y = 0.0; y <= kWallY; y += kPointStep) {
      points.emplace_back(x, y, 0.0);
      segments.push_back(kFloorSegment);
    }
    for (double z = 0.0; z <= kWallTop; z += kPointStep) {
      points.emplace_back(x, kWallY, z);
      segments.push_back(kWallSegment);
    }
  }
  // The top and the front (low y) face of a box.
  const auto add_box = [&](const Vec3& lower, const Vec3& upper,
                           int segment) {
    for (double x = lower.x(); x <= upper.x(); x += kPointStep) {
      for (double y = lower.y(); y <= upper.y(); y += kPointStep) {
        points.emplace_back(x, y, upper.z());
        segments.push_back(segment);
      }
      for (double z = lower.z(); z <= upper.z(); z += kPointStep) {
        points.emplace_back(x, lower.y(), z);
        segments.push_back(segment);
      }
    }
  };
  add_box(Vec3(5.0, 4.0, 0.0), Vec3(9.5, 5.8, 1.5), kFirstBoxSegment);
  add_box(Vec3(12.0, 4.5, 0.0), Vec3(16.0, 6.3, 1.4), kSecondBoxSegment);
}

GR_TEST(IdentityAssociatorTest, LinksMasksAcrossViewsThroughGeometry) {
  constexpr int kImageWidth = 320;
  constexpr int kImageHeight = 240;
  constexpr double kFocal = 200.0;  // Pixels.
  constexpr unsigned kSeed = 3;
  // Views every kViewStep m from x = 0 to kLastViewX, 1.2 m up and 1 m
  // before the floor, tilted down by kTilt rad.
  constexpr double kLastViewX = 20.0;
  constexpr double kViewStep = 0.5;
  constexpr double kTilt = 0.25;
  // Ground-truth renders.
  constexpr int kMinSplatRadius = 1;  // Pixels.
  constexpr int kMaxSplatRadius = 3;
  constexpr double kMinRange = 0.1;   // m
  constexpr double kMaxRange = 15.0;  // m
  // Mask ids: kMaskIdStride x a per-view shuffled id; every
  // kOverSegmentPeriod-th view splits the first box at the image middle,
  // its right part getting the spare id plus one.
  constexpr int kMaskIdStride = 10;
  constexpr int kSpareId = kSegments;
  constexpr int kOverSegmentPeriod = 3;
  constexpr int kMinCovered = 100;
  constexpr double kMinDominantShare = 0.95;
  Points points;
  std::vector<int> segments;
  Scene(points, segments);
  const auto camera =
      MakeCamera("pinhole", kImageWidth, kImageHeight,
                 {kFocal, kFocal, kImageWidth / 2.0, kImageHeight / 2.0});
  GR_ASSERT_TRUE(camera != nullptr);
  // Camera at 1 m height looking +y (left of a vehicle driving along x).
  Mat3 world_from_optical;
  world_from_optical << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  IdentityAssociator associator(points, {}, IdentityConfig());
  std::mt19937 rng(kSeed);
  int view = 0;
  for (double x = 0.0; x <= kLastViewX; x += kViewStep, ++view) {
    const CameraExposure exposure(
        Pose(So3Exp(Vec3(-kTilt, 0.0, 0.0)) * world_from_optical,
             Vec3(x, -1.0, 1.2)));
    // A perfect segmenter with arbitrary mask ids per view; every third
    // view over-segments the first box into left and right parts.
    const RenderResult render =
        RenderPoints(*camera, exposure, points, segments, kPointStep,
                     kMinSplatRadius, kMaxSplatRadius, kMinRange, kMaxRange);
    std::vector<int> ids(kSegments + 1);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int32_t> masks(render.label.size(), -1);
    for (size_t pixel = 0; pixel < masks.size(); ++pixel) {
      const int segment = render.label[pixel];
      if (segment == RenderResult::kEmpty) continue;
      masks[pixel] = kMaskIdStride * ids[static_cast<size_t>(segment)];
      if (segment == kFirstBoxSegment && view % kOverSegmentPeriod == 0 &&
          pixel % kImageWidth >= kImageWidth / 2) {
        masks[pixel] = kMaskIdStride * ids[kSpareId] + 1;
      }
    }
    associator.AddView(*camera, exposure, masks.data());
  }
  const IdentityAssociator::Result result = associator.Finish();
  GR_EXPECT_EQ(associator.num_views(), view);
  GR_EXPECT_FALSE(result.masks.empty());
  // Every segment ends with one dominant identity, all different.
  std::map<int, int> dominant;
  for (int segment = 0; segment < kSegments; ++segment) {
    std::map<int, int> counts;
    int covered = 0;
    for (size_t point = 0; point < points.size(); ++point) {
      if (segments[point] != segment ||
          result.point_identity[point] == kNoIdentity) {
        continue;
      }
      ++counts[result.point_identity[point]];
      ++covered;
    }
    GR_ASSERT_TRUE(covered > kMinCovered);
    int best = kNoIdentity;
    int best_count = 0;
    for (const auto& [identity, count] : counts) {
      if (count > best_count) {
        best = identity;
        best_count = count;
      }
    }
    GR_EXPECT_GT(static_cast<double>(best_count) / covered,
                 kMinDominantShare);
    dominant[best] = segment;
  }
  GR_EXPECT_EQ(dominant.size(), static_cast<size_t>(kSegments));
}

GR_TEST(IdentityAssociatorTest, NormalsCullTheOtherFaceOfASlab) {
  // A ceiling (normal down) at z = 3 and the floor of the level above on the
  // same 0.3 m slab (normal up). Seen from below, only the ceiling is visible.
  constexpr double kCeilingZ = 3.0;        // m
  constexpr double kSlabThickness = 0.3;   // m
  constexpr double kHalfSide = 3.0;        // m
  constexpr double kSlabStep = 0.1;        // m
  constexpr int kImageWidth = 160;
  constexpr int kImageHeight = 120;
  constexpr double kFocal = 80.0;          // Pixels.
  constexpr double kLooseDepthTol = 0.6;   // m: lets the upper floor through.
  constexpr double kViewHalfSpan = 1.0;    // m
  constexpr double kViewStep = 0.5;        // m
  constexpr double kViewZ = 1.0;           // m
  constexpr int kCeilingMask = 0;
  constexpr int kMinCeilingPoints = 500;
  Points points;
  Points normals;
  for (double x = -kHalfSide; x <= kHalfSide; x += kSlabStep) {
    for (double y = -kHalfSide; y <= kHalfSide; y += kSlabStep) {
      points.emplace_back(x, y, kCeilingZ);
      normals.push_back(-Vec3::UnitZ());
      points.emplace_back(x, y, kCeilingZ + kSlabThickness);
      normals.push_back(Vec3::UnitZ());
    }
  }
  const auto camera =
      MakeCamera("pinhole", kImageWidth, kImageHeight,
                 {kFocal, kFocal, kImageWidth / 2.0, kImageHeight / 2.0});
  GR_ASSERT_TRUE(camera != nullptr);
  IdentityConfig config;
  config.depth_abs_tol = kLooseDepthTol;
  for (const bool with_normals : {false, true}) {
    IdentityAssociator associator(points, with_normals ? normals : Points(),
                                  config);
    for (double x = -kViewHalfSpan; x <= kViewHalfSpan; x += kViewStep) {
      // Identity rotation: the optical axis (camera z) points up.
      const CameraExposure exposure(
          Pose(Mat3::Identity(), Vec3(x, 0, kViewZ)));
      const std::vector<int32_t> masks(kImageWidth * kImageHeight,
                                       kCeilingMask);
      associator.AddView(*camera, exposure, masks.data());
    }
    const IdentityAssociator::Result result = associator.Finish();
    int upper = 0;
    int ceiling = 0;
    for (size_t point = 0; point < points.size(); ++point) {
      if (result.point_identity[point] == kNoIdentity) continue;
      (points[point].z() > kCeilingZ + kSlabThickness / 2.0 ? upper
                                                            : ceiling) += 1;
    }
    GR_EXPECT_GT(ceiling, kMinCeilingPoints);
    if (with_normals) {
      GR_EXPECT_EQ(upper, 0);
    } else {
      GR_EXPECT_GT(upper, 0);  // Without normals the slab leaks through.
    }
  }
}

}  // namespace
}  // namespace gr
