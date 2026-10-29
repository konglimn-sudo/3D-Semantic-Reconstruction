#include "gr/semantics/labeling.h"

#include <cmath>
#include <memory>
#include <vector>

#include "gr/sensors/exposure.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr int kFloorLabel = 1;
constexpr int kWallLabel = 4;
constexpr int kCarLabel = 12;
constexpr double kPointStep = 0.05;  // m between map points.
// Rounding slack of the floor grid's sample count.
constexpr double kCountSlack = 1e-6;
// A 320 x 240 pinhole camera with a 160 px focal length.
constexpr int kImageWidth = 320;
constexpr int kImageHeight = 240;
constexpr double kFocal = 160.0;

// Floor grid (kFloorLabel) on z = 0 over [x_min, x_max] x [y_min, y_max].
void AddFloor(double x_min, double x_max, double y_min, double y_max,
              double step, Points& points, std::vector<int>& labels) {
  const int steps_x = static_cast<int>((x_max - x_min) / step + kCountSlack);
  const int steps_y = static_cast<int>((y_max - y_min) / step + kCountSlack);
  for (int i = 0; i <= steps_x; ++i) {
    for (int j = 0; j <= steps_y; ++j) {
      points.emplace_back(x_min + step * i, y_min + step * j, 0.0);
      labels.push_back(kFloorLabel);
    }
  }
}

std::shared_ptr<CameraModel> TestCamera() {
  return MakeCamera("pinhole", kImageWidth, kImageHeight,
                    {kFocal, kFocal, kImageWidth / 2.0, kImageHeight / 2.0});
}

// Forward-looking camera 1.5 m above the floor, pitched 20 deg down.
Pose WorldFromCamera() {
  constexpr double kPitch = 20.0;         // deg
  constexpr double kCameraHeight = 1.5;   // m
  // OpenCV axes for a camera looking along world +x.
  const Mat3 forward = (Mat3() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished();
  const Mat3 pitch =
      Eigen::AngleAxisd(DegToRad(kPitch), Vec3::UnitY()).toRotationMatrix();
  return Pose(pitch * forward, Vec3(0, 0, kCameraHeight));
}

ObjectBox CarBox(const Vec3& center) {
  constexpr int kTrack = 7;
  ObjectBox box;
  box.track_id = kTrack;
  box.label = kCarLabel;
  box.center = center;
  box.size = Vec3(4.6, 1.9, 1.5);
  return box;
}

GR_TEST(FrameLabelerTest, ProjectsVisibleSlotsAndDropsOccludedOnes) {
  // A wall at x = 6 hides the right-hand side beyond it: kWallColumns
  // columns of kWallRows points from y = -6.
  constexpr double kWallX = 6.0;       // m
  constexpr double kWallStartY = -6.0;  // m
  constexpr int kWallColumns = 120;
  constexpr int kWallRows = 50;
  constexpr double kMinVisible = 0.6;
  constexpr double kMinEntranceVisible = 0.99;
  constexpr size_t kMinFloorPixels = 10000;
  constexpr size_t kMinWallPixels = 1000;
  constexpr double kMinBoxVisible = 0.2;
  constexpr uint8_t kGrey = 50;
  Points points;
  std::vector<int> labels;
  AddFloor(0, 14, -6, 6, kPointStep, points, labels);
  for (int column = 0; column <= kWallColumns; ++column) {
    for (int row = 1; row <= kWallRows; ++row) {
      points.emplace_back(kWallX, kWallStartY + kPointStep * column,
                          kPointStep * row);
      labels.push_back(kWallLabel);
    }
  }
  std::vector<SlotGeometry> slots(2);
  slots[0].corners = {Vec2(4, 1.0), Vec2(4, 3.5), Vec2(9, 3.5), Vec2(9, 1.0)};
  slots[1].corners = {Vec2(8, -4.0), Vec2(8, -1.5), Vec2(12, -1.5),
                      Vec2(12, -4.0)};  // Behind the wall.
  std::vector<MarkingGeometry> markings(1);
  markings[0].p0 = Vec2(4, 1.0);
  markings[0].p1 = Vec2(9, 1.0);
  const FrameLabeler labeler(points, labels, slots, markings,
                             FrameLabelConfig());
  const auto camera = TestCamera();
  GR_ASSERT_TRUE(camera != nullptr);
  const CameraExposure exposure(WorldFromCamera());
  const FrameLabels frame = labeler.Label(*camera, exposure, {});
  GR_ASSERT_TRUE(frame.slots.size() == 1);
  GR_EXPECT_EQ(frame.slots[0].index, 0);
  // The far edge is seen at a grazing angle, where nearer floor splats win
  // the depth test.
  GR_EXPECT_GT(frame.slots[0].visible_fraction, kMinVisible);
  GR_EXPECT_GT(frame.slots[0].entrance_visible, kMinEntranceVisible);
  GR_EXPECT_TRUE(frame.slots[0].corner_visible[0]);
  GR_EXPECT_FALSE(frame.slots[0].polylines.empty());
  GR_EXPECT_EQ(frame.markings.size(), 1u);
  GR_EXPECT_TRUE(std::isnan(frame.consistency_score));  // No image given.
  size_t floor_pixels = 0;
  size_t wall_pixels = 0;
  for (uint8_t label : frame.label_image) {
    floor_pixels += label == kFloorLabel;
    wall_pixels += label == kWallLabel;
  }
  GR_EXPECT_GT(floor_pixels, kMinFloorPixels);
  GR_EXPECT_GT(wall_pixels, kMinWallPixels);

  // A box in front of the camera occludes the slot and is projected.
  const FrameLabels occluded =
      labeler.Label(*camera, exposure, {CarBox(Vec3(6.0, 2.2, 0.75))});
  GR_ASSERT_TRUE(occluded.boxes.size() == 1);
  GR_EXPECT_GT(occluded.boxes[0].visible_fraction, kMinBoxVisible);
  GR_EXPECT_TRUE(occluded.slots.empty() ||
                 occluded.slots[0].visible_fraction <
                     frame.slots[0].visible_fraction);

  // With an image the consistency score is defined.
  const std::vector<uint8_t> image(kImageWidth * kImageHeight * kRgbChannels,
                                   kGrey);
  const FrameLabels with_image =
      labeler.Label(*camera, exposure, {},
                    ImageView{image.data(), kImageHeight, kImageWidth,
                              kRgbChannels});
  GR_EXPECT_FALSE(std::isnan(with_image.consistency_score));
}

GR_TEST(EdgeStrengthTest, PeaksOnAnIntensityStep) {
  // A dark left half and a bright right half (from column kStepColumn).
  constexpr int kSize = 20;
  constexpr int kStepColumn = 10;
  constexpr uint8_t kBright = 200;
  constexpr int kFlatColumn = 2;
  constexpr double kPeakTolerance = 1e-6;  // float32 strengths.
  constexpr double kFlatTolerance = 1e-9;
  std::vector<uint8_t> image(kSize * kSize, 0);
  for (int y = 0; y < kSize; ++y) {
    for (int x = kStepColumn; x < kSize; ++x) {
      image[static_cast<size_t>(y * kSize + x)] = kBright;
    }
  }
  const std::vector<float> edges =
      EdgeStrength(ImageView{image.data(), kSize, kSize, 1});
  for (int y = 0; y < kSize; ++y) {
    GR_EXPECT_NEAR(edges[static_cast<size_t>(y * kSize + kStepColumn - 1)],
                   1.0, kPeakTolerance);
    GR_EXPECT_NEAR(edges[static_cast<size_t>(y * kSize + kStepColumn)], 1.0,
                   kPeakTolerance);
    GR_EXPECT_NEAR(edges[static_cast<size_t>(y * kSize + kFlatColumn)], 0.0,
                   kFlatTolerance);
  }
}

GR_TEST(LabelLifterTest, RecoversRenderedLabels) {
  constexpr int kRightLabel = 1;  // y < 0.
  constexpr int kLeftLabel = 2;
  constexpr int kClasses = 3;
  constexpr double kMaxRange = 15.0;  // m
  constexpr int kBoundary = 1;        // Pixels.
  constexpr int kMinViews = 2;
  constexpr int kMinSplatRadius = 1;  // Pixels.
  constexpr int kMaxSplatRadius = 3;
  constexpr double kMinRange = 0.1;  // m
  // Lifted points: at least a third of them.
  constexpr size_t kMinLiftedShare = 3;
  Points points;
  std::vector<int> truth;
  AddFloor(2, 10, -4, 4, kPointStep, points, truth);
  for (size_t point = 0; point < points.size(); ++point) {
    truth[point] = points[point].y() < 0 ? kRightLabel : kLeftLabel;
  }
  const auto camera = TestCamera();
  GR_ASSERT_TRUE(camera != nullptr);
  LabelLifter lifter(points, kClasses, kPointStep, kMaxRange, kBoundary,
                     kMinViews);
  for (double shift_y : {-0.5, 0.0, 0.5}) {
    Pose world_from_camera = WorldFromCamera();
    world_from_camera.translation.y() += shift_y;
    const CameraExposure exposure(world_from_camera);
    // A perfect 2D model: the rendered ground-truth labels.
    const RenderResult render =
        RenderPoints(*camera, exposure, points, truth, kPointStep,
                     kMinSplatRadius, kMaxSplatRadius, kMinRange, kMaxRange);
    lifter.AddView(*camera, exposure, render.label.data());
  }
  const LabelLifter::Lifted lifted = lifter.Lift();
  size_t num_lifted = 0;
  size_t wrong = 0;
  for (size_t point = 0; point < points.size(); ++point) {
    if (lifted.labels[point] == LabelLifter::kNoLabel) continue;
    ++num_lifted;
    wrong += lifted.labels[point] != truth[point];
  }
  GR_EXPECT_GT(num_lifted, points.size() / kMinLiftedShare);
  GR_EXPECT_EQ(wrong, 0u);
}

}  // namespace
}  // namespace gr
