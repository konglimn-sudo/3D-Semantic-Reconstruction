#include "gr/semantics/labeling.h"

#include <cmath>
#include <vector>

#include "gr/sensors/exposure.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

// Floor grid (label 1) on z = 0 over [x0, x1] x [y0, y1].
void AddFloor(double x0, double x1, double y0, double y1, double step,
              Points& points, std::vector<int>& labels) {
  const int nx = static_cast<int>((x1 - x0) / step + 1e-6);
  const int ny = static_cast<int>((y1 - y0) / step + 1e-6);
  for (int i = 0; i <= nx; ++i) {
    for (int j = 0; j <= ny; ++j) {
      points.emplace_back(x0 + step * i, y0 + step * j, 0.0);
      labels.push_back(1);
    }
  }
}

// Forward-looking camera 1.5 m above the floor, pitched 20 deg down.
Pose WorldFromCamera() {
  // OpenCV axes for a camera looking along world +x.
  const Mat3 forward = (Mat3() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished();
  const Mat3 pitch =
      Eigen::AngleAxisd(DegToRad(20.0), Vec3::UnitY()).toRotationMatrix();
  return Pose(pitch * forward, Vec3(0, 0, 1.5));
}

ObjectBox CarBox(const Vec3& center) {
  ObjectBox box;
  box.track_id = 7;
  box.label = 12;
  box.center = center;
  box.size = Vec3(4.6, 1.9, 1.5);
  return box;
}

GR_TEST(FrameLabelerTest, ProjectsVisibleSlotsAndDropsOccludedOnes) {
  Points points;
  std::vector<int> labels;
  AddFloor(0, 14, -6, 6, 0.05, points, labels);
  // A wall hiding the right-hand side beyond x = 6.
  for (int j = 0; j <= 120; ++j) {
    for (int k = 1; k <= 50; ++k) {
      points.emplace_back(6.0, -6.0 + 0.05 * j, 0.05 * k);
      labels.push_back(4);
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
  const auto camera = MakeCamera("pinhole", 320, 240, {160, 160, 160, 120});
  GR_ASSERT_TRUE(camera != nullptr);
  const CameraExposure exposure(WorldFromCamera());
  const FrameLabels frame = labeler.Label(*camera, exposure, {});
  GR_ASSERT_TRUE(frame.slots.size() == 1);
  GR_EXPECT_EQ(frame.slots[0].index, 0);
  // The far edge is seen at a grazing angle, where nearer floor splats win
  // the depth test.
  GR_EXPECT_GT(frame.slots[0].visible_fraction, 0.6);
  GR_EXPECT_GT(frame.slots[0].entrance_visible, 0.99);
  GR_EXPECT_TRUE(frame.slots[0].corner_visible[0]);
  GR_EXPECT_FALSE(frame.slots[0].polylines.empty());
  GR_EXPECT_EQ(frame.markings.size(), 1u);
  GR_EXPECT_TRUE(std::isnan(frame.consistency_score));  // No image given.
  size_t floor_pixels = 0;
  size_t wall_pixels = 0;
  for (uint8_t label : frame.label_image) {
    floor_pixels += label == 1;
    wall_pixels += label == 4;
  }
  GR_EXPECT_GT(floor_pixels, 10000u);
  GR_EXPECT_GT(wall_pixels, 1000u);

  // A box in front of the camera occludes the slot and is projected.
  const FrameLabels occluded =
      labeler.Label(*camera, exposure, {CarBox(Vec3(6.0, 2.2, 0.75))});
  GR_ASSERT_TRUE(occluded.boxes.size() == 1);
  GR_EXPECT_GT(occluded.boxes[0].visible_fraction, 0.2);
  GR_EXPECT_TRUE(occluded.slots.empty() || occluded.slots[0].visible_fraction <
                                               frame.slots[0].visible_fraction);

  // With an image the consistency score is defined.
  const std::vector<uint8_t> image(320 * 240 * 3, 50);
  const FrameLabels with_image = labeler.Label(
      *camera, exposure, {}, ImageView{image.data(), 240, 320, 3});
  GR_EXPECT_FALSE(std::isnan(with_image.consistency_score));
}

GR_TEST(EdgeStrengthTest, PeaksOnAnIntensityStep) {
  std::vector<uint8_t> image(20 * 20, 0);
  for (int y = 0; y < 20; ++y) {
    for (int x = 10; x < 20; ++x) image[static_cast<size_t>(y * 20 + x)] = 200;
  }
  const std::vector<float> edges =
      EdgeStrength(ImageView{image.data(), 20, 20, 1});
  for (int y = 0; y < 20; ++y) {
    GR_EXPECT_NEAR(edges[static_cast<size_t>(y * 20 + 9)], 1.0, 1e-6);
    GR_EXPECT_NEAR(edges[static_cast<size_t>(y * 20 + 10)], 1.0, 1e-6);
    GR_EXPECT_NEAR(edges[static_cast<size_t>(y * 20 + 2)], 0.0, 1e-9);
  }
}

GR_TEST(LabelLifterTest, RecoversRenderedLabels) {
  Points points;
  std::vector<int> truth;
  AddFloor(2, 10, -4, 4, 0.05, points, truth);
  for (size_t i = 0; i < points.size(); ++i)
    truth[i] = points[i].y() < 0 ? 1 : 2;
  const auto camera = MakeCamera("pinhole", 320, 240, {160, 160, 160, 120});
  GR_ASSERT_TRUE(camera != nullptr);
  LabelLifter lifter(points, 3, 0.05, 15.0, 1, 2);
  for (double dy : {-0.5, 0.0, 0.5}) {
    Pose world_from_camera = WorldFromCamera();
    world_from_camera.translation.y() += dy;
    const CameraExposure exposure(world_from_camera);
    // A perfect 2D model: the rendered ground-truth labels.
    const RenderResult render =
        RenderPoints(*camera, exposure, points, truth, 0.05, 1, 3, 0.1, 15.0);
    lifter.AddView(*camera, exposure, render.label.data());
  }
  const LabelLifter::Lifted lifted = lifter.Lift();
  size_t num_lifted = 0;
  size_t wrong = 0;
  for (size_t i = 0; i < points.size(); ++i) {
    if (lifted.labels[i] < 0) continue;
    ++num_lifted;
    wrong += lifted.labels[i] != truth[i];
  }
  GR_EXPECT_GT(num_lifted, points.size() / 3);
  GR_EXPECT_EQ(wrong, 0u);
}

}  // namespace
}  // namespace gr
