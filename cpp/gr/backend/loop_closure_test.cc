#include "gr/backend/loop_closure.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kSweepVoxel = 0.2;  // m

Pose PoseXyYaw(double x, double y, double yaw) {
  return Pose(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
              Vec3(x, y, 0));
}

GR_TEST(LoopClosureTest, PcmRejectsTheLoopThatContradictsOdometry) {
  // A straight drive of 40 keyframes 1 m apart. Keyframes 30.. revisit
  // 2.. with the true relative pose; a fifth loop 36 -> 8 is off by 3 m, 1 m
  // and 0.2 rad.
  constexpr int kKeyframes = 40;
  constexpr int kConsistentLoops = 4;
  constexpr int kFirstSource = 30;
  constexpr int kFirstTarget = 2;
  constexpr int kBadSource = 36;
  constexpr int kBadTarget = 8;
  KeyframeSet keyframes({kSweepVoxel});
  std::vector<Pose> poses;
  for (int index = 0; index < kKeyframes; ++index) {
    Keyframe keyframe;
    keyframe.odom = PoseXyYaw(index, 0, 0);
    keyframes.Add(keyframe, {});
    poses.push_back(keyframe.odom);
  }
  std::vector<LoopConstraint> loops;
  for (int offset = 0; offset < kConsistentLoops; ++offset) {
    LoopConstraint loop;
    loop.i = kFirstSource + offset;
    loop.j = kFirstTarget + offset;
    loop.j_from_i = poses[static_cast<size_t>(loop.j)].Inverse() *
                    poses[static_cast<size_t>(loop.i)];
    loops.push_back(loop);
  }
  LoopConstraint bad;
  bad.i = kBadSource;
  bad.j = kBadTarget;
  bad.j_from_i = poses[kBadTarget].Inverse() * poses[kBadSource] *
                 PoseXyYaw(3.0, 1.0, 0.2);
  loops.insert(loops.begin() + kConsistentLoops / 2, bad);
  const std::vector<LoopConstraint> kept =
      SelectConsistentLoops(loops, keyframes, poses, LoopConfig());
  GR_EXPECT_EQ(kept.size(), static_cast<size_t>(kConsistentLoops));
  for (const LoopConstraint& loop : kept) GR_EXPECT_TRUE(loop.i != kBadSource);
}

GR_TEST(LoopClosureTest, DisabledOrTinySetsHaveNoLoops) {
  KeyframeSet keyframes({kSweepVoxel});
  LoopConfig config;
  LoopStats stats;
  GR_EXPECT_TRUE(DetectLoops(keyframes, {}, config, &stats).empty());
  Keyframe keyframe;
  keyframes.Add(keyframe, {Vec3(1, 0, 0)});
  keyframes.Add(keyframe, {Vec3(1, 0, 0)});
  config.enabled = false;
  GR_EXPECT_TRUE(
      DetectLoops(keyframes, keyframes.OdomPoses(), config, &stats).empty());
  GR_EXPECT_EQ(stats.tested, 0);
}

}  // namespace
}  // namespace gr
