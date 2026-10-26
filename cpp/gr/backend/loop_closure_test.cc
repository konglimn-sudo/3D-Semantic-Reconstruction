#include "gr/backend/loop_closure.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

Pose PoseXyYaw(double x, double y, double yaw) {
  return Pose(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
              Vec3(x, y, 0));
}

GR_TEST(LoopClosureTest, PcmRejectsTheLoopThatContradictsOdometry) {
  KeyframeSet keyframes({0.2});
  std::vector<Pose> poses;
  for (int k = 0; k < 40; ++k) {
    Keyframe keyframe;
    keyframe.odom = PoseXyYaw(k, 0, 0);
    keyframes.Add(keyframe, {});
    poses.push_back(keyframe.odom);
  }
  std::vector<LoopConstraint> loops;
  // Consistent loops: keyframe i revisits j with the true relative pose.
  for (int a = 0; a < 4; ++a) {
    LoopConstraint loop;
    loop.i = 30 + a;
    loop.j = 2 + a;
    loop.j_from_i = poses[2 + a].Inverse() * poses[30 + a];
    loops.push_back(loop);
  }
  LoopConstraint bad;
  bad.i = 36;
  bad.j = 8;
  bad.j_from_i = poses[8].Inverse() * poses[36] * PoseXyYaw(3.0, 1.0, 0.2);
  loops.insert(loops.begin() + 2, bad);
  const std::vector<LoopConstraint> kept =
      SelectConsistentLoops(loops, keyframes, poses, LoopConfig());
  GR_EXPECT_EQ(kept.size(), 4u);
  for (const LoopConstraint& loop : kept) GR_EXPECT_TRUE(loop.i != 36);
}

GR_TEST(LoopClosureTest, DisabledOrTinySetsHaveNoLoops) {
  KeyframeSet keyframes({0.2});
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
