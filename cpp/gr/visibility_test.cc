#include "gr/visibility.h"

#include <algorithm>
#include <cmath>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(VisibilityVoterTest, IgnoresGrazingRaysAndFlagsAnAbsentObject) {
  // Map: floor (z = 0, grazed by every ray) and a box face at x = 6 that is
  // absent from the sweep, which instead sees the wall at x = 12.
  Points map;
  Points normals;
  Points sweep;
  for (int i = 0; i < 120; ++i) {
    for (int j = -30; j <= 30; ++j) {
      map.emplace_back(0.1 * i, 0.1 * j, 0.0);
      normals.emplace_back(0, 0, 1);
    }
  }
  const size_t first_object = map.size();
  for (int j = -10; j <= 10; ++j) {
    for (int k = 2; k < 15; ++k) {
      map.emplace_back(6.0, 0.05 * j, 0.1 * k);
      normals.emplace_back(-1, 0, 0);
    }
  }
  const Vec3 origin(0, 0, 1.9);
  for (int a = -40; a <= 40; ++a) {
    for (int e = -25; e <= 15; ++e) {
      const double azimuth = DegToRad(a);
      const double elevation = DegToRad(e);
      const Vec3 d(std::cos(elevation) * std::cos(azimuth),
                   std::cos(elevation) * std::sin(azimuth),
                   std::sin(elevation));
      double t = 12.0 / d.x();                       // Wall at x = 12.
      if (d.z() < 0) t = std::min(t, -1.9 / d.z());  // Floor.
      if (d.z() > 0) t = std::min(t, 1.4 / d.z());   // Ceiling at 3.3 m.
      sweep.push_back(d * t);
    }
  }
  VisibilityVoter voter(map, normals, 4.0);
  voter.AddSweep(Pose(Mat3::Identity(), origin), sweep,
                 VisibilityVoteOptions());
  int object_through = 0;
  int floor_through = 0;
  for (size_t i = 0; i < map.size(); ++i) {
    const bool through = voter.ThroughCounts()[i] > 0;
    if (i >= first_object) {
      object_through += through;
    } else {
      floor_through += through;
    }
  }
  GR_EXPECT_GT(object_through, static_cast<int>(map.size() - first_object) / 2);
  GR_EXPECT_EQ(floor_through, 0);
  GR_EXPECT_TRUE((voter.Sessions() == std::vector<int>{0}));
  GR_EXPECT_EQ(voter.ThroughCounts(5)[first_object], 0);  // Unknown session.
}

}  // namespace
}  // namespace gr
