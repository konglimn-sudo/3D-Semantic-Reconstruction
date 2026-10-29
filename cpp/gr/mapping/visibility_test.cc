#include "gr/mapping/visibility.h"

#include <algorithm>
#include <cmath>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(VisibilityVoterTest, IgnoresGrazingRaysAndFlagsAnAbsentObject) {
  // Map: floor (z = 0, grazed by every ray) and a box face at x = kObjectX
  // that is absent from the sweep, which instead sees the wall at kWallX.
  constexpr int kFloorRows = 120;       // Along x.
  constexpr int kFloorHalfColumns = 30;  // Along y.
  constexpr double kFloorStep = 0.1;    // m
  constexpr double kObjectX = 6.0;      // m
  constexpr int kObjectHalfColumns = 10;
  constexpr int kObjectFirstRow = 2;
  constexpr int kObjectRows = 15;
  constexpr double kObjectColumnStep = 0.05;  // m
  constexpr double kObjectRowStep = 0.1;      // m
  constexpr double kSensorHeight = 1.9;       // m
  constexpr double kWallX = 12.0;             // m
  constexpr double kCeilingAbove = 1.4;       // m above the sensor (3.3 m).
  constexpr int kHalfAzimuth = 40;            // deg
  constexpr int kLowestElevation = -25;       // deg
  constexpr int kHighestElevation = 15;       // deg
  constexpr double kBucket = 4.0;             // m
  constexpr int kUnknownSession = 5;
  Points map;
  Points normals;
  Points sweep;
  for (int row = 0; row < kFloorRows; ++row) {
    for (int column = -kFloorHalfColumns; column <= kFloorHalfColumns;
         ++column) {
      map.emplace_back(kFloorStep * row, kFloorStep * column, 0.0);
      normals.emplace_back(0, 0, 1);
    }
  }
  const size_t first_object = map.size();
  for (int column = -kObjectHalfColumns; column <= kObjectHalfColumns;
       ++column) {
    for (int row = kObjectFirstRow; row < kObjectRows; ++row) {
      map.emplace_back(kObjectX, kObjectColumnStep * column,
                       kObjectRowStep * row);
      normals.emplace_back(-1, 0, 0);
    }
  }
  const Vec3 origin(0, 0, kSensorHeight);
  for (int azimuth_deg = -kHalfAzimuth; azimuth_deg <= kHalfAzimuth;
       ++azimuth_deg) {
    for (int elevation_deg = kLowestElevation;
         elevation_deg <= kHighestElevation; ++elevation_deg) {
      const double azimuth = DegToRad(azimuth_deg);
      const double elevation = DegToRad(elevation_deg);
      const Vec3 direction(std::cos(elevation) * std::cos(azimuth),
                           std::cos(elevation) * std::sin(azimuth),
                           std::sin(elevation));
      double range = kWallX / direction.x();
      if (direction.z() < 0) {
        range = std::min(range, -kSensorHeight / direction.z());  // Floor.
      }
      if (direction.z() > 0) {
        range = std::min(range, kCeilingAbove / direction.z());  // Ceiling.
      }
      sweep.push_back(direction * range);
    }
  }
  VisibilityVoter voter(map, normals, kBucket);
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
  // Most of the absent object is seen through, none of the floor.
  GR_EXPECT_GT(object_through, static_cast<int>(map.size() - first_object) / 2);
  GR_EXPECT_EQ(floor_through, 0);
  GR_EXPECT_TRUE((voter.Sessions() == std::vector<int>{0}));
  GR_EXPECT_EQ(voter.ThroughCounts(kUnknownSession)[first_object], 0);
}

}  // namespace
}  // namespace gr
