#include "gr/mapping/change_detection.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Points every 0.05 m on an axis-aligned box surface's faces.
void AddBox(const Vec3& lo, const Vec3& hi, Points* points) {
  for (double x = lo.x(); x <= hi.x() + 1e-9; x += 0.05) {
    for (double y = lo.y(); y <= hi.y() + 1e-9; y += 0.05) {
      for (double z = lo.z(); z <= hi.z() + 1e-9; z += 0.05) {
        const int on = (std::abs(x - lo.x()) < 1e-6) +
                       (std::abs(x - hi.x()) < 1e-6) +
                       (std::abs(y - lo.y()) < 1e-6) +
                       (std::abs(y - hi.y()) < 1e-6) +
                       (std::abs(z - lo.z()) < 1e-6) +
                       (std::abs(z - hi.z()) < 1e-6);
        if (on) points->emplace_back(x, y, z);
      }
    }
  }
}

GR_TEST(ChangeDetectionTest, FindsAddedAndRemovedStructure) {
  // Map: a floor 20 x 10 m and a pillar at (15, 5); the new trip sees a
  // fence at x = 5 (added), no pillar (removed), a parked car (occupancy)
  // and floor where the map had none (first seen: flat, ignored).
  Points floor;
  for (double x = 0.0; x <= 20.0; x += 0.1) {
    for (double y = 0.0; y <= 10.0; y += 0.1) {
      if (x > 9.0 && x < 11.0 && y > 7.0) continue;  // Unseen in the map.
      floor.emplace_back(x, y, 0.0);
    }
  }
  Points map = floor;
  AddBox(Vec3(14.8, 4.8, 0.0), Vec3(15.2, 5.2, 2.5), &map);
  Points now;
  for (double x = 0.0; x <= 20.0; x += 0.1) {
    for (double y = 0.0; y <= 10.0; y += 0.1) now.emplace_back(x, y, 0.0);
  }
  AddBox(Vec3(5.0, 2.0, 0.0), Vec3(5.2, 5.0, 1.0), &now);
  AddBox(Vec3(2.0, 7.0, 0.0), Vec3(6.6, 8.8, 1.5), &now);
  ChangeConfig config;
  ChangeDetector detector(map, config);
  double stamp = 0.0;
  for (double x : {1.0, 9.0, 12.0, 18.0}) {
    detector.AddSweep(now, Vec3(x, 1.0, 1.9), stamp);
    stamp += 2.0;
  }
  const std::vector<ChangeCluster> clusters = detector.Clusters();
  int fences = 0;
  int cars = 0;
  int removed = 0;
  for (const ChangeCluster& c : clusters) {
    if (!c.added) {
      ++removed;
      GR_EXPECT_NEAR(c.center.x(), 15.0, 0.3);
    } else if (c.vehicle_like) {
      ++cars;
    } else {
      ++fences;
      GR_EXPECT_NEAR(c.center.x(), 5.1, 0.3);
      GR_EXPECT_NEAR(c.center.y(), 3.5, 0.3);
    }
  }
  GR_EXPECT_EQ(fences, 1);
  GR_EXPECT_EQ(cars, 1);
  GR_EXPECT_EQ(removed, 1);
}

}  // namespace
}  // namespace gr
