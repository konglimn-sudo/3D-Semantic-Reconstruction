#include "gr/mapping/change_detection.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Point spacing (m) of the test surfaces and the slack of their loops.
constexpr double kSurfaceStep = 0.05;
constexpr double kLoopSlack = 1e-9;
constexpr double kOnFaceTolerance = 1e-6;

// Points every kSurfaceStep on the faces of an axis-aligned box.
void AddBox(const Vec3& lo, const Vec3& hi, Points* points) {
  const auto on_face = [](double value, double bound) {
    return std::abs(value - bound) < kOnFaceTolerance;
  };
  for (double x = lo.x(); x <= hi.x() + kLoopSlack; x += kSurfaceStep) {
    for (double y = lo.y(); y <= hi.y() + kLoopSlack; y += kSurfaceStep) {
      for (double z = lo.z(); z <= hi.z() + kLoopSlack; z += kSurfaceStep) {
        if (on_face(x, lo.x()) || on_face(x, hi.x()) || on_face(y, lo.y()) ||
            on_face(y, hi.y()) || on_face(z, lo.z()) || on_face(z, hi.z())) {
          points->emplace_back(x, y, z);
        }
      }
    }
  }
}

GR_TEST(ChangeDetectionTest, FindsAddedAndRemovedStructure) {
  // Map: a floor kFloorLength x kFloorWidth and a pillar; the new trip sees
  // a fence (added), no pillar (removed), a parked car (occupancy) and floor
  // where the map had none (first seen: flat, ignored).
  constexpr double kFloorLength = 20.0;  // m
  constexpr double kFloorWidth = 10.0;   // m
  constexpr double kFloorStep = 0.1;     // m
  constexpr double kSensorY = 1.0;       // m
  constexpr double kSensorHeight = 1.9;  // m
  constexpr double kSweepInterval = 2.0;  // s
  constexpr double kPillarX = 15.0;      // m
  constexpr double kFenceX = 5.1;        // m, centre
  constexpr double kFenceY = 3.5;        // m, centre
  constexpr double kCentreTolerance = 0.3;  // m
  Points floor;
  for (double x = 0.0; x <= kFloorLength; x += kFloorStep) {
    for (double y = 0.0; y <= kFloorWidth; y += kFloorStep) {
      if (x > 9.0 && x < 11.0 && y > 7.0) continue;  // Unseen in the map.
      floor.emplace_back(x, y, 0.0);
    }
  }
  Points map = floor;
  AddBox(Vec3(14.8, 4.8, 0.0), Vec3(15.2, 5.2, 2.5), &map);  // Pillar.
  Points now;
  for (double x = 0.0; x <= kFloorLength; x += kFloorStep) {
    for (double y = 0.0; y <= kFloorWidth; y += kFloorStep) {
      now.emplace_back(x, y, 0.0);
    }
  }
  AddBox(Vec3(5.0, 2.0, 0.0), Vec3(5.2, 5.0, 1.0), &now);  // Fence.
  AddBox(Vec3(2.0, 7.0, 0.0), Vec3(6.6, 8.8, 1.5), &now);  // Car.
  ChangeConfig config;
  ChangeDetector detector(map, config);
  double stamp = 0.0;
  for (double sensor_x : {1.0, 9.0, 12.0, 18.0}) {
    detector.AddSweep(now, Vec3(sensor_x, kSensorY, kSensorHeight), stamp);
    stamp += kSweepInterval;
  }
  const std::vector<ChangeCluster> clusters = detector.Clusters();
  int fences = 0;
  int cars = 0;
  int removed = 0;
  for (const ChangeCluster& cluster : clusters) {
    if (!cluster.added) {
      ++removed;
      GR_EXPECT_NEAR(cluster.center.x(), kPillarX, kCentreTolerance);
    } else if (cluster.vehicle_like) {
      ++cars;
    } else {
      ++fences;
      GR_EXPECT_NEAR(cluster.center.x(), kFenceX, kCentreTolerance);
      GR_EXPECT_NEAR(cluster.center.y(), kFenceY, kCentreTolerance);
    }
  }
  GR_EXPECT_EQ(fences, 1);
  GR_EXPECT_EQ(cars, 1);
  GR_EXPECT_EQ(removed, 1);
}

}  // namespace
}  // namespace gr
