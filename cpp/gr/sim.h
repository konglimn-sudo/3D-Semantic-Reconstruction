// Ray caster against a scene of oriented boxes. Used by the synthetic garage
// generator (LiDAR sweeps and fisheye / pinhole renders) so that the whole
// pipeline can be exercised and evaluated without real data.

#ifndef GARAGE_RECON_GR_SIM_H_
#define GARAGE_RECON_GR_SIM_H_

#include <vector>

#include "gr/types.h"

namespace gr {

struct RaycastHits {
  Eigen::VectorXd distance;  // +inf for misses
  Eigen::VectorXi box_ids;   // -1 for misses
  RowMatX3d normals;         // World-frame face normals.
};

class BoxScene {
 public:
  BoxScene() = default;
  BoxScene(const BoxScene&) = default;
  BoxScene& operator=(const BoxScene&) = default;
  BoxScene(BoxScene&&) = default;
  BoxScene& operator=(BoxScene&&) = default;

  // `size` is the full extent along the box axes; world = R * local + center.
  // Returns the box id.
  int AddBox(const Vec3& center, const Vec3& size, const Mat3& rotation);
  void Clear() { boxes_.clear(); }
  size_t size() const { return boxes_.size(); }

  // origins: N x 3, or 1 x 3 shared by all rays; directions: N x 3 unit
  // vectors.
  RaycastHits Raycast(const RowMatX3d& origins, const RowMatX3d& directions,
                      double max_distance) const;

 private:
  struct Box {
    Vec3 center;
    Vec3 half_size;
    Mat3 rotation;
    Mat3 rotation_t;  // Transpose of `rotation`.
    double radius;    // Bounding sphere.
  };

  std::vector<Box> boxes_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_SIM_H_
