// LiDAR-anchored 3D Gaussians (plan section 3.4): the initial Gaussian set of
// the 3DGS stage, built from every point of the survey cloud, and a CPU
// renderer along LiDAR rays. The renderer gives the LiDAR intensity channel
// and the geometric verification (section 3.5): rendered depth along held-out
// sweeps against the measured ranges.
//
// A Gaussian is evaluated along a ray as in 3DGRT / 3DGUT ("eval3d"): the
// response peaks at t* = argmax_t G(o + t d), with alpha = opacity *
// exp(-0.5 m^2) where m is the Mahalanobis distance of the peak. Hits are
// composited front to back.

#ifndef GARAGE_RECON_GR_SPLAT_GAUSSIANS_H_
#define GARAGE_RECON_GR_SPLAT_GAUSSIANS_H_

#include <cstdint>
#include <vector>

#include "gr/core/types.h"

namespace gr {

// Bits of GaussianSet::flags.
enum GaussianFlag : uint8_t {
  kGaussianAnchored = 1,    // Initialised from a LiDAR point (source L).
  kGaussianAppearance = 2,  // Enough well-exposed camera views to learn colour.
  // No camera supervision: geometry and opacity stay as initialised, and the
  // Gaussian is exempt from opacity resets and pruning.
  kGaussianFrozen = 4,
  kGaussianTripSpecific = 8,  // Changed between trips: only in its trip.
  kGaussianPrior = 16,        // From a depth prior (DA3), source P.
};

struct GaussianSet {
  // Trip of the Gaussians shared by every trip.
  static constexpr int kSharedTrip = -1;

  Points means;
  std::vector<Eigen::Quaterniond> rotations;  // World <- Gaussian axes.
  std::vector<Vec3> scales;  // Standard deviation along each axis (m).
  std::vector<float> opacity;
  std::vector<float> intensity;  // LiDAR intensity in [0, 1].
  std::vector<int> trip;  // Trip of a trip-specific Gaussian, or kSharedTrip.

  size_t size() const { return means.size(); }
  // Resizes every attribute to the size of `means`, with defaults.
  void FillDefaults();
};

struct GaussianInitConfig {
  int neighbors = 6;            // For the local point spacing.
  double search_radius = 0.25;  // Neighbour search (m).
  double default_spacing = 0.05;  // Without enough neighbours (m).
  // Tangential sigma = factor * spacing; 0.8 keeps a 5 cm grid of 0.9-opaque
  // discs at least 99.8 % opaque between the points (0.6: 97 %).
  double tangent_factor = 0.8;
  double min_tangent = 0.005;     // m.
  double max_tangent = 0.25;      // m.
  double normal_floor = 0.001;    // Minimum sigma along the normal (m).
  double opacity = 0.9;
};

// One flat Gaussian per point: the third axis along the normal with the
// point's sigma (at least normal_floor), the other two set by the local point
// spacing. Points without a normal (zero vector) get an isotropic Gaussian.
// `sigma` may be empty. Intensity and trip are left at their defaults.
GaussianSet InitGaussians(const Points& points, const Points& normals,
                          const std::vector<double>& sigma,
                          const GaussianInitConfig& config);

struct RayRenderConfig {
  double cutoff = 3.0;    // Support of a Gaussian in standard deviations.
  double cell = 0.5;      // Grid of the Gaussian bounding boxes (m).
  int64_t max_cells = 64000000;  // The cell grows until the grid fits.
  double min_alpha = 1.0 / 255.0;
  double max_alpha = 0.99;
  double min_transmittance = 1e-3;  // Stop once a ray is this opaque.
  double max_range = 100.0;
};

struct RayRender {
  // Dominant Gaussian of rays without any hit.
  static constexpr int kNoGaussian = -1;
  // `depth` is where the transmittance falls below this.
  static constexpr double kDepthTransmittance = 0.5;

  // Range where the transmittance falls below kDepthTransmittance (NaN if it
  // never does).
  std::vector<double> depth;
  // Alpha-weighted mean range / opacity (NaN without any hit).
  std::vector<double> mean_depth;
  std::vector<float> intensity;  // Alpha-weighted intensity / opacity.
  std::vector<float> opacity;    // 1 - final transmittance.
  // Gaussian with the largest weight, or kNoGaussian.
  std::vector<int> dominant;
};

class GaussianRayRenderer {
 public:
  // Grid cell (m) of a renderer without Gaussians.
  static constexpr double kEmptyCell = 0.5;

  explicit GaussianRayRenderer(GaussianSet gaussians,
                               const RayRenderConfig& config = {});

  // Renders rays (unit `directions`). Trip-specific Gaussians take part only
  // in rays of their own `trip` (GaussianSet::kSharedTrip: shared Gaussians
  // only).
  RayRender Render(const Points& origins, const Points& directions,
                   int trip) const;

  double cell() const { return cell_; }
  const GaussianSet& gaussians() const { return set_; }

 private:
  struct Hit {
    double t;
    double alpha;
    int index;
  };
  // Hits of one ray, sorted by range and without duplicates.
  void CollectHits(const Vec3& origin, const Vec3& direction, int trip,
                   std::vector<Hit>& hits) const;
  bool Evaluate(int index, const Vec3& origin, const Vec3& direction,
                Hit& hit) const;

  GaussianSet set_;
  RayRenderConfig config_;
  std::vector<Eigen::Matrix3f> axes_;  // Rows: Gaussian axes / sigma.
  Vec3 min_bound_ = Vec3::Zero();
  double cell_ = kEmptyCell;
  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;
  double max_extent_ = 0.0;  // Largest half-diagonal of a bounding box (m).
  std::vector<int64_t> offsets_;  // CSR: cell -> range of `entries_`.
  std::vector<int> entries_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_SPLAT_GAUSSIANS_H_
