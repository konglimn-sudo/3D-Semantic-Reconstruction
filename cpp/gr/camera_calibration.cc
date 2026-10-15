#include "gr/camera_calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "Eigen/Dense"

#include "gr/check.h"

namespace gr {
namespace {

using Vec7 = Eigen::Matrix<double, 7, 1>;
using Mat7 = Eigen::Matrix<double, 7, 7>;
using Mat73 = Eigen::Matrix<double, 7, 3>;

constexpr int kMaxRowIterations = 4;

// The calibration moved by x = [time offset, rotation, translation]; the
// rotation and translation act in the camera frame.
CameraCalibration Apply(const CameraCalibration& base, const Vec7& x) {
  CameraCalibration out = base;
  out.time_offset = base.time_offset + x[0];
  out.vehicle_from_camera =
      base.vehicle_from_camera * Pose(So3Exp(x.segment<3>(1)), x.segment<3>(4));
  return out;
}

double RowTime(const CameraCalibration& calibration,
               const CalibrationImage& image, double row) {
  return image.stamp + calibration.time_offset + row * calibration.line_time;
}

struct Landmark {
  Vec3 position = Vec3::Zero();
  std::vector<TrackObservation> observations;
  int surface = -1;  // Nearest LiDAR point, -1 without.
};

// Normalised reprojection residual of one observation for a vehicle pose at
// the observation's row time.
std::optional<Vec2> PixelResidual(const CameraModel& camera,
                                  const Pose& world_from_vehicle,
                                  const Pose& vehicle_from_camera,
                                  const TrackObservation& observation,
                                  const Vec3& point, double sigma) {
  const Pose camera_from_world =
      (world_from_vehicle * vehicle_from_camera).Inverse();
  const std::optional<Vec2> uv = camera.Project(camera_from_world * point);
  if (!uv) return std::nullopt;
  return Vec2((*uv - observation.uv) / sigma);
}

// Point closest to all rays; nullopt for too little parallax.
std::optional<Vec3> Triangulate(const std::vector<Vec3>& centers,
                                const std::vector<Vec3>& directions,
                                double min_parallax_rad) {
  double max_angle = 0.0;
  for (size_t a = 0; a < directions.size(); ++a) {
    for (size_t b = a + 1; b < directions.size(); ++b) {
      const double c = std::clamp(directions[a].dot(directions[b]), -1.0, 1.0);
      max_angle = std::max(max_angle, std::acos(c));
    }
  }
  if (max_angle < min_parallax_rad) return std::nullopt;
  Mat3 a = Mat3::Zero();
  Vec3 b = Vec3::Zero();
  for (size_t k = 0; k < centers.size(); ++k) {
    const Mat3 projector =
        Mat3::Identity() - directions[k] * directions[k].transpose();
    a += projector;
    b += projector * centers[k];
  }
  const Vec3 point = a.ldlt().solve(b);
  if (!point.allFinite()) return std::nullopt;
  for (size_t k = 0; k < centers.size(); ++k) {
    if ((point - centers[k]).dot(directions[k]) <= 0.0) return std::nullopt;
  }
  return point;
}

// The least-squares problem over the calibration correction x and the
// landmark positions.
class Problem {
 public:
  Problem(const CameraModel& camera, const CameraCalibration& initial,
          const std::vector<CalibrationImage>& images,
          const PointCloudIndex* surface, const Points* normals,
          const CameraCalibrationConfig& config)
      : camera_(camera),
        initial_(initial),
        images_(images),
        surface_(surface),
        normals_(normals),
        config_(config) {
    prior_sigma_ << config.time_offset_prior,
        Vec3::Constant(DegToRad(config.rotation_prior_deg)),
        Vec3::Constant(config.translation_prior);
    enabled_ << (config.estimate_time_offset ? 1.0 : 0.0),
        Vec3::Constant(config.estimate_rotation ? 1.0 : 0.0),
        Vec3::Constant(config.estimate_translation ? 1.0 : 0.0);
  }

  // Robust (Huber) cost of a normalised residual with |r|^2 = s.
  double Rho(double s) const {
    const double k = config_.huber;
    return s <= k * k ? s : 2.0 * k * std::sqrt(s) - k * k;
  }
  double Weight(double s) const {
    const double k = config_.huber;
    return s <= k * k ? 1.0 : k / std::sqrt(s);
  }

  Pose VehicleAt(const CameraCalibration& calibration,
                 const TrackObservation& observation) const {
    const CalibrationImage& image =
        images_[static_cast<size_t>(observation.image)];
    return image.trajectory->At(
        RowTime(calibration, image, observation.uv.y()));
  }

  // Nearest LiDAR point within the search radius, -1 if none.
  int NearestSurface(const Vec3& point) const {
    if (surface_ == nullptr) return -1;
    int best = -1;
    double best_d2 = config_.plane_radius * config_.plane_radius;
    for (int i : surface_->RadiusSearch(point, config_.plane_radius)) {
      const double d2 =
          (surface_->points()[static_cast<size_t>(i)] - point).squaredNorm();
      if (d2 < best_d2) {
        best_d2 = d2;
        best = i;
      }
    }
    return best;
  }

  double PlaneResidual(const Landmark& landmark, const Vec3& point) const {
    const size_t s = static_cast<size_t>(landmark.surface);
    return (*normals_)[s].dot(point - surface_->points()[s]) /
           config_.plane_sigma;
  }

  double Cost(const Vec7& x, const std::vector<Landmark>& landmarks,
              const std::vector<Vec3>& positions) const {
    const CameraCalibration calibration = Apply(initial_, x);
    double cost = 0.0;
    for (size_t j = 0; j < landmarks.size(); ++j) {
      for (const TrackObservation& observation : landmarks[j].observations) {
        const std::optional<Vec2> r =
            PixelResidual(camera_, VehicleAt(calibration, observation),
                          calibration.vehicle_from_camera, observation,
                          positions[j], config_.pixel_sigma);
        cost += r ? Rho(r->squaredNorm()) : 2.0 * config_.huber * 10.0;
      }
      if (landmarks[j].surface >= 0) {
        const double r = PlaneResidual(landmarks[j], positions[j]);
        cost += r * r;
      }
    }
    for (int k = 0; k < 7; ++k) {
      if (enabled_[k] == 0.0) continue;
      const double r = x[k] / prior_sigma_[k];
      cost += r * r;
    }
    return cost;
  }

  // Gauss-Newton blocks: calibration (7), landmark (3) and their coupling.
  struct Normal {
    Mat7 h_cc = Mat7::Zero();
    Vec7 b_c = Vec7::Zero();
    std::vector<Mat3> h_ll;
    std::vector<Mat73> h_cl;
    std::vector<Vec3> b_l;
  };

  Normal Linearize(const Vec7& x, const std::vector<Landmark>& landmarks,
                   const std::vector<Vec3>& positions) const {
    const CameraCalibration calibration = Apply(initial_, x);
    Normal normal;
    normal.h_ll.assign(landmarks.size(), Mat3::Zero());
    normal.h_cl.assign(landmarks.size(), Mat73::Zero());
    normal.b_l.assign(landmarks.size(), Vec3::Zero());
    const double time_step = 1e-4;
    const double angle_step = 1e-6;
    const double length_step = 1e-5;
    for (size_t j = 0; j < landmarks.size(); ++j) {
      const Vec3& point = positions[j];
      for (const TrackObservation& observation : landmarks[j].observations) {
        const Pose vehicle = VehicleAt(calibration, observation);
        const std::optional<Vec2> r =
            PixelResidual(camera_, vehicle, calibration.vehicle_from_camera,
                          observation, point, config_.pixel_sigma);
        if (!r) continue;
        Eigen::Matrix<double, 2, 7> j_c = Eigen::Matrix<double, 2, 7>::Zero();
        Eigen::Matrix<double, 2, 3> j_l;
        bool ok = true;
        for (int k = 0; k < 7 && ok; ++k) {
          if (enabled_[k] == 0.0) continue;
          const double h =
              k == 0 ? time_step : (k < 4 ? angle_step : length_step);
          Vec2 sides[2];
          for (int side = 0; side < 2; ++side) {
            Vec7 dx = x;
            dx[k] += side == 0 ? h : -h;
            const CameraCalibration moved = Apply(initial_, dx);
            const Pose moved_vehicle =
                k == 0 ? VehicleAt(moved, observation) : vehicle;
            const std::optional<Vec2> rr =
                PixelResidual(camera_, moved_vehicle, moved.vehicle_from_camera,
                              observation, point, config_.pixel_sigma);
            if (!rr) {
              ok = false;
              break;
            }
            sides[side] = *rr;
          }
          if (ok) j_c.col(k) = (sides[0] - sides[1]) / (2.0 * h);
        }
        for (int a = 0; a < 3 && ok; ++a) {
          Vec2 sides[2];
          for (int side = 0; side < 2; ++side) {
            Vec3 moved = point;
            moved[a] += side == 0 ? length_step : -length_step;
            const std::optional<Vec2> rr =
                PixelResidual(camera_, vehicle, calibration.vehicle_from_camera,
                              observation, moved, config_.pixel_sigma);
            if (!rr) {
              ok = false;
              break;
            }
            sides[side] = *rr;
          }
          if (ok) j_l.col(a) = (sides[0] - sides[1]) / (2.0 * length_step);
        }
        if (!ok) continue;
        const double w = Weight(r->squaredNorm());
        normal.h_cc += w * j_c.transpose() * j_c;
        normal.b_c += w * j_c.transpose() * *r;
        normal.h_ll[j] += w * j_l.transpose() * j_l;
        normal.h_cl[j] += w * j_c.transpose() * j_l;
        normal.b_l[j] += w * j_l.transpose() * *r;
      }
      if (landmarks[j].surface >= 0) {
        const Vec3 n = (*normals_)[static_cast<size_t>(landmarks[j].surface)] /
                       config_.plane_sigma;
        normal.h_ll[j] += n * n.transpose();
        normal.b_l[j] += n * PlaneResidual(landmarks[j], point);
      }
    }
    for (int k = 0; k < 7; ++k) {
      if (enabled_[k] == 0.0) continue;
      const double info = 1.0 / (prior_sigma_[k] * prior_sigma_[k]);
      normal.h_cc(k, k) += info;
      normal.b_c[k] += info * x[k];
    }
    return normal;
  }

  // Damped Schur-complement step; false if a block is singular.
  bool Solve(const Normal& normal, double lambda, Vec7& dx,
             std::vector<Vec3>& dl, Mat7* schur = nullptr) const {
    Mat7 s = normal.h_cc;
    Vec7 g = normal.b_c;
    for (int k = 0; k < 7; ++k) s(k, k) += lambda * std::max(s(k, k), 1e-9);
    std::vector<Mat3> inverse(normal.h_ll.size());
    for (size_t j = 0; j < normal.h_ll.size(); ++j) {
      Mat3 h = normal.h_ll[j];
      for (int a = 0; a < 3; ++a) h(a, a) += lambda * std::max(h(a, a), 1e-9);
      Eigen::FullPivLU<Mat3> lu(h);
      if (!lu.isInvertible()) return false;
      inverse[j] = lu.inverse();
      s -= normal.h_cl[j] * inverse[j] * normal.h_cl[j].transpose();
      g -= normal.h_cl[j] * inverse[j] * normal.b_l[j];
    }
    for (int k = 0; k < 7; ++k) {
      if (enabled_[k] != 0.0) continue;
      s.row(k).setZero();
      s.col(k).setZero();
      s(k, k) = 1.0;
      g[k] = 0.0;
    }
    if (schur != nullptr) *schur = s;
    dx = s.ldlt().solve(-g);
    if (!dx.allFinite()) return false;
    dl.resize(normal.h_ll.size());
    for (size_t j = 0; j < dl.size(); ++j) {
      dl[j] = -inverse[j] * (normal.b_l[j] + normal.h_cl[j].transpose() * dx);
    }
    return true;
  }

  // Pixel errors (not normalised) of every observation.
  std::vector<double> PixelErrors(const Vec7& x,
                                  const std::vector<Landmark>& landmarks,
                                  const std::vector<Vec3>& positions) const {
    const CameraCalibration calibration = Apply(initial_, x);
    std::vector<double> errors;
    for (size_t j = 0; j < landmarks.size(); ++j) {
      for (const TrackObservation& observation : landmarks[j].observations) {
        const std::optional<Vec2> r = PixelResidual(
            camera_, VehicleAt(calibration, observation),
            calibration.vehicle_from_camera, observation, positions[j], 1.0);
        errors.push_back(r ? r->norm()
                           : std::numeric_limits<double>::infinity());
      }
    }
    return errors;
  }

  const Vec7& enabled() const { return enabled_; }

 private:
  const CameraModel& camera_;
  const CameraCalibration& initial_;
  const std::vector<CalibrationImage>& images_;
  const PointCloudIndex* surface_;
  const Points* normals_;
  const CameraCalibrationConfig& config_;
  Vec7 prior_sigma_;
  Vec7 enabled_;
};

// RMS of the finite errors (failed projections are outliers).
double Rms(const std::vector<double>& errors) {
  double sum = 0.0;
  int count = 0;
  for (double e : errors) {
    if (!std::isfinite(e)) continue;
    sum += e * e;
    ++count;
  }
  return count > 0 ? std::sqrt(sum / count) : 0.0;
}

// Drops observations with errors above `max_error` (or failed projections)
// and landmarks left with fewer than two observations.
void PruneOutliers(const std::vector<double>& errors, double max_error,
                   std::vector<Landmark>& landmarks,
                   std::vector<Vec3>& positions) {
  std::vector<Landmark> kept;
  std::vector<Vec3> kept_positions;
  size_t e = 0;
  for (size_t j = 0; j < landmarks.size(); ++j) {
    Landmark landmark;
    landmark.surface = landmarks[j].surface;
    for (const TrackObservation& observation : landmarks[j].observations) {
      if (errors[e++] <= max_error) {
        landmark.observations.push_back(observation);
      }
    }
    if (landmark.observations.size() < 2) continue;
    kept.push_back(std::move(landmark));
    kept_positions.push_back(positions[j]);
  }
  landmarks.swap(kept);
  positions.swap(kept_positions);
}

// Levenberg-Marquardt over x and the landmark positions. With `fix_x`, only
// the landmarks move. Returns the number of accepted iterations.
int Optimize(const Problem& problem, const std::vector<Landmark>& landmarks,
             int iterations, bool fix_x, Vec7& x,
             std::vector<Vec3>& positions) {
  double lambda = 1e-3;
  double cost = problem.Cost(x, landmarks, positions);
  int accepted = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    Problem::Normal normal = problem.Linearize(x, landmarks, positions);
    if (fix_x) {
      normal.h_cc = Mat7::Identity();
      normal.b_c.setZero();
      for (Mat73& h : normal.h_cl) h.setZero();
    }
    bool improved = false;
    bool converged = false;
    for (int attempt = 0; attempt < 10; ++attempt) {
      Vec7 dx;
      std::vector<Vec3> dl;
      if (!problem.Solve(normal, lambda, dx, dl)) {
        lambda *= 10.0;
        continue;
      }
      if (fix_x) dx.setZero();
      const Vec7 candidate_x = x + dx;
      std::vector<Vec3> candidate = positions;
      for (size_t j = 0; j < candidate.size(); ++j) candidate[j] += dl[j];
      const double candidate_cost =
          problem.Cost(candidate_x, landmarks, candidate);
      if (candidate_cost < cost) {
        const double decrease = (cost - candidate_cost) / std::max(cost, 1e-12);
        x = candidate_x;
        positions.swap(candidate);
        cost = candidate_cost;
        lambda = std::max(lambda / 10.0, 1e-9);
        improved = true;
        converged = decrease < 1e-9;
        ++accepted;
        break;
      }
      lambda *= 10.0;
    }
    if (!improved || converged) break;
  }
  return accepted;
}

}  // namespace

std::optional<Vec2> ProjectWithCalibration(const CameraModel& camera,
                                           const CameraCalibration& calibration,
                                           const CalibrationImage& image,
                                           const Vec3& point_world) {
  GR_CHECK(image.trajectory != nullptr);
  double row = 0.5 * (camera.height() - 1);
  std::optional<Vec2> uv;
  for (int iteration = 0; iteration < kMaxRowIterations; ++iteration) {
    const Pose world_from_camera =
        image.trajectory->At(RowTime(calibration, image, row)) *
        calibration.vehicle_from_camera;
    uv = camera.Project(world_from_camera.Inverse() * point_world);
    if (!uv || calibration.line_time == 0.0) return uv;
    const double next = std::clamp(uv->y(), 0.0, camera.height() - 1.0);
    if (std::abs(next - row) < 0.01) return uv;
    row = next;
  }
  return uv;
}

CameraCalibrationResult CalibrateCamera(
    const CameraModel& camera, const CameraCalibration& initial,
    const std::vector<CalibrationImage>& images,
    const std::vector<Track>& tracks, const PointCloudIndex* surface,
    const Points* normals, const CameraCalibrationConfig& config) {
  GR_CHECK(surface == nullptr ||
           (normals != nullptr && normals->size() == surface->size()));
  for (const CalibrationImage& image : images) {
    GR_CHECK(image.trajectory != nullptr && !image.trajectory->empty());
  }
  CameraCalibrationResult result;
  result.calibration = initial;
  const Problem problem(camera, initial, images, surface, normals, config);

  // Landmarks triangulated with the initial calibration.
  std::vector<Landmark> landmarks;
  std::vector<Vec3> positions;
  for (const Track& track : tracks) {
    Landmark landmark;
    std::vector<Vec3> centers;
    std::vector<Vec3> directions;
    for (const TrackObservation& observation : track) {
      if (observation.image < 0 ||
          observation.image >= static_cast<int>(images.size())) {
        continue;
      }
      const std::optional<Vec3> ray = camera.Unproject(observation.uv);
      if (!ray) continue;
      const Pose world_from_camera =
          problem.VehicleAt(initial, observation) * initial.vehicle_from_camera;
      centers.push_back(world_from_camera.translation);
      directions.push_back(world_from_camera.rotation * ray->normalized());
      landmark.observations.push_back(observation);
    }
    if (landmark.observations.size() < 2) continue;
    const std::optional<Vec3> point =
        Triangulate(centers, directions, DegToRad(config.min_parallax_deg));
    if (!point) continue;
    landmarks.push_back(std::move(landmark));
    positions.push_back(*point);
  }

  // Landmarks alone (calibration fixed), then outliers out.
  Vec7 x = Vec7::Zero();
  Optimize(problem, landmarks, 5, /*fix_x=*/true, x, positions);
  PruneOutliers(problem.PixelErrors(x, landmarks, positions),
                config.max_reprojection, landmarks, positions);
  for (size_t j = 0; j < landmarks.size(); ++j) {
    landmarks[j].surface = problem.NearestSurface(positions[j]);
  }
  Optimize(problem, landmarks, 5, /*fix_x=*/true, x, positions);
  result.rms_before = Rms(problem.PixelErrors(x, landmarks, positions));

  // Joint refinement; outliers and the surface association are updated
  // between rounds.
  int iterations = 0;
  for (int round = 0; round < 3; ++round) {
    iterations += Optimize(problem, landmarks, config.iterations,
                           /*fix_x=*/false, x, positions);
    PruneOutliers(problem.PixelErrors(x, landmarks, positions),
                  config.max_reprojection, landmarks, positions);
    for (size_t j = 0; j < landmarks.size(); ++j) {
      landmarks[j].surface = problem.NearestSurface(positions[j]);
    }
  }
  result.iterations = iterations;
  result.calibration = Apply(initial, x);
  const std::vector<double> errors =
      problem.PixelErrors(x, landmarks, positions);
  result.rms_after = Rms(errors);
  result.landmarks = static_cast<int>(landmarks.size());
  result.observations = static_cast<int>(errors.size());

  // Uncertainty of the calibration from the undamped Schur complement.
  const Problem::Normal normal = problem.Linearize(x, landmarks, positions);
  Vec7 dx;
  std::vector<Vec3> dl;
  Mat7 schur;
  if (problem.Solve(normal, 0.0, dx, dl, &schur)) {
    const Mat7 covariance = schur.inverse();
    const Vec7& enabled = problem.enabled();
    const auto sigma = [&](int k) {
      return enabled[k] != 0.0 ? std::sqrt(std::max(covariance(k, k), 0.0))
                               : 0.0;
    };
    result.time_offset_sigma = sigma(0);
    result.rotation_sigma = Vec3(sigma(1), sigma(2), sigma(3));
    result.translation_sigma = Vec3(sigma(4), sigma(5), sigma(6));
  }
  return result;
}

}  // namespace gr
