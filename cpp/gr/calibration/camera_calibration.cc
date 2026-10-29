#include "gr/calibration/camera_calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "Eigen/Dense"

#include "gr/core/check.h"

namespace gr {
namespace {

// The calibration correction x = [time offset, rotation, translation].
constexpr int kTimeOffsetParam = 0;
constexpr int kRotationParam = kTimeOffsetParam + 1;
constexpr int kTranslationParam = kRotationParam + kSpaceDim;
constexpr int kCalibrationParams = kTranslationParam + kSpaceDim;

using CalibrationVector = Eigen::Matrix<double, kCalibrationParams, 1>;
using CalibrationMatrix =
    Eigen::Matrix<double, kCalibrationParams, kCalibrationParams>;
// Coupling of the calibration and one landmark.
using CouplingMatrix = Eigen::Matrix<double, kCalibrationParams, kSpaceDim>;

// The residual of a pixel: u and v.
constexpr int kPixelDim = 2;

constexpr int kMaxRowIterations = 4;
// Rows closer than this are the same: the projection is self-consistent.
constexpr double kRowTolerance = 0.01;
// A landmark needs two observations.
constexpr size_t kMinObservations = 2;
// A failed projection costs as much as a residual of this many sigmas on
// the linear part of the Huber loss.
constexpr double kFailedProjectionSigmas = 10.0;
// Central-difference steps: time offset (s), angles (rad), lengths (m).
constexpr double kTimeStep = 1e-4;
constexpr double kAngleStep = 1e-6;
constexpr double kLengthStep = 1e-5;
// Levenberg-Marquardt: damping (relative to the diagonal, floored), its
// start, change and floor, the attempts per iteration and the relative cost
// decrease below which the solve has converged.
constexpr double kMinDampedDiagonal = 1e-9;
constexpr double kInitialDamping = 1e-3;
constexpr double kDampingFactor = 10.0;
constexpr double kMinDamping = 1e-9;
constexpr int kMaxDampingAttempts = 10;
constexpr double kMinCost = 1e-12;
constexpr double kMinRelativeDecrease = 1e-9;
// Iterations that refine the landmarks alone, and rounds of the joint
// refinement (outliers and surfaces updated between them).
constexpr int kLandmarkIterations = 5;
constexpr int kJointRounds = 3;
// Landmark::surface without a LiDAR point.
constexpr int kNoSurface = -1;

// The calibration moved by x; the rotation and translation act in the
// camera frame.
CameraCalibration Apply(const CameraCalibration& base,
                        const CalibrationVector& x) {
  CameraCalibration out = base;
  out.time_offset = base.time_offset + x[kTimeOffsetParam];
  out.vehicle_from_camera = base.vehicle_from_camera *
                            Pose(So3Exp(x.segment<kSpaceDim>(kRotationParam)),
                                 x.segment<kSpaceDim>(kTranslationParam));
  return out;
}

double RowTime(const CameraCalibration& calibration,
               const CalibrationImage& image, double row) {
  return image.stamp + calibration.time_offset + row * calibration.line_time;
}

struct Landmark {
  Vec3 position = Vec3::Zero();
  std::vector<TrackObservation> observations;
  int surface = kNoSurface;  // Nearest LiDAR point.
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
  for (size_t first = 0; first < directions.size(); ++first) {
    for (size_t second = first + 1; second < directions.size(); ++second) {
      const double cosine =
          std::clamp(directions[first].dot(directions[second]), -1.0, 1.0);
      max_angle = std::max(max_angle, std::acos(cosine));
    }
  }
  if (max_angle < min_parallax_rad) return std::nullopt;
  // Normal equations of the distances to the rays: sum of the projectors
  // onto the planes normal to each ray.
  Mat3 normal_matrix = Mat3::Zero();
  Vec3 right_side = Vec3::Zero();
  for (size_t k = 0; k < centers.size(); ++k) {
    const Mat3 projector =
        Mat3::Identity() - directions[k] * directions[k].transpose();
    normal_matrix += projector;
    right_side += projector * centers[k];
  }
  const Vec3 point = normal_matrix.ldlt().solve(right_side);
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
  // Gauss-Newton blocks: calibration (c), landmarks (l) and their coupling;
  // H for the normal matrix, b for the gradient.
  struct Normal {
    CalibrationMatrix h_cc = CalibrationMatrix::Zero();
    CalibrationVector b_c = CalibrationVector::Zero();
    std::vector<Mat3> h_ll;
    std::vector<CouplingMatrix> h_cl;
    std::vector<Vec3> b_l;
  };

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

  // Robust (Huber) cost of a normalised residual with |r|^2 = squared_norm.
  double Rho(double squared_norm) const {
    const double threshold = config_.huber;
    return squared_norm <= threshold * threshold
               ? squared_norm
               : 2.0 * threshold * std::sqrt(squared_norm) -
                     threshold * threshold;
  }
  double Weight(double squared_norm) const {
    const double threshold = config_.huber;
    return squared_norm <= threshold * threshold
               ? 1.0
               : threshold / std::sqrt(squared_norm);
  }

  Pose VehicleAt(const CameraCalibration& calibration,
                 const TrackObservation& observation) const {
    const CalibrationImage& image =
        images_[static_cast<size_t>(observation.image)];
    return image.trajectory->At(
        RowTime(calibration, image, observation.uv.y()));
  }

  // Nearest LiDAR point within the search radius, kNoSurface if none.
  int NearestSurface(const Vec3& point) const {
    if (surface_ == nullptr) return kNoSurface;
    int best = kNoSurface;
    double best_squared_distance = config_.plane_radius * config_.plane_radius;
    for (int index : surface_->RadiusSearch(point, config_.plane_radius)) {
      const double squared_distance =
          (surface_->points()[static_cast<size_t>(index)] - point)
              .squaredNorm();
      if (squared_distance < best_squared_distance) {
        best_squared_distance = squared_distance;
        best = index;
      }
    }
    return best;
  }

  double PlaneResidual(const Landmark& landmark, const Vec3& point) const {
    const size_t surface = static_cast<size_t>(landmark.surface);
    return (*normals_)[surface].dot(point - surface_->points()[surface]) /
           config_.plane_sigma;
  }

  double Cost(const CalibrationVector& x,
              const std::vector<Landmark>& landmarks,
              const std::vector<Vec3>& positions) const {
    const CameraCalibration calibration = Apply(initial_, x);
    double cost = 0.0;
    for (size_t j = 0; j < landmarks.size(); ++j) {
      for (const TrackObservation& observation : landmarks[j].observations) {
        const std::optional<Vec2> residual =
            PixelResidual(camera_, VehicleAt(calibration, observation),
                          calibration.vehicle_from_camera, observation,
                          positions[j], config_.pixel_sigma);
        cost += residual ? Rho(residual->squaredNorm())
                         : 2.0 * config_.huber * kFailedProjectionSigmas;
      }
      if (landmarks[j].surface != kNoSurface) {
        const double residual = PlaneResidual(landmarks[j], positions[j]);
        cost += residual * residual;
      }
    }
    for (int k = 0; k < kCalibrationParams; ++k) {
      if (enabled_[k] == 0.0) continue;
      const double residual = x[k] / prior_sigma_[k];
      cost += residual * residual;
    }
    return cost;
  }

  Normal Linearize(const CalibrationVector& x,
                   const std::vector<Landmark>& landmarks,
                   const std::vector<Vec3>& positions) const {
    const CameraCalibration calibration = Apply(initial_, x);
    Normal normal;
    normal.h_ll.assign(landmarks.size(), Mat3::Zero());
    normal.h_cl.assign(landmarks.size(), CouplingMatrix::Zero());
    normal.b_l.assign(landmarks.size(), Vec3::Zero());
    // Central differences: the residual at +step and at -step.
    constexpr int kPlus = 0;
    constexpr int kMinus = 1;
    for (size_t j = 0; j < landmarks.size(); ++j) {
      const Vec3& point = positions[j];
      for (const TrackObservation& observation : landmarks[j].observations) {
        const Pose vehicle = VehicleAt(calibration, observation);
        const std::optional<Vec2> residual =
            PixelResidual(camera_, vehicle, calibration.vehicle_from_camera,
                          observation, point, config_.pixel_sigma);
        if (!residual) continue;
        Eigen::Matrix<double, kPixelDim, kCalibrationParams>
            calibration_jacobian =
                Eigen::Matrix<double, kPixelDim, kCalibrationParams>::Zero();
        Eigen::Matrix<double, kPixelDim, kSpaceDim> landmark_jacobian;
        bool ok = true;
        for (int k = 0; k < kCalibrationParams && ok; ++k) {
          if (enabled_[k] == 0.0) continue;
          const double step = k == kTimeOffsetParam   ? kTimeStep
                              : k < kTranslationParam ? kAngleStep
                                                      : kLengthStep;
          Vec2 sides[2];
          for (int side : {kPlus, kMinus}) {
            CalibrationVector perturbed = x;
            perturbed[k] += side == kPlus ? step : -step;
            const CameraCalibration moved = Apply(initial_, perturbed);
            const Pose moved_vehicle =
                k == kTimeOffsetParam ? VehicleAt(moved, observation) : vehicle;
            const std::optional<Vec2> moved_residual =
                PixelResidual(camera_, moved_vehicle, moved.vehicle_from_camera,
                              observation, point, config_.pixel_sigma);
            if (!moved_residual) {
              ok = false;
              break;
            }
            sides[side] = *moved_residual;
          }
          if (ok) {
            calibration_jacobian.col(k) =
                (sides[kPlus] - sides[kMinus]) / (2.0 * step);
          }
        }
        for (int axis = 0; axis < kSpaceDim && ok; ++axis) {
          Vec2 sides[2];
          for (int side : {kPlus, kMinus}) {
            Vec3 moved = point;
            moved[axis] += side == kPlus ? kLengthStep : -kLengthStep;
            const std::optional<Vec2> moved_residual =
                PixelResidual(camera_, vehicle, calibration.vehicle_from_camera,
                              observation, moved, config_.pixel_sigma);
            if (!moved_residual) {
              ok = false;
              break;
            }
            sides[side] = *moved_residual;
          }
          if (ok) {
            landmark_jacobian.col(axis) =
                (sides[kPlus] - sides[kMinus]) / (2.0 * kLengthStep);
          }
        }
        if (!ok) continue;
        const double weight = Weight(residual->squaredNorm());
        normal.h_cc +=
            weight * calibration_jacobian.transpose() * calibration_jacobian;
        normal.b_c += weight * calibration_jacobian.transpose() * *residual;
        normal.h_ll[j] +=
            weight * landmark_jacobian.transpose() * landmark_jacobian;
        normal.h_cl[j] +=
            weight * calibration_jacobian.transpose() * landmark_jacobian;
        normal.b_l[j] += weight * landmark_jacobian.transpose() * *residual;
      }
      if (landmarks[j].surface != kNoSurface) {
        const Vec3 scaled_normal =
            (*normals_)[static_cast<size_t>(landmarks[j].surface)] /
            config_.plane_sigma;
        normal.h_ll[j] += scaled_normal * scaled_normal.transpose();
        normal.b_l[j] += scaled_normal * PlaneResidual(landmarks[j], point);
      }
    }
    for (int k = 0; k < kCalibrationParams; ++k) {
      if (enabled_[k] == 0.0) continue;
      const double information = 1.0 / (prior_sigma_[k] * prior_sigma_[k]);
      normal.h_cc(k, k) += information;
      normal.b_c[k] += information * x[k];
    }
    return normal;
  }

  // Damped Schur-complement step; false if a block is singular.
  bool Solve(const Normal& normal, double lambda, CalibrationVector* dx,
             std::vector<Vec3>* dl, CalibrationMatrix* schur = nullptr) const {
    CalibrationMatrix reduced = normal.h_cc;
    CalibrationVector reduced_gradient = normal.b_c;
    for (int k = 0; k < kCalibrationParams; ++k) {
      reduced(k, k) += lambda * std::max(reduced(k, k), kMinDampedDiagonal);
    }
    std::vector<Mat3> inverse(normal.h_ll.size());
    for (size_t j = 0; j < normal.h_ll.size(); ++j) {
      Mat3 block = normal.h_ll[j];
      for (int axis = 0; axis < kSpaceDim; ++axis) {
        block(axis, axis) +=
            lambda * std::max(block(axis, axis), kMinDampedDiagonal);
      }
      Eigen::FullPivLU<Mat3> lu(block);
      if (!lu.isInvertible()) return false;
      inverse[j] = lu.inverse();
      reduced -= normal.h_cl[j] * inverse[j] * normal.h_cl[j].transpose();
      reduced_gradient -= normal.h_cl[j] * inverse[j] * normal.b_l[j];
    }
    for (int k = 0; k < kCalibrationParams; ++k) {
      if (enabled_[k] != 0.0) continue;
      reduced.row(k).setZero();
      reduced.col(k).setZero();
      reduced(k, k) = 1.0;
      reduced_gradient[k] = 0.0;
    }
    if (schur != nullptr) *schur = reduced;
    *dx = reduced.ldlt().solve(-reduced_gradient);
    if (!dx->allFinite()) return false;
    dl->resize(normal.h_ll.size());
    for (size_t j = 0; j < dl->size(); ++j) {
      (*dl)[j] =
          -inverse[j] * (normal.b_l[j] + normal.h_cl[j].transpose() * *dx);
    }
    return true;
  }

  // Pixel errors (not normalised) of every observation.
  std::vector<double> PixelErrors(const CalibrationVector& x,
                                  const std::vector<Landmark>& landmarks,
                                  const std::vector<Vec3>& positions) const {
    const CameraCalibration calibration = Apply(initial_, x);
    std::vector<double> errors;
    for (size_t j = 0; j < landmarks.size(); ++j) {
      for (const TrackObservation& observation : landmarks[j].observations) {
        const std::optional<Vec2> residual = PixelResidual(
            camera_, VehicleAt(calibration, observation),
            calibration.vehicle_from_camera, observation, positions[j], 1.0);
        errors.push_back(residual ? residual->norm()
                                  : std::numeric_limits<double>::infinity());
      }
    }
    return errors;
  }

  const CalibrationVector& enabled() const { return enabled_; }

 private:
  const CameraModel& camera_;
  const CameraCalibration& initial_;
  const std::vector<CalibrationImage>& images_;
  const PointCloudIndex* surface_;
  const Points* normals_;
  const CameraCalibrationConfig& config_;
  CalibrationVector prior_sigma_;
  CalibrationVector enabled_;  // 1 for an estimated parameter, else 0.
};

// RMS of the finite errors (failed projections are outliers).
double Rms(const std::vector<double>& errors) {
  double sum = 0.0;
  int count = 0;
  for (double error : errors) {
    if (!std::isfinite(error)) continue;
    sum += error * error;
    ++count;
  }
  return count > 0 ? std::sqrt(sum / count) : 0.0;
}

// Drops observations with errors above `max_error` (or failed projections)
// and landmarks left with fewer than two observations.
void PruneOutliers(const std::vector<double>& errors, double max_error,
                   std::vector<Landmark>* landmarks,
                   std::vector<Vec3>* positions) {
  std::vector<Landmark> kept;
  std::vector<Vec3> kept_positions;
  size_t error_index = 0;
  for (size_t j = 0; j < landmarks->size(); ++j) {
    Landmark landmark;
    landmark.surface = (*landmarks)[j].surface;
    for (const TrackObservation& observation : (*landmarks)[j].observations) {
      if (errors[error_index++] <= max_error) {
        landmark.observations.push_back(observation);
      }
    }
    if (landmark.observations.size() < kMinObservations) continue;
    kept.push_back(std::move(landmark));
    kept_positions.push_back((*positions)[j]);
  }
  landmarks->swap(kept);
  positions->swap(kept_positions);
}

// Levenberg-Marquardt over x and the landmark positions. With `fix_x`, only
// the landmarks move. Returns the number of accepted iterations.
int Optimize(const Problem& problem, const std::vector<Landmark>& landmarks,
             int iterations, bool fix_x, CalibrationVector* x,
             std::vector<Vec3>* positions) {
  double lambda = kInitialDamping;
  double cost = problem.Cost(*x, landmarks, *positions);
  int accepted = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    Problem::Normal normal = problem.Linearize(*x, landmarks, *positions);
    if (fix_x) {
      normal.h_cc = CalibrationMatrix::Identity();
      normal.b_c.setZero();
      for (CouplingMatrix& coupling : normal.h_cl) coupling.setZero();
    }
    bool improved = false;
    bool converged = false;
    for (int attempt = 0; attempt < kMaxDampingAttempts; ++attempt) {
      CalibrationVector dx;
      std::vector<Vec3> dl;
      if (!problem.Solve(normal, lambda, &dx, &dl)) {
        lambda *= kDampingFactor;
        continue;
      }
      if (fix_x) dx.setZero();
      const CalibrationVector candidate_x = *x + dx;
      std::vector<Vec3> candidate = *positions;
      for (size_t j = 0; j < candidate.size(); ++j) candidate[j] += dl[j];
      const double candidate_cost =
          problem.Cost(candidate_x, landmarks, candidate);
      if (candidate_cost < cost) {
        const double decrease =
            (cost - candidate_cost) / std::max(cost, kMinCost);
        *x = candidate_x;
        positions->swap(candidate);
        cost = candidate_cost;
        lambda = std::max(lambda / kDampingFactor, kMinDamping);
        improved = true;
        converged = decrease < kMinRelativeDecrease;
        ++accepted;
        break;
      }
      lambda *= kDampingFactor;
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
  double row = 0.5 * (camera.height() - 1);  // Start in the middle.
  std::optional<Vec2> uv;
  for (int iteration = 0; iteration < kMaxRowIterations; ++iteration) {
    const Pose world_from_camera =
        image.trajectory->At(RowTime(calibration, image, row)) *
        calibration.vehicle_from_camera;
    uv = camera.Project(world_from_camera.Inverse() * point_world);
    if (!uv || calibration.line_time == 0.0) return uv;
    const double landing_row = std::clamp(uv->y(), 0.0, camera.height() - 1.0);
    if (std::abs(landing_row - row) < kRowTolerance) return uv;
    row = landing_row;
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
    if (landmark.observations.size() < kMinObservations) continue;
    const std::optional<Vec3> point =
        Triangulate(centers, directions, DegToRad(config.min_parallax_deg));
    if (!point) continue;
    landmarks.push_back(std::move(landmark));
    positions.push_back(*point);
  }

  // Landmarks alone (calibration fixed), then outliers out.
  CalibrationVector x = CalibrationVector::Zero();
  Optimize(problem, landmarks, kLandmarkIterations, /*fix_x=*/true, &x,
           &positions);
  PruneOutliers(problem.PixelErrors(x, landmarks, positions),
                config.max_reprojection, &landmarks, &positions);
  for (size_t j = 0; j < landmarks.size(); ++j) {
    landmarks[j].surface = problem.NearestSurface(positions[j]);
  }
  Optimize(problem, landmarks, kLandmarkIterations, /*fix_x=*/true, &x,
           &positions);
  result.rms_before = Rms(problem.PixelErrors(x, landmarks, positions));

  // Joint refinement; outliers and the surface association are updated
  // between rounds.
  int iterations = 0;
  for (int round = 0; round < kJointRounds; ++round) {
    iterations += Optimize(problem, landmarks, config.iterations,
                           /*fix_x=*/false, &x, &positions);
    PruneOutliers(problem.PixelErrors(x, landmarks, positions),
                  config.max_reprojection, &landmarks, &positions);
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
  CalibrationVector dx;
  std::vector<Vec3> dl;
  CalibrationMatrix schur;
  if (problem.Solve(normal, 0.0, &dx, &dl, &schur)) {
    const CalibrationMatrix covariance = schur.inverse();
    const CalibrationVector& enabled = problem.enabled();
    const auto sigma = [&](int k) {
      return enabled[k] != 0.0 ? std::sqrt(std::max(covariance(k, k), 0.0))
                               : 0.0;
    };
    const auto sigmas = [&](int first) {
      return Vec3(sigma(first), sigma(first + 1), sigma(first + 2));
    };
    result.time_offset_sigma = sigma(kTimeOffsetParam);
    result.rotation_sigma = sigmas(kRotationParam);
    result.translation_sigma = sigmas(kTranslationParam);
  }
  return result;
}

}  // namespace gr
