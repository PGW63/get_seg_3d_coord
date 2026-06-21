#include "get_seg_3d_coord/position_ekf.hpp"

#include <algorithm>

#include <Eigen/LU>

namespace get_seg_3d_coord {

void PositionEkf::reset() {
  initialized_ = false;
  position_.setZero();
  covariance_.setIdentity();
}

bool PositionEkf::initialized() const { return initialized_; }

void PositionEkf::initialize(const Eigen::Vector3f &measurement,
                             const double initial_variance) {
  initialized_ = true;
  position_ = measurement;
  covariance_.setIdentity();
  covariance_ *= static_cast<float>(std::max(1e-6, initial_variance));
}

void PositionEkf::predict(const double dt_sec,
                          const double process_variance_per_sec) {
  if (!initialized_) {
    return;
  }
  const double q = std::max(0.0, dt_sec) *
                   std::max(0.0, process_variance_per_sec);
  covariance_ +=
      static_cast<float>(q) * Eigen::Matrix3f::Identity();
}

void PositionEkf::update(const Eigen::Vector3f &measurement,
                         const double measurement_variance) {
  if (!initialized_) {
    initialize(measurement, measurement_variance);
    return;
  }

  const Eigen::Matrix3f r =
      static_cast<float>(std::max(1e-6, measurement_variance)) *
      Eigen::Matrix3f::Identity();
  const Eigen::Matrix3f s = covariance_ + r;
  const Eigen::Matrix3f k = covariance_ * s.inverse();
  position_ = position_ + k * (measurement - position_);
  covariance_ = (Eigen::Matrix3f::Identity() - k) * covariance_;
}

double PositionEkf::normalizedInnovationSquared(
    const Eigen::Vector3f &measurement,
    const double measurement_variance) const {
  if (!initialized_) {
    return -1.0;
  }
  const Eigen::Matrix3f r =
      static_cast<float>(std::max(1e-6, measurement_variance)) *
      Eigen::Matrix3f::Identity();
  const Eigen::Matrix3f s = covariance_ + r;
  const Eigen::Vector3f innovation = measurement - position_;
  return static_cast<double>(innovation.transpose() * s.inverse() *
                             innovation);
}

const Eigen::Vector3f &PositionEkf::position() const { return position_; }

const Eigen::Matrix3f &PositionEkf::covariance() const {
  return covariance_;
}

double PositionEkf::maxVariance() const {
  return static_cast<double>(covariance_.diagonal().maxCoeff());
}

}  // namespace get_seg_3d_coord
