#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <Eigen/Core>
#include <Eigen/Geometry>

void setTransNRot(const Eigen::Isometry3d &T, Eigen::Vector3d &trans, Eigen::Vector3d &rot);

#endif
