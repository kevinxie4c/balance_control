#include "MathUtil.h"

using namespace Eigen;

void setTransNRot(const Isometry3d &T, Vector3d &trans, Vector3d &rot)
{
    trans = T.translation();
    AngleAxisd aa(T.rotation());
    rot = aa.angle() * aa.axis();
}
