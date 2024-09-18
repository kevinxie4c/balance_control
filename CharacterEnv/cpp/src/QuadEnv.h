#ifndef QUAD_ENV_H
#define QUAD_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class QuadEnv: public CharacterEnv
{
    public:
        QuadEnv(const char *cfgFilename);
        void reset() override;
        void step() override;
        void updateState();

        int actionRate = 30;
        int forceRate = 300;

        dart::dynamics::SkeletonPtr floor;
        Eigen::VectorXd scales;
        Eigen::Vector3d prev_com;

        bool fallen = false;
};

#endif
