#ifndef SIMPLE_ENV_H
#define SIMPLE_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class SimpleEnv: public CharacterEnv
{
    public:
        SimpleEnv(const char *cfgFilename);
        void reset() override;
        void step() override;
        void updateState();

        int actionRate = 30;
        int forceRate = 6000;

        dart::dynamics::SkeletonPtr floor;
        Eigen::VectorXd scales;
};

#endif
