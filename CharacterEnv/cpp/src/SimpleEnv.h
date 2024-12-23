#ifndef SIMPLE_ENV_H
#define SIMPLE_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class SimpleEnv: public CharacterEnv
{
    public:
        SimpleEnv(const char *cfgFilename);
        ~SimpleEnv();
        void reset() override;
        void step() override;
        void updateState();

        int actionRate = 30;
        int forceRate = 300;

        dart::dynamics::SkeletonPtr floor;
        Eigen::VectorXd scales;
        Eigen::Vector3d prev_com;

        std::vector<Eigen::VectorXd> refMotion;
        int frameRate;
};

#endif
