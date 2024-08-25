#ifndef BALANCER_ENV_H
#define BALANCER_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class BalancerEnv: public CharacterEnv
{
    public:
        BalancerEnv(const char *cfgFilename);
        void reset() override;
        void step() override;
        void updateState();

        int actionRate = 30;
        int forceRate = 100;

        Eigen::VectorXd scales;
};

#endif
