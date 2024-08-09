#ifndef CARTPOLE_ENV_H
#define CARTPOLE_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class CartpoleEnv: public CharacterEnv
{
    public:
        CartpoleEnv(const char *cfgFilename);
        void reset() override;
        void step() override;
        void updateState();

        int actionRate = 10;
        int forceRate = 10;

        Eigen::VectorXd scales;
};

#endif
