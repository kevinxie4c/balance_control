#ifndef ROD_ENV_H
#define ROD_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class RodEnv: public CharacterEnv
{
    public:
        RodEnv(const char *cfgFilename);
        void reset() override;
        void step() override;
        void updateState();

        int actionRate = 10;
        int forceRate = 100;

        Eigen::VectorXd scales;
};

#endif
