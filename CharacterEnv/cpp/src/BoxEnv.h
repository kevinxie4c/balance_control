#ifndef BOX_ENV_H
#define BOX_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include <random>
#include <fstream>
#include "CharacterEnv.h"

class BoxEnv: public CharacterEnv
{
    public:
        BoxEnv(const char *cfgFilename);
        void reset() override;
        void step() override;
        void updateState();
        ~BoxEnv();

        int actionRate = 100;
        int forceRate = 100;

        Eigen::VectorXd scales;

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution;

        std::ofstream fh_pos;
        double w_a = 1;
};

#endif
