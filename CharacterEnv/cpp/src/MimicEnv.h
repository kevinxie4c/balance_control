#ifndef MIMIC_ENV_H
#define MIMIC_ENV_H

#include <random>
#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class MimicEnv: public CharacterEnv
{
    public:
        MimicEnv(const char *cfgFilename);
        void reset() override;
        void step() override;
        void updateState();
        //void setTimeStep(double h);
        double cost();
        void print_info() override;

        dart::dynamics::SkeletonPtr kin_skeleton, floor;
        Eigen::VectorXd kp, kd;
        Eigen::MatrixXd mkp, mkd;
        size_t endEffectorIndices[4];
        std::vector<std::string> endEffectorNames{"Foot", "foot", "Hand", "hand"};
        std::vector<Eigen::VectorXd> positions;
        std::vector<double> scales;

        int mocapFPS = 120;
        int actionRate = 30;
        int forceRate = 600;
        size_t frameIdx = 0;

        double w_p = 5, w_r = 3, w_e = 30, w_b = 10;

        bool enableRSI = false;

        std::mt19937 rng;
        std::uniform_int_distribution<size_t> uni_dist;
};

#endif
