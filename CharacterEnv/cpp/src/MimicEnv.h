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
        std::vector<Eigen::VectorXd> refMotion;
        Eigen::VectorXd scales;

        int mocapFPS = 120;
        int actionRate = 30;
        int forceRate = 600;
        int frameIdx = 0;
        double period, phase, phaseShift;

        double w_p = 5, w_r = 3, w_e = 30, w_b = 10;

        std::default_random_engine generator;
        std::uniform_real_distribution<double> uni_dist;
        std::normal_distribution<double> norm_dist;
};

#endif
