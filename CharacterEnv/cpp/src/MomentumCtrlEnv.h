#ifndef MOMENTUM_CTRL_ENV_H
#define MOMENTUM_CTRL_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>
#include "CharacterEnv.h"

class MomentumCtrlEnv: public CharacterEnv
{
    public:
        MomentumCtrlEnv(const char *cfgFilename);
        ~MomentumCtrlEnv();
        void reset() override;
        void step() override;
        void updateState();

        int actionRate = 30;
        int forceRate = 300;

        dart::dynamics::SkeletonPtr floor;
        Eigen::VectorXd scales;

        Eigen::VectorXd kp, kd;
        Eigen::MatrixXd mkp, mkd;

        bool fallen = false;
        Eigen::Vector3d L_prev;
        double t_prev;
        std::ofstream f_dL;
        std::ofstream f_forces;

        bool implicit = false;
};

#endif
