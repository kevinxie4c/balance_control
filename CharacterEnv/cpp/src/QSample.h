#ifndef QSAMPLE_H
#define QSAMPLE_H

#include <memory>
#include <Eigen/Core>
#include "CharacterEnv.h"

class QSample
{
    public:
        QSample(std::shared_ptr<QSample> parent, CharacterEnv *env);

        Eigen::VectorXd position;
        Eigen::VectorXd velocity;
        Eigen::VectorXd observation;
        Eigen::VectorXd action; // action taken from the parent to the current
        double reward;
        double done;
        double retval; // "real" value (Q(parent->state, action))
        double value; // value based on the critic (Q(parent->state, action))
        double accReward;
        double maxval; // max_a Q(state, a)

        std::shared_ptr<QSample> parent = nullptr;
        std::shared_ptr<QSample> firstChild = nullptr;
};

#endif
