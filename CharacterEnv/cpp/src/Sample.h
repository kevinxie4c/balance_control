#ifndef SAMPLE_H
#define SAMPLE_H

#include <memory>
#include <Eigen/Core>
#include "CharacterEnv.h"

class Sample
{
    public:
        Sample(std::shared_ptr<Sample> parent, double logprob, CharacterEnv *env);

        Eigen::VectorXd position;
        Eigen::VectorXd velocity;
        Eigen::VectorXd observation;
        Eigen::VectorXd action; // action taken from the parent to the current
        double logprob;
        double reward;
        double done;
        double retval; // "real" value
        double value; // value based on the critic
        double delta;
        double advantage;
        double accReward;

        std::shared_ptr<Sample> parent = nullptr;
        std::shared_ptr<Sample> firstChild = nullptr;
};

class RewardCmp
{
    public:
	bool operator()(const std::shared_ptr<Sample> &lhs, const std::shared_ptr<Sample> &rhs);
};

#endif
