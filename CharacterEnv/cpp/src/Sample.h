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
        Eigen::VectorXd action;
        double logprob;
        double reward;
        double done;
        double retval;
        double value;

        std::shared_ptr<Sample> parent = nullptr;
};

class RewardCmp
{
    public:
	bool operator()(const std::shared_ptr<Sample> &lhs, const std::shared_ptr<Sample> &rhs);
};

#endif
