#include "Sample.h"

Sample::Sample(std::shared_ptr<Sample> parent, double logprob, CharacterEnv *env): parent(parent), logprob(logprob)
{
    position = env->skeleton->getPositions();
    velocity = env->skeleton->getVelocities();
    observation = env->state;
    reward = env->reward;
    done = env->done;
}

bool RewardCmp::operator()(const std::shared_ptr<Sample> &lhs, const std::shared_ptr<Sample> &rhs)
{
    return lhs->reward > rhs->reward;
}
