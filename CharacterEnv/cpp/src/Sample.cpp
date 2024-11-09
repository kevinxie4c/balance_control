#include <limits>
#include "Sample.h"

Sample::Sample(std::shared_ptr<Sample> parent, double logprob, CharacterEnv *env): parent(parent), logprob(logprob), value(0), delta(0), advantage(0)
{
    position = env->skeleton->getPositions();
    velocity = env->skeleton->getVelocities();
    observation = env->state;
    action = env->action;
    reward = env->reward;
    retval = std::numeric_limits<double>::min();
    done = env->done;
}

bool RewardCmp::operator()(const std::shared_ptr<Sample> &lhs, const std::shared_ptr<Sample> &rhs)
{
    return lhs->reward > rhs->reward;
}
