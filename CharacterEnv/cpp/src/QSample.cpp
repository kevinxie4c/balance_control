#include <limits>
#include "QSample.h"

QSample::QSample(std::shared_ptr<QSample> parent, CharacterEnv *env): parent(parent), value(0)
{
    position = env->skeleton->getPositions();
    velocity = env->skeleton->getVelocities();
    observation = env->state;
    action = env->action;
    reward = env->reward;
    retval = std::numeric_limits<double>::min();
    maxval = std::numeric_limits<double>::min();
    done = env->done;
    if (parent == nullptr)
        accReward = reward;
    else
        accReward = parent->accReward + reward;
}
