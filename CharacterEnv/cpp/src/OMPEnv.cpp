#include <iostream>
#include <pthread.h>
#include "OMPEnv.h"

using namespace std;

OMPEnv::OMPEnv(const char *cfgFilename, size_t num_threads): num_threads(num_threads)
{
    for (size_t i = 0; i < num_threads; ++i)
    {
	envs.push_back(CharacterEnv::makeEnv(cfgFilename));
    }
}

void OMPEnv::reset()
{
    for (size_t i = 0; i < num_threads; ++i)
    {
	envs[i]->reset();
    }
}


void OMPEnv::step()
{
    #pragma omp for
    for (size_t i = 0; i < positions.size(); ++i)
    {
        size_t id = i % num_threads;
        envs[id]->skeleton->setPositions(positions[i]);
        envs[id]->skeleton->setVelocities(velocities[i]);
        envs[id]->action = actions[i];
        envs[id]->step();
    }
}

OMPEnv::~OMPEnv()
{
    for (CharacterEnv* ptr: envs)
	delete ptr;
}
