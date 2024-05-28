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


void ParallelEnv::step(size_t id)
{
}
