#include <iostream>
#include <pthread.h>
#include <queue>
#include "OMPEnv.h"
#include "StdNormalDistVec.h"

using namespace std;
using namespace Eigen;

OMPEnv::OMPEnv(const char *cfgFilename, size_t num_threads): num_threads(num_threads)
{
    for (size_t i = 0; i < num_threads; ++i)
    {
	envs.push_back(CharacterEnv::makeEnv(cfgFilename));
        samplers.push_back(StdNormalDistVec(envs.back()->action.size()));
    }
}

void OMPEnv::reset()
{
    observations = MatrixXd(envs[0]->state.size(), envs.size());
    savedSamples.clear();
    for (size_t i = 0; i < num_threads; ++i)
    {
	envs[i]->reset();
        savedSamples.push_back(make_shared<Sample>(nullptr, 0, envs[i]));
    }

    observations = MatrixXd(savedSamples[0]->observation.size(), envs.size());
    for (size_t i = 0; i < savedSamples.size(); ++i)
    {
        observations.col(i) = savedSamples[i]->observation;
        //observations.col(i) = VectorXd::Ones(envs[i]->state.size()) * i;
    }
}


void OMPEnv::step()
{
    priority_queue<shared_ptr<Sample>, vector<shared_ptr<Sample>>, RewardCmp> queue;
#pragma omp for
    for (int i = savedSamples.size() - 1; i >= 0; --i)
    {
        size_t id = i % num_threads;
        shared_ptr<Sample> sample = savedSamples[i];

        for (size_t j = 0; j < numSample / savedSamples.size(); ++j)
        {
            envs[id]->skeleton->setPositions(sample->position);
            envs[id]->skeleton->setVelocities(sample->velocity);
            
            VectorXd x = samplers[id]();
            double logprob = StdNormalDistVec::logProbability(x);
            envs[id]->action = means.col(i) + (stds.col(i).array() * x.array()).matrix();
            //cout << "a " << envs[id]->action.transpose() << endl;

            envs[id]->step();
#pragma omp critical (queue_section)
            {
                queue.push(make_shared<Sample>(sample, logprob, envs[id]));
                if (queue.size() > numSave)
                    queue.pop();
            }
        }
    }

    savedSamples.clear();
    while (!queue.empty())
    {
        if (queue.top()->done == false)
        {
            savedSamples.push_back(queue.top());
            queue.pop();
        }
    }
    numObs = savedSamples.size();
    if (numObs > 0)
    {
        observations = MatrixXd(savedSamples[0]->observation.size(), numObs);
        for (size_t i = 0; i < savedSamples.size(); ++i)
            observations.col(i) = savedSamples[i]->observation;
    }
}

OMPEnv::~OMPEnv()
{
    for (CharacterEnv* ptr: envs)
	delete ptr;
}
