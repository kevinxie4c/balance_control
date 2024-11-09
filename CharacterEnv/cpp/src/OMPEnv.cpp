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
    savedSamples.clear();
    savedSamples.push_back(std::vector<std::shared_ptr<Sample>>());
    for (size_t i = 0; i < num_threads; ++i)
    {
	envs[i]->reset();
        savedSamples.back().push_back(make_shared<Sample>(nullptr, 0, envs[i]));
    }

    observations = MatrixXd(savedSamples.back()[0]->observation.size(), envs.size());
    for (size_t i = 0; i < savedSamples.back().size(); ++i)
    {
        observations.col(i) = savedSamples.back()[i]->observation;
        //observations.col(i) = VectorXd::Ones(envs[i]->state.size()) * i;
    }
}


void OMPEnv::step()
{
    priority_queue<shared_ptr<Sample>, vector<shared_ptr<Sample>>, RewardCmp> queue;
#pragma omp for
    for (int i = savedSamples.back().size() - 1; i >= 0; --i)
    {
        size_t id = i % num_threads;
        shared_ptr<Sample> sample = savedSamples.back()[i];
        sample->value = values(0, i);

        for (size_t j = 0; j < numSample / savedSamples.back().size(); ++j)
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

    savedSamples.push_back(std::vector<std::shared_ptr<Sample>>());
    while (!queue.empty())
    {
        if (queue.top()->done == false)
        {
            savedSamples.back().push_back(queue.top());
        }
        queue.pop();
    }
    numObs = savedSamples.back().size();
    //cout << "numObs: " << numObs << endl;
    observations = MatrixXd(savedSamples.back()[0]->observation.size(), numObs);
    if (numObs > 0)
    {
        for (size_t i = 0; i < numObs; ++i)
            observations.col(i) = savedSamples.back()[i]->observation;
    }
}

void OMPEnv::trace_back()
{
    size_t n = 0;
    for (shared_ptr<Sample> &s: savedSamples.back())
        s->retval = s->value;
    for (int i = savedSamples.size() - 1; i > 0; --i)
    {
        for (shared_ptr<Sample> &s: savedSamples[i])
        {
            double r = s->reward + gamma * s->retval;
            if (r > s->parent->retval)
            {
                s->parent->retval = r;
                //s->parent->delta = s->reward + gamma * s->value - s->parent->value;
                //s->parent->advantage = s->parent->delta + gamma * lam * s->advantage;
                s->delta = s->reward + gamma * s->value - s->parent->value;
                s->advantage = s->delta + gamma * lam * s->advantage;
            }
        }
        n += savedSamples[i].size();
    }
    obs_buffer = Eigen::MatrixXd::Zero(envs[0]->state.size(), n);
    act_buffer = Eigen::MatrixXd::Zero(envs[0]->action.size(), n);
    adv_buffer = Eigen::MatrixXd::Zero(1, n);
    ret_buffer = Eigen::MatrixXd::Zero(1, n);
    logp_buffer = Eigen::MatrixXd::Zero(1, n);
    int j = 0;
    for (int i = 1; i < savedSamples.size(); ++i)
    {
        for (shared_ptr<Sample> &s: savedSamples[i])
        {
            obs_buffer.col(j) = s->parent->observation;
            act_buffer.col(j) = s->action;
            adv_buffer(0, j) = s->advantage;
            ret_buffer(0, j) = s->parent->retval;
            logp_buffer(0, j) = s->logprob;
            ++j;
        }
    }
    
    double avg_ret = 0;
    for (shared_ptr<Sample> &s: savedSamples[0])
        avg_ret += s->retval;
    avg_ret /= savedSamples[0].size();
    cout << avg_ret << endl;
}

OMPEnv::~OMPEnv()
{
    for (CharacterEnv* ptr: envs)
	delete ptr;
}
