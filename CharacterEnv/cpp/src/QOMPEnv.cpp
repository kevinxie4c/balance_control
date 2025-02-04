#include <iostream>
#include <pthread.h>
#include <queue>
#include <algorithm>
#include "QOMPEnv.h"
#include "StdNormalDistVec.h"

using namespace std;
using namespace Eigen;

QOMPEnv::QOMPEnv(const char *cfgFilename, size_t num_threads): num_threads(num_threads)
{
    for (size_t i = 0; i < num_threads; ++i)
    {
	envs.push_back(CharacterEnv::makeEnv(cfgFilename));
        samplers.push_back(StdNormalDistVec(envs.back()->action.size()));
    }
}

void QOMPEnv::reset()
{
    savedSamples.clear();
    savedSamples.push_back(std::vector<std::shared_ptr<QSample>>());
    for (size_t i = 0; i < num_threads; ++i)
    {
	envs[i]->reset();
        savedSamples.back().push_back(make_shared<QSample>(nullptr, envs[i]));
    }

    observations = MatrixXd(savedSamples.back()[0]->observation.size(), envs.size());
    for (size_t i = 0; i < savedSamples.back().size(); ++i)
    {
        observations.col(i) = savedSamples.back()[i]->observation;
        //observations.col(i) = VectorXd::Ones(envs[i]->state.size()) * i;
    }
    buffer_size = 0;
}



// use sort
void QOMPEnv::step()
{
    for (int i = 0; i < savedSamples.back().size(); ++i)
        savedSamples.back()[i]->value = values(0, i);
    if (savedSamples.back().size() > numSave)
    {
        sort(savedSamples.back().begin(), savedSamples.back().end(), [](const shared_ptr<QSample>  &lhs, const shared_ptr<QSample> &rhs)
                {
                    //return lhs->value > rhs->value;
                    //return lhs->reward > rhs->reward;
                    return lhs->accReward > rhs->accReward;
                    //return lhs->accReward + lhs->value > rhs->accReward + rhs->value;
                });
        savedSamples.back().resize(numSave);
        //cout << "sort" << endl;
    }
    vector<shared_ptr<QSample>> nextSamples;
#pragma omp for
    for (int i = 0; i < savedSamples.back().size(); ++i)
    {
        size_t id = i % num_threads;
        shared_ptr<QSample> sample = savedSamples.back()[i];
        //sample->value = values(0, i); // Should I remove this?

        for (size_t j = 0; j < numSample / savedSamples.back().size(); ++j)
        {
            envs[id]->skeleton->setPositions(sample->position);
            envs[id]->skeleton->setVelocities(sample->velocity);
            
            VectorXd x = samplers[id]();
            double logprob = StdNormalDistVec::logProbability(x);
            envs[id]->action = means.col(i) + (stds.col(i).array() * x.array()).matrix();
            //envs[id]->action = actions.col(i);
            //cout << "a " << envs[id]->action.transpose() << endl;

            envs[id]->step();
#pragma omp critical (queue_section)
            {
                shared_ptr<QSample> s = make_shared<QSample>(sample, envs[id]);
                if (s->done == false)
                    nextSamples.push_back(s);
            }
        }
    }

    savedSamples.push_back(nextSamples);
    numObs = savedSamples.back().size();
    //cout << "numObs: " << numObs << endl;
    observations = MatrixXd(envs[0]->state.size(), numObs);
    if (numObs > 0)
    {
        for (size_t i = 0; i < numObs; ++i)
            observations.col(i) = savedSamples.back()[i]->observation;
    }
    else
    {
        savedSamples.pop_back();
    }
    buffer_size += numObs > numSave ? numSave : numObs;
}

void QOMPEnv::trace_back()
{
    size_t n = 0;
    for (shared_ptr<QSample> &s: savedSamples.back())
    {
        s->retval = 0;
        //s->maxval = s->value;
        s->maxval = 0;
    }
    for (int i = savedSamples.size() - 1; i > 0; --i)
    {
        for (shared_ptr<QSample> &s: savedSamples[i])
        {
            s->retval = s->reward + gamma * s->maxval;
            if (s->retval > s->parent->maxval)
            {
                s->parent->firstChild = s;
                s->parent->maxval = s->retval;
            }
        }
        n += savedSamples[i].size();
    }
    obs_buffer = Eigen::MatrixXd::Zero(envs[0]->state.size(), n);
    act_buffer = Eigen::MatrixXd::Zero(envs[0]->action.size(), n);
    ret_buffer = Eigen::MatrixXd::Zero(n, 1);
    int j = 0;
    for (int i = 1; i < savedSamples.size(); ++i)
    {
        for (shared_ptr<QSample> &s: savedSamples[i])
        {
            obs_buffer.col(j) = s->parent->observation;
            act_buffer.col(j) = s->action;
            ret_buffer(j, 0) = s->retval;
            ++j;
        }
    }
    
    avg_ret = 0;
    max_ret = numeric_limits<double>::min();
    max_len = savedSamples.size() - 1;
    shared_ptr<QSample> best = nullptr;
    for (shared_ptr<QSample> &s: savedSamples[0])
    {
        avg_ret += s->maxval;
        if (s->maxval > max_ret)
        {
            max_ret = s->maxval;
            best = s;
        }
    }
    std::vector<Eigen::VectorXd> bestTraj;
    while (best != nullptr)
    {
        bestTraj.push_back(best->position);
        best = best->firstChild;
    }
    best_traj = Eigen::MatrixXd(bestTraj[0].size(), bestTraj.size());
    for (int i = 0; i < bestTraj.size(); ++i)
        best_traj.col(i) = bestTraj[i];
    avg_ret /= savedSamples[0].size();
}

QOMPEnv::~QOMPEnv()
{
    for (CharacterEnv* ptr: envs)
	delete ptr;
}
