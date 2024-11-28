#include <iostream>
#include <pthread.h>
#include <queue>
#include <algorithm>
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
    buffer_size = 0;
}


/*
// use queue
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
            //double logprob = logps(0, i);
            //envs[id]->action = actions.col(i);
            //cout << "a " << envs[id]->action.transpose() << endl;

            envs[id]->step();
#pragma omp critical (queue_section)
            {
                shared_ptr<Sample> s = make_shared<Sample>(sample, logprob, envs[id]);
                if (s->done == false)
                    queue.push(s);
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
            //cout << queue.top()->reward << " ";
        }
        queue.pop();
    }
    //cout << endl;
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
    buffer_size += numObs;
}
*/

// use sort
void OMPEnv::step()
{
    for (int i = 0; i < savedSamples.back().size(); ++i)
        savedSamples.back()[i]->value = values(0, i);
    if (savedSamples.back().size() > numSave)
    {
        sort(savedSamples.back().begin(), savedSamples.back().end(), [](const shared_ptr<Sample>  &lhs, const shared_ptr<Sample> &rhs)
                {
                    //return lhs->value > rhs->value;
                    //return lhs->reward > rhs->reward;
                    return lhs->accReward > rhs->accReward;
                });
        savedSamples.back().resize(numSave);
        //cout << "sort" << endl;
    }
    vector<shared_ptr<Sample>> nextSamples;
#pragma omp for
    for (int i = 0; i < savedSamples.back().size(); ++i)
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
            //double logprob = logps(0, i);
            //envs[id]->action = actions.col(i);
            //cout << "a " << envs[id]->action.transpose() << endl;

            envs[id]->step();
#pragma omp critical (queue_section)
            {
                shared_ptr<Sample> s = make_shared<Sample>(sample, logprob, envs[id]);
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

void OMPEnv::trace_back()
{
    size_t n = 0;
    for (shared_ptr<Sample> &s: savedSamples.back())
        //s->retval = s->value;
        s->retval = 0;
    for (int i = savedSamples.size() - 1; i > 0; --i)
    {
        for (shared_ptr<Sample> &s: savedSamples[i])
        {
            double r = s->reward + gamma * s->retval;
            if (r > s->parent->retval)
            {
                s->parent->firstChild = s;
                s->parent->retval = r;
                //s->parent->delta = s->reward + gamma * s->value - s->parent->value;
                //s->parent->advantage = s->parent->delta + gamma * lam * s->advantage;
                s->delta = s->reward + gamma * s->value - s->parent->value;
                if (s->firstChild == nullptr)
                    s->advantage = s->delta;
                else
                    s->advantage = s->delta + gamma * lam * s->firstChild->advantage;
            }
        }
        n += savedSamples[i].size();
    }
    obs_buffer = Eigen::MatrixXd::Zero(envs[0]->state.size(), n);
    act_buffer = Eigen::MatrixXd::Zero(envs[0]->action.size(), n);
    //adv_buffer = Eigen::MatrixXd::Zero(1, n);
    //ret_buffer = Eigen::MatrixXd::Zero(1, n);
    //logp_buffer = Eigen::MatrixXd::Zero(1, n);
    adv_buffer = Eigen::MatrixXd::Zero(n, 1);
    ret_buffer = Eigen::MatrixXd::Zero(n, 1);
    logp_buffer = Eigen::MatrixXd::Zero(n, 1);
    int j = 0;
    for (int i = 1; i < savedSamples.size(); ++i)
    {
        for (shared_ptr<Sample> &s: savedSamples[i])
        {
            obs_buffer.col(j) = s->parent->observation;
            act_buffer.col(j) = s->action;
            //adv_buffer(0, j) = s->advantage;
            //ret_buffer(0, j) = s->parent->retval;
            //logp_buffer(0, j) = s->logprob;
            adv_buffer(j, 0) = s->advantage;
            ret_buffer(j, 0) = s->parent->retval;
            logp_buffer(j, 0) = s->logprob;
            ++j;
        }
    }
    
    avg_ret = 0;
    max_ret = numeric_limits<double>::min();
    max_len = savedSamples.size() - 1;
    shared_ptr<Sample> best = nullptr;
    for (shared_ptr<Sample> &s: savedSamples[0])
    {
        avg_ret += s->retval;
        if (s->retval > max_ret)
        {
            max_ret = s->retval;
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

OMPEnv::~OMPEnv()
{
    for (CharacterEnv* ptr: envs)
	delete ptr;
}
