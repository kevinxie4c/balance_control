#ifndef OMP_ENV_H
#define OMP_ENV_H

#include <vector>
#include <memory>
#include "CharacterEnv.h"
#include "Sample.h"
#include "StdNormalDistVec.h"

class OMPEnv
{
    public:
	std::vector<CharacterEnv*> envs;

	OMPEnv(const char *cfgFilename, size_t num_threads);
	void reset();
	void step();
        void trace_back();
	~OMPEnv();

        size_t num_threads;
        // Eigen: column-major; MXNet: row-major
        Eigen::MatrixXd positions;
        Eigen::MatrixXd velocities;
        Eigen::MatrixXd means, stds;
        Eigen::MatrixXd observations;
        Eigen::MatrixXd actions;
        Eigen::MatrixXd logps;
        Eigen::MatrixXd values;
        size_t numSample = 400, numSave = 400;
        //size_t numSample = 1, numSave = 1;
        size_t numObs; // Why need this?
        std::vector<std::vector<std::shared_ptr<Sample>>> savedSamples;
        std::vector<StdNormalDistVec> samplers;

        Eigen::MatrixXd obs_buffer, act_buffer, adv_buffer, ret_buffer, logp_buffer;
        Eigen::MatrixXd best_traj;

        double gamma = 0.99;
        double lam = 0.97;
        double avg_ret = 0;
        double max_ret = 0;
        int max_len = 0;

        int buffer_size = 0;
};

#endif
