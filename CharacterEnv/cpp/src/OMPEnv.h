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
	~OMPEnv();

        size_t num_threads;
        // Eigen: column-major; MXNet: row-major
        Eigen::MatrixXd positions;
        Eigen::MatrixXd velocities;
        Eigen::MatrixXd means, stds;
        Eigen::MatrixXd observations;
        size_t numSample = 20, numSave = 4;
        size_t numObs; // Why need this?
        std::vector<std::shared_ptr<Sample>> savedSamples;
        std::vector<StdNormalDistVec> samplers;
};

#endif
