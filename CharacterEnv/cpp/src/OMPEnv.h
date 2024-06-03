#ifndef OMP_ENV_H
#define OMP_ENV_H

#include <vector>
#include <memory>
#include "CharacterEnv.h"

class OMPEnv
{
    public:
	std::vector<CharacterEnv*> envs;

	OMPEnv(const char *cfgFilename, size_t num_threads);
	void reset();
	void step();
	~OMPEnv();

        size_t num_threads;
        std::vector<Eigen::VectorXd> positions;
        std::vector<Eigen::VectorXd> velocities;
        std::vector<Eigen::VectorXd> actions;
};

#endif
