#ifndef CHARACTER_ENV_H
#define CHARACTER_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>

class CharacterEnv
{
    public:
	CharacterEnv(const char *filename);
	void reset();
	void step();
	double getTime();

	dart::simulation::WorldPtr world;
	dart::dynamics::SkeletonPtr skeleton;
	double subStepSize = 0.001;
	double stepSize = 0.02;
	size_t subStepPerStep = stepSize / subStepSize;
	Eigen::VectorXd kp, kd;
	Eigen::VectorXd action;
	Eigen::VectorXd state;
	const double period = 0.0;
	double phase = 0.0;
	double reward = 0.0;
};

#endif
