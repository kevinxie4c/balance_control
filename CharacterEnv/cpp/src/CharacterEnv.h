#ifndef CHARACTER_ENV_H
#define CHARACTER_ENV_H

#include <Eigen/Core>
#include <dart/dart.hpp>

class CharacterEnv
{
    public:
	CharacterEnv(const char *characterFilename, const char *poseFilename);
	void reset();
	void step();
	void updateState();
	double getTime();
	double getTimeStep();
	//void setTimeStep(double h);
	double cost();

	dart::simulation::WorldPtr world;
	dart::dynamics::SkeletonPtr skeleton;
	Eigen::VectorXd kp, kd;
	Eigen::VectorXd action;
	Eigen::VectorXd state;
	const double period = 0.0;
	double phase = 0.0;
	double reward = 0.0;
	size_t endEffectorIndices[4];
	std::vector<std::string> endEffectorNames{"Foot", "foot", "Hand", "hand"};
	std::vector<Eigen::VectorXd> positions;
	const static int mocapFPS = 120;
	const static int actionRate = 30;
	const static int forceRate = 600;
};

#endif
