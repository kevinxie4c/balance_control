#ifndef CHARACTER_ENV_H
#define CHARACTER_ENV_H

#include <random>
#include <Eigen/Core>
#include <dart/dart.hpp>

class CharacterEnv
{
    public:
	CharacterEnv(const char *cfgFilename);
	void reset();
	void step();
	void updateState();
	double getTime();
	double getTimeStep();
	//void setTimeStep(double h);
	double cost();
	void print_info();

	dart::simulation::WorldPtr world;
	dart::dynamics::SkeletonPtr skeleton, kin_skeleton, floor;
	Eigen::VectorXd kp, kd;
	Eigen::VectorXd action;
	Eigen::VectorXd state;
	Eigen::MatrixXd mkp, mkd;
	double period = 0.0;
	double phase = 0.0;
	double reward = 0.0;
	size_t endEffectorIndices[4];
	std::vector<std::string> endEffectorNames{"Foot", "foot", "Hand", "hand"};
	std::vector<Eigen::VectorXd> positions;
	std::vector<size_t> indices;
	std::vector<double> scales;

	int mocapFPS = 120;
	int actionRate = 30;
	int forceRate = 600;
	size_t frameIdx = 0;

	double w_p = 5, w_r = 3, w_e = 30, w_b = 10;

	bool enableRSI = false;

	std::mt19937 rng;
	std::uniform_int_distribution<size_t> uni_dist;
};

#endif
