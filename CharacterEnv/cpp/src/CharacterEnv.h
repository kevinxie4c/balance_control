#ifndef CHARACTER_ENV_H
#define CHARACTER_ENV_H

#include <random>
#include <Eigen/Core>
#include <dart/dart.hpp>

class CharacterEnv
{
    public:
        static CharacterEnv* makeEnv(const char *cfgFilename);
        virtual ~CharacterEnv() = default;
        virtual void reset() = 0;
        virtual void step() = 0;
        double getTime();
        double getTimeStep();
        //void setTimeStep(double h);
        virtual void print_info();

        dart::simulation::WorldPtr world;
        dart::dynamics::SkeletonPtr skeleton;

        Eigen::VectorXd action;
        Eigen::VectorXd state;
        double period = 0.0;
        double phase = 0.0;
        double reward = 0.0;
        bool done = false;
        bool enableRSI = false;
};

#endif
