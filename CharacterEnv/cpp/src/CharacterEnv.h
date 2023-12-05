#ifndef CHARACTER_ENV_H
#define CHARACTER_ENV_H

#include <random>
#include <Eigen/Core>
#include <dart/dart.hpp>
#include <dart/gui/osg/osg.hpp>
#include <dart/external/imgui/imgui.h>

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
        void run_viewer();

        dart::simulation::WorldPtr world = nullptr;
        dart::dynamics::SkeletonPtr skeleton = nullptr;

        osg::ref_ptr<dart::gui::osg::ImGuiViewer> viewer = nullptr;
        osg::ref_ptr<dart::gui::osg::WorldNode> worldNode = nullptr;

        Eigen::VectorXd action;
        Eigen::VectorXd state;
        double period = 0.0;
        double phase = 0.0;
        double reward = 0.0;
        bool done = false;
        bool enableRSI = false;
};

#endif
