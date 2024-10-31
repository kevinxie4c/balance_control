#include <cmath>
#include <fstream>
#include "SimCharacter.h"
#include "RodEnv.h"
#include "IOUtil.h"

#define SIM_MODE 2

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

RodEnv::RodEnv(const char *cfgFilename)
{
    ifstream input(cfgFilename);
    if (input.fail())
        throw ios_base::failure(string("cannot open ") + cfgFilename);
    nlohmann::json json;
    input >> json;
    input.close();

    SimCharacter character(json["character"]);
    skeleton = character.skeleton;
    world = dart::simulation::World::create();
    world->setGravity(Vector3d(0, -9.8, 0));
    world->addSkeleton(skeleton);
    world->setTimeStep(1.0 / forceRate);
    //((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::DantzigBoxedLcpSolver()));
    ((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::PgsBoxedLcpSolver()));

    scales = readVectorXdFrom(json["scales"]);
    state = VectorXd(skeleton->getNumDofs() * 2);
    action = VectorXd(skeleton->getNumDofs());

    reset();
}

void RodEnv::reset()
{
    world->reset();
    VectorXd rands = VectorXd::Random(skeleton->getNumDofs());
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    skeleton->setPositions(0.05 * rands);
    skeleton->setVelocities(zeros);
    updateState();
}

void RodEnv::step()
{
    VectorXd force = (action.array() * scales.array()).matrix();
    double h = 1.0 / actionRate;
    Vector3d extFrc = Vector3d::Zero();
    Vector3d offset = Vector3d::Zero();
    offset[1] = 0.5;
    double rem = fmod(world->getTime(), 20.0);
    if (rem >= 5.0 && rem < 5.2)
        extFrc[0] = 1;
    else if (rem >= 15.0 && rem < 15.2)
        extFrc[0] = -1;
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        skeleton->getBodyNode(1)->addExtForce(extFrc, offset);
        skeleton->setForces(force);
        world->step();
    }
    updateState();
}

void RodEnv::updateState()
{
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    state << q, dq;
    done = abs(q[0]) > 0.523599;
    reward = done ? 0 : exp(-abs(q[0]));
}
