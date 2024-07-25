#include <cmath>
#include <fstream>
#include "SimCharacter.h"
#include "CartpoleEnv.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

CartpoleEnv::CartpoleEnv(const char *cfgFilename)
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

void CartpoleEnv::reset()
{
    world->reset();
    VectorXd rands = VectorXd::Random(skeleton->getNumDofs());
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    skeleton->setPositions(0.05 * rands);
    skeleton->setVelocities(zeros);
    updateState();
}

void CartpoleEnv::step()
{
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        VectorXd force = (action.array() * scales.array()).matrix();
        skeleton->setForces(force);
        world->step();
    }
    updateState();
}

void CartpoleEnv::updateState()
{
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    state << q, dq;
    done = abs(q[0]) > 2.4 || abs(q[1]) > 0.20944;
    reward = done ? 0 : 1;
}
