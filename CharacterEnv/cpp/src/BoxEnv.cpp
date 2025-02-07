#include <cmath>
#include <fstream>
#include <chrono>
#include "SimCharacter.h"
#include "BoxEnv.h"
#include "IOUtil.h"

#define SIM_MODE 2

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

BoxEnv::BoxEnv(const char *cfgFilename)
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
    //world->setGravity(Vector3d(0, -9.8, 0));
    world->setGravity(Vector3d(0, 0, 0));
    world->addSkeleton(skeleton);
    world->setTimeStep(1.0 / forceRate);
    //((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::DantzigBoxedLcpSolver()));
    ((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::PgsBoxedLcpSolver()));

    scales = readVectorXdFrom(json["scales"]);
    state = VectorXd(skeleton->getNumDofs() * 2);
    action = VectorXd(skeleton->getNumDofs());

    generator = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
    distribution = std::uniform_real_distribution<double>(-10.0, 10.0);

    //fh_pos.open("pos.txt");
    //cout << "fh_pos.open" << endl;
    ifstream fh_w("weight.txt");
    fh_w >> w_a;
    fh_w.close();
    cout << "w_a: " << w_a << endl;

    reset();
}

void BoxEnv::reset()
{
    world->reset();
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    skeleton->setPositions(zeros);
    skeleton->setVelocities(zeros);
    updateState();
}

void BoxEnv::step()
{
    VectorXd force = (action.array() * scales.array()).matrix();
    double h = 1.0 / actionRate;
    Vector3d extFrc = Vector3d::Zero();
    Vector3d offset = Vector3d::Zero();
    double rem = fmod(world->getTime(), 2.0);
    if (rem >= 1.0 && rem < 1.2)
        extFrc[0] = distribution(generator);
    //double rem = fmod(world->getTime(), 20.0);
    //if (rem >= 5.0 && rem < 5.2)
    //    extFrc[0] = -20;
    //else if (rem >= 15.0 && rem < 15.2)
    //    extFrc[0] = 20;
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        //fh_pos << skeleton->getPositions().transpose() << endl;
        skeleton->getBodyNode(0)->addExtForce(extFrc, offset);
        skeleton->setForces(force);
        world->step();
    }
    updateState();
}

void BoxEnv::updateState()
{
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    state << q, dq;
    done = abs(q[0]) > 2;
    reward = done ? 0 : exp(-abs(q[0])) - w_a * (action.norm() * action.norm());
}

BoxEnv::~BoxEnv()
{
    //fh_pos.close();
    //cout << "fh_pos.close" << endl;
}
