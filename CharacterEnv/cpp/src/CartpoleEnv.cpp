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
    /*
    VectorXd force = (action.array() * scales.array()).matrix();
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    VectorXd ddq = skeleton->getAccelerations();
    MatrixXd M = skeleton->getMassMatrix();
    VectorXd C = skeleton->getCoriolisAndGravityForces();
    //cout << "M\n" << M << endl;
    //cout << "C\n" << C << endl;
    MatrixXd aS = MatrixXd::Zero(action.size(), action.size());
    aS.diagonal() = scales;
    MatrixXd sS = MatrixXd::Zero(state.size(), state.size());
    sS.diagonal() = (normalizerStd.array() + 1e-8).matrix().cwiseInverse();
    //cout << "sS\n" << sS << endl;
    MatrixXd dsdq(4, 2);
    dsdq << 1, 0,
            0, 1,
            0, 0,
            0, 0;
    MatrixXd J = aS * policyJacobian * sS * dsdq;
    //cout << "pJ\n" << policyJacobian << endl;
    //cout << "J\n" << J << endl;
    double h = 1.0 / actionRate;
    //MatrixXd K(2, 2);
    //MatrixXd D(2, 2);
    //K << 500, 0,
    //     0, 500;
    //D << 0.0, 0.0,
    //     0.0, 0.0;
    //force = -K * q - D * dq;
    //MatrixXd A = (M + h * h * K);
    MatrixXd A = (M - h * h * J);
    VectorXd b = M * dq + h * (force - C);
    //cout << "A\n" << A << endl;
    //cout << "b\n" << b << endl;
    VectorXd dq_n = A.inverse() * b;
    VectorXd q_n = q + h * dq_n;
    skeleton->setPositions(q_n);
    skeleton->setVelocities(dq_n);
    world->setTime(world->getTime() + h);
    */
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
