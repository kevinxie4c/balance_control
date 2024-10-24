#include <cmath>
#include <fstream>
#include "SimCharacter.h"
#include "CartpoleEnv.h"
#include "IOUtil.h"

#define SIM_MODE 2

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
    VectorXd force = (action.array() * scales.array()).matrix();
    double h = 1.0 / actionRate;
    VectorXd extFrc = Vector2d::Zero();
    double rem = fmod(world->getTime(), 10.0);
    if (rem >= 5.0 && rem < 5.2)
        extFrc[0] = 10;
    else if (rem >= 10.0 && rem < 10.2)
        extFrc[0] = -10;
    force += extFrc;
#if SIM_MODE == 1
        VectorXd q = skeleton->getPositions();
        VectorXd dq = skeleton->getVelocities();
        MatrixXd aS = MatrixXd::Zero(action.size(), action.size());
        aS.diagonal() = scales;
        MatrixXd sS = MatrixXd::Zero(state.size(), state.size());
        sS.diagonal() = (normalizerStd.array() + 1e-8).matrix().cwiseInverse();
        MatrixXd dsdq(4, 2);
        dsdq << 1, 0,
                0, 1,
                0, 0,
                0, 0;
        MatrixXd J = aS * policyJacobian * sS * dsdq;
#endif
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
#if SIM_MODE == 0
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
        skeleton->setVelocities(dq_n);
        skeleton->integratePositions(h);
        world->setTime(world->getTime() + h);
#elif SIM_MODE == 1
        VectorXd f = force + i * h * J * dq;
        skeleton->setForces(f);
        world->step();
#else
        skeleton->setForces(force);
        world->step();
#endif
    }
    updateState();
}

void CartpoleEnv::updateState()
{
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    state << q, dq;
    done = abs(q[0]) > 2.4 || abs(q[1]) > 0.20944;
    reward = done ? 0 : exp(-abs(q[0]));
}
