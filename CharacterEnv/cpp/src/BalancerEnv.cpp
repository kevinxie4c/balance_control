#include <cmath>
#include <fstream>
#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include "SimCharacter.h"
#include "BalancerEnv.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

BalancerEnv::BalancerEnv(const char *cfgFilename)
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
    world->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    //((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::DantzigBoxedLcpSolver()));
    ((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::PgsBoxedLcpSolver()));
    //skeleton->disableSelfCollisionCheck();

    skeleton->setPositionLowerLimits(lowerLimits = readVectorXdFrom(json["lower_limits"]));
    skeleton->setPositionUpperLimits(upperLimits = readVectorXdFrom(json["upper_limits"]));
    for (Joint *joint: skeleton->getJoints())
    {
        joint->setActuatorType(Joint::FORCE);
        joint->setLimitEnforcement(true);
    }

    scales = readVectorXdFrom(json["scales"]);
    state = VectorXd(skeleton->getNumDofs() * 2);
    action = VectorXd(skeleton->getNumDofs() - 1); // floor is not actuated

    reset();
}

void BalancerEnv::reset()
{
    world->reset();
    VectorXd rands = VectorXd::Random(skeleton->getNumDofs());
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    skeleton->setPositions(zeros);
    skeleton->setVelocities(zeros);
    updateState();
}

void BalancerEnv::step()
{
    double h = 1.0 / forceRate;
    VectorXd force = VectorXd::Zero(5);
    force.tail(4) = (action.array() * scales.array()).matrix();
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
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
        MatrixXd dsdq = MatrixXd::Zero(10, 4);
        for (size_t i = 0; i < 4; ++i)
            dsdq(i, i) = 1;
        MatrixXd J = MatrixXd::Zero(5, 5);
        J.bottomRows(4) = aS * policyJacobian * sS * dsdq;
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
        //VectorXd dq_n = A.inverse() * b;
        VectorXd dq_n = A.fullPivLu().solve(b);
        //VectorXd q_n = q + h * dq_n;
        //q_n = q_n.cwiseMax(lowerLimits);
        //q_n = q_n.cwiseMin(upperLimits);
        //dq_n = (q_n - q) / h;
        //skeleton->setPositions(q_n);
        skeleton->setVelocities(dq_n);
        world->getConstraintSolver()->solve();
        if (skeleton->isImpulseApplied())
        {
          skeleton->computeImpulseForwardDynamics();
          skeleton->setImpulseApplied(false);
        }
        skeleton->integratePositions(h);
        //cout << "v: " << skeleton->getVelocities().transpose() << endl;
        //cout << "p: " << skeleton->getPositions().transpose() << endl;
        world->setTime(world->getTime() + h);
    }
    /*
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        VectorXd force = VectorXd::Zero(5);
        force.tail(4) = (action.array() * scales.array()).matrix();
        skeleton->setForces(force);
        world->step();
    }
    */
    updateState();
}

void BalancerEnv::updateState()
{
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    state << q, dq;
    done = abs(q[0]) > 0.5;
    reward = -action.norm() + (done ? 0 : 100);
}
