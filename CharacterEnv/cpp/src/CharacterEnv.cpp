#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "SimCharacter.h"
#include "CharacterEnv.h"

using namespace Eigen;
using namespace dart::dynamics;

CharacterEnv::CharacterEnv(const char *filename)
{
    SimCharacter character(filename);
    skeleton = character.skeleton;
    skeleton->setGravity(Vector3d(0, -9.8, 0));
    world = dart::simulation::World::create();
    world->setTimeStep(subStepSize);
    reset();
}

void CharacterEnv::reset()
{
    world->reset();
    phase = 0.0;
    Eigen::VectorXd q = skeleton->getPositions();
    Eigen::VectorXd dq = skeleton->getVelocities();
    double intPart;
    phase = modf(getTime() / period, &intPart);
    //VectorXd s(skeleton->getNumBodyNodes() * 6);
    //const BodyNode *root = skeleton->getRootBodyNode();
    //size_t i = 0;
    //for (BodyNodePtr bn: skeleton->getBodyNodes())
    //{
    //    s.segment(i * 6, 3) = bn->getLinearVelocity(Frame::World(), root);
    //    s.segment(i * 6 + 3, 3) = bn->getAngularVelocity(Frame::World(), root);
    //}
    //state << q, dq, phase;
    //reward = M_PI / 4 - (fabs(sin(phase * 2 * M_PI) - q[0]) + fabs(q[1]));
}

void CharacterEnv::step()
{
    Eigen::MatrixXd kp = Eigen::MatrixXd::Zero(2, 2), kd = Eigen::MatrixXd::Zero(2, 2);
    kp.diagonal() = this->kp;
    kd.diagonal() = this->kd;
    double intPart;
    phase = modf(getTime() / period, &intPart);
    for (size_t i = 0; i < subStepPerStep; ++i)
    {
	// stable PD
	Eigen::VectorXd q = skeleton->getPositions();
	Eigen::VectorXd dq = skeleton->getVelocities();
	Eigen::Vector2d target{sin(phase * 2 * M_PI), 0};
	Eigen::Vector2d ref = target + action;
	Eigen::MatrixXd invM = (skeleton->getMassMatrix() + kd * skeleton->getTimeStep()).inverse();
	Eigen::VectorXd p = -kp * skeleton->getPositionDifferences(q + dq * skeleton->getTimeStep(), ref);
	Eigen::VectorXd d = -kd * dq;
	Eigen::VectorXd qddot = invM * (-skeleton->getCoriolisAndGravityForces() + p + d + skeleton->getConstraintForces());
	Eigen::VectorXd force = p + d - kd * qddot * world->getTimeStep();
	skeleton->setForces(force);
	world->step();
    }
    Eigen::VectorXd q = skeleton->getPositions();
    Eigen::VectorXd dq = skeleton->getVelocities();
    phase = modf(getTime() / period, &intPart);
    state << q, dq, phase;
    reward = M_PI / 4 - (fabs(sin(phase * 2 * M_PI) - q[0]) + fabs(q[1]));
}

double CharacterEnv::getTime()
{
    return world->getTime();
}
