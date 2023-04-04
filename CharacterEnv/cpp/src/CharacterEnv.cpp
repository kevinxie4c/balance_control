#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "SimCharacter.h"
#include "CharacterEnv.h"
#include "MathUtil.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

CharacterEnv::CharacterEnv(const char *characterFilename, const char *poseFilename)
{
    SimCharacter character(characterFilename);
    skeleton = character.skeleton;
    skeleton->setGravity(Vector3d(0, -9.8, 0));
    world = dart::simulation::World::create();
    world->addSkeleton(skeleton);
    world->setTimeStep(1.0 / forceRate);
    state = VectorXd(skeleton->getNumBodyNodes() * 12 + 1);
    cout << "BodyNode:" << endl;
    size_t j = 0;
    for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i)
    {
	const BodyNode *bn = skeleton->getBodyNode(i);
	cout << bn->getName() << endl;
	for (const string &s: endEffectorNames)
	{
	    if (bn->getName().rfind(s) != string::npos)
	    {
		endEffectorIndices[j++] = i;
		break;
	    }
	}
    }
    cout << "Dofs:" << endl;
    for (const DegreeOfFreedom *dof: skeleton->getDofs())
	cout << dof->getName() << endl;
    cout << "endEffectorIndices:" << std::endl;
    for (size_t i = 0; i < 4; ++i)
	cout << endEffectorIndices[i] << endl;
    positions = readVectorXdListFrom(poseFilename);
    reset();
}

void CharacterEnv::reset()
{
    world->reset();
    skeleton->setPositions(positions[0]);
    phase = 0.0;
    updateState();
}

void CharacterEnv::step()
{
    //Eigen::MatrixXd kp = Eigen::MatrixXd::Zero(2, 2), kd = Eigen::MatrixXd::Zero(2, 2);
    //kp.diagonal() = this->kp;
    //kd.diagonal() = this->kd;
    double intPart;
    phase = modf(getTime() / period, &intPart);
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
	// stable PD
	//Eigen::VectorXd q = skeleton->getPositions();
	//Eigen::VectorXd dq = skeleton->getVelocities();
	//Eigen::Vector2d target{sin(phase * 2 * M_PI), 0};
	//Eigen::Vector2d ref = target + action;
	//Eigen::MatrixXd invM = (skeleton->getMassMatrix() + kd * skeleton->getTimeStep()).inverse();
	//Eigen::VectorXd p = -kp * skeleton->getPositionDifferences(q + dq * skeleton->getTimeStep(), ref);
	//Eigen::VectorXd d = -kd * dq;
	//Eigen::VectorXd qddot = invM * (-skeleton->getCoriolisAndGravityForces() + p + d + skeleton->getConstraintForces());
	//Eigen::VectorXd force = p + d - kd * qddot * world->getTimeStep();
	//skeleton->setForces(force);
	world->step();
    }
    updateState();
}

void CharacterEnv::updateState()
{
    Eigen::VectorXd q = skeleton->getPositions();
    Eigen::VectorXd dq = skeleton->getVelocities();
    double intPart;
    phase = modf(getTime() / period, &intPart);
    VectorXd s(skeleton->getNumBodyNodes() * 12);
    const BodyNode *root = skeleton->getRootBodyNode();
    Isometry3d T = root->getTransform();
    Vector3d trans, rot;
    setTransNRot(T, trans, rot);
    s.segment(0, 3) = trans;
    s.segment(3, 3) = rot;
    s.segment(6, 3) = root->getLinearVelocity();
    s.segment(9, 3) = root->getAngularVelocity();
    for (size_t i = 1; i < skeleton->getNumBodyNodes(); ++i)
    {
	const BodyNode *bn = skeleton->getBodyNode(i);
	T = bn->getTransform(root, root);
	setTransNRot(T, trans, rot);
        s.segment(i * 12, 3) = trans;
        s.segment(i * 12 + 3, 3) = rot;
        s.segment(i * 12 + 6, 3) = bn->getLinearVelocity(Frame::World(), root);
        s.segment(i * 12 + 9, 3) = bn->getAngularVelocity(Frame::World(), root);
    }
    state << s, phase;
    reward = 1;
}

double CharacterEnv::getTime()
{
    return world->getTime();
}

double  CharacterEnv::getTimeStep()
{
    return world->getTimeStep();
}

/*
void CharacterEnv::setTimeStep(double h)
{
    world->setTimeStep(h);
}
*/

double CharacterEnv::cost()
{
    std::vector<BodyNode*> nodes = skeleton->getBodyNodes();
    size_t n = nodes.size();
    for (size_t i = 0; i < n; ++i)
    {
	const BodyNode *bn = nodes[i];
    }
}
