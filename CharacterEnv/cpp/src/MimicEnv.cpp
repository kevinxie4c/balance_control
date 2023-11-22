#include <iostream>
#include <cmath>
#include <random>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include "SimCharacter.h"
#include "MimicEnv.h"
#include "MathUtil.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

constexpr size_t n_ef = 4;

MimicEnv::MimicEnv(const char *cfgFilename)
{
    ifstream input(cfgFilename);
    if (input.fail())
        throw ios_base::failure(string("cannot open ") + cfgFilename);
    nlohmann::json json;
    input >> json;
    input.close();

    SimCharacter character(json["character"]);
    skeleton = character.skeleton;
    skeleton->setGravity(Vector3d(0, -9.8, 0));
    world = dart::simulation::World::create();
    world->addSkeleton(skeleton);
    world->setTimeStep(1.0 / forceRate);
    world->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
    state = VectorXd(skeleton->getNumBodyNodes() * 12 + 1);
    size_t j = 0;
    for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i)
    {
        const BodyNode *bn = skeleton->getBodyNode(i);
        for (const string &s: endEffectorNames)
        {
            if (bn->getName().rfind(s) != string::npos)
            {
                endEffectorIndices[j++] = i;
                break;
            }
        }
    }
    positions = readVectorXdListFrom(json["poses"]);

    floor = Skeleton::create("floor");
    BodyNodePtr body = floor->createJointAndBodyNodePair<WeldJoint>(nullptr).second;
    // Deprecated
    //body->setFrictionCoeff(json["friction_coeff"].get<double>());
    //body->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    body->setName("floor");
    double floor_width = 1e8;
    double floor_height = 0.01;
    shared_ptr<BoxShape> box(new BoxShape(Vector3d(floor_width, floor_width, floor_height)));
    auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect>(box);
    shapeNode->getDynamicsAspect()->setFrictionCoeff(json["friction_coeff"].get<double>());
    shapeNode->getDynamicsAspect()->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    Isometry3d tf(Isometry3d::Identity());
    tf.translation() = Vector3d(0.0, 0.0, json["floor_z"].get<double>() - floor_height / 2);
    body->getParentJoint()->setTransformFromParentBodyNode(tf);
    world->addSkeleton(floor);

    kp = readVectorXdFrom(json["kp"]);
    kd = readVectorXdFrom(json["kd"]);
    mkp = MatrixXd::Zero(kp.size(), kp.size());
    mkd = MatrixXd::Zero(kd.size(), kd.size());
    mkp.diagonal() = kp;
    mkd.diagonal() = kd;

    scales = readListFrom<double>(json["scales"]);

    if (json.contains("enableRSI"))
        enableRSI = json["enableRSI"].get<bool>();

    action = VectorXd(skeleton->getNumDofs());
    period = (double)positions.size() / mocapFPS;

    kin_skeleton = skeleton->cloneSkeleton();

    std::random_device rd;
    rng = std::mt19937(rd());
    uni_dist = std::uniform_int_distribution<size_t>(0, positions.size() - 2);

    reset();
}

void MimicEnv::reset()
{
    world->reset();
    size_t idx = 0;
    if (enableRSI)
        idx = uni_dist(rng);
    skeleton->setPositions(positions[idx]);
    VectorXd vel = skeleton->getPositionDifferences(positions[idx + 1], positions[idx]) * mocapFPS;
    skeleton->setVelocities(vel);
    phase = (double)idx / positions.size();
    world->setTime((double)idx / mocapFPS);
    updateState();
}

void MimicEnv::step()
{
    double intPart;
    phase = modf(getTime() / period, &intPart);
    frameIdx = (size_t)round(phase * positions.size());
    if (frameIdx >= positions.size())
        frameIdx -= positions.size();

    const VectorXd &target = positions[frameIdx];
    VectorXd offset = VectorXd::Zero(target.size());
    for (size_t i = 0; i < action.size(); ++i)
    {
        offset[i] += action[i] * scales[i];
    }
    VectorXd ref = target + offset;

    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        // stable PD
        VectorXd q = skeleton->getPositions();
        VectorXd dq = skeleton->getVelocities();
        MatrixXd invM = (skeleton->getMassMatrix() + mkd * skeleton->getTimeStep()).inverse();
        VectorXd p = -mkp * skeleton->getPositionDifferences(q + dq * skeleton->getTimeStep(), ref);
        VectorXd d = -mkd * dq;
        VectorXd qddot = invM * (-skeleton->getCoriolisAndGravityForces() + p + d + skeleton->getConstraintForces());
        VectorXd force = p + d - mkd * qddot * world->getTimeStep();

        // PD
        /*
        VectorXd q = skeleton->getPositions();
        VectorXd dq = skeleton->getVelocities();
        VectorXd p = -mkp * skeleton->getPositionDifferences(q, ref);
        VectorXd d = -mkd * dq;
        VectorXd force = p + d;
        */

        skeleton->setForces(force);
        world->step();
    }
    updateState();
}

void MimicEnv::updateState()
{
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
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
    reward = 20 - cost();
}

double MimicEnv::cost()
{
    kin_skeleton->setPositions(positions[frameIdx]);
    const vector<Joint*> &joints = skeleton->getJoints();
    const vector<Joint*> &kin_joints = kin_skeleton->getJoints();
    size_t n = joints.size();

    double err_p = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const Joint *joint = joints[i];
        const Joint *kin_joint = kin_joints[i];
        //err_p += joint->getPositionDifferences(joint->getPositions(), kin_joint->getPositions()).norm() + 0.1 * (joint->getVelocities() - kin_joint->getVelocities()).norm();
        err_p += joint->getPositionDifferences(joint->getPositions(), kin_joint->getPositions()).norm();
    }
    err_p /= n;

    double err_r = 0;
    const Joint *joint = skeleton->getRootJoint();
    const Joint *kin_joint = kin_skeleton->getRootJoint();
    //err_r += joint->getPositionDifferences(joint->getPositions(), kin_joint->getPositions()).norm() + 0.1 * (joint->getVelocities() - kin_joint->getVelocities()).norm();
    err_r += joint->getPositionDifferences(joint->getPositions(), kin_joint->getPositions()).norm();

    double err_e = 0;
    for (size_t i = 0; i < n_ef; ++i)
    {
        const BodyNode *node = skeleton->getBodyNode(i);
        const BodyNode *kin_node = kin_skeleton->getBodyNode(i);
        Eigen::Vector3d p = node->getCOM();
        Eigen::Vector3d pr = kin_node->getCOM();
        err_e += fabs(p.y() - pr.y());
    }
    err_e /= n_ef;

    double err_b = 0;
    Eigen::Vector3d COM = skeleton->getCOM();
    Eigen::Vector3d COMr = kin_skeleton->getCOM();
    for (size_t i = 0; i < n_ef; ++i)
    {
        const BodyNode *node = skeleton->getBodyNode(i);
        const BodyNode *kin_node = kin_skeleton->getBodyNode(i);
        Eigen::Vector3d p = node->getCOM();
        Eigen::Vector3d pr = kin_node->getCOM();
        Eigen::Vector3d rci = COM - p;
        rci.z() = 0;
        Eigen::Vector3d rci_r = COMr - pr;
        rci_r.z() = 0;
        err_b += (rci - rci_r).norm();
    }
    constexpr double h = 1.6;
    err_b /= h * n_ef;
    //err_b += (skeleton->getCOMLinearVelocity() - kin_skeleton->getCOMLinearVelocity()).norm() * 0.1;

    //cout << "cost: " << " " << err_p << " " << err_r << " " << err_e << " " << err_b << endl;
    return w_p * err_p + w_r * err_r + w_e * err_e + w_b * err_b;
}

void MimicEnv::print_info()
{
    cout << "BodyNode:" << endl;
    for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i)
    {
        const BodyNode *bn = skeleton->getBodyNode(i);
        cout << bn->getName() << endl;
    }
    cout << "Dofs:" << endl;
    for (const DegreeOfFreedom *dof: skeleton->getDofs())
        cout << dof->getName() << endl;
    cout << "endEffectorIndices:" << endl;
    for (size_t i = 0; i < n_ef; ++i)
        cout << endEffectorIndices[i] << endl;
}
