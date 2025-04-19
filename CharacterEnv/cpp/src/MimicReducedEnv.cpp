#include <iostream>
#include <cmath>
#include <random>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include "SimCharacter.h"
#include "MimicReducedEnv.h"
#include "MathUtil.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

constexpr size_t n_ef = 4;

MimicReducedEnv::MimicReducedEnv(const char *cfgFilename)
{
    ifstream input(cfgFilename);
    if (input.fail())
        throw ios_base::failure(string("cannot open ") + cfgFilename);
    nlohmann::json json;
    input >> json;
    input.close();

    SimCharacter character(json["character"]);
    skeleton = character.skeleton;
    //skeleton->setGravity(Vector3d(0, -9.8, 0)); // y-axis is up in skeleton's coordinate
    world = dart::simulation::World::create();
    world->setGravity(Vector3d(0, 0, -9.8)); // z-axis is up in world coordinate
    world->addSkeleton(skeleton);
    world->setTimeStep(1.0 / forceRate);
    world->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

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

    for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i)
    {
        const BodyNode *bn = skeleton->getBodyNode(i);
        for (const string &s: fallenNodeNames)
        {
            if (bn->getName().rfind(s) != string::npos)
            {
                fallenNodeIndices.push_back(i);
                break;
            }
        }
    }

    for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i)
    {
        const BodyNode *bn = skeleton->getBodyNode(i);
        if (bn->getParentJoint()->getType() != "WeldJoint")
            bodyNodeIndices.push_back(i);
    }

    for (size_t i = 0; i < skeleton->getNumJoints(); ++i)
    {
        Joint *joint = skeleton->getJoint(i);
        if (joint->getType() != "WeldJoint")
            jointIndices.push_back(i);
    }

    refMotion = readVectorXdListFrom(json["ref_motion"]);

    floor = Skeleton::create("floor");
    BodyNodePtr body = floor->createJointAndBodyNodePair<WeldJoint>(nullptr).second;
    // Deprecated
    //body->setFrictionCoeff(json["friction_coeff"].get<double>());
    //body->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    body->setName("floor");
    double floor_length = 1e4;
    double floor_width = 1e2;
    double floor_height = 1;
    shared_ptr<BoxShape> box(new BoxShape(Vector3d(floor_length, floor_width, floor_height)));
    //auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect>(box);
    auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect, VisualAspect>(box);
    shapeNode->getDynamicsAspect()->setFrictionCoeff(json["friction_coeff"].get<double>());
    shapeNode->getDynamicsAspect()->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(0.0, 0.0, 1.0));
    Isometry3d tf(Isometry3d::Identity());
    tf.translation() = Vector3d(0.0, 0.0, json["floor_z"].get<double>() - floor_height / 2);
    body->getParentJoint()->setTransformFromParentBodyNode(tf);
    world->addSkeleton(floor);
    
    double cfm = dart::constraint::JointLimitConstraint::getConstraintForceMixing();
    double erp = dart::constraint::JointLimitConstraint::getErrorReductionParameter();
    cfm = 0.01;
    erp = 0.2;
    dart::constraint::JointLimitConstraint::setConstraintForceMixing(cfm);
    dart::constraint::JointLimitConstraint::setErrorReductionParameter(erp);
    dart::constraint::ContactConstraint::setConstraintForceMixing(cfm);
    dart::constraint::ContactConstraint::setErrorReductionParameter(erp);
    cout << "CFM: " << cfm << endl;
    cout << "ERP: " << erp << endl;

    kp = readVectorXdFrom(json["kp"]);
    kd = readVectorXdFrom(json["kd"]);
    mkp = MatrixXd::Zero(kp.size(), kp.size());
    mkd = MatrixXd::Zero(kd.size(), kd.size());
    mkp.diagonal() = kp;
    mkd.diagonal() = kd;

    scales = readVectorXdFrom(json["scales"]);

    if (json.contains("enableRSI"))
        enableRSI = json["enableRSI"].get<bool>();

    state = VectorXd(bodyNodeIndices.size() * 12 + 1);
    action = VectorXd(skeleton->getNumDofs() - 6);

    skeleton->setPositionLowerLimits(readVectorXdFrom(json["lower_limits"]));
    skeleton->setPositionUpperLimits(readVectorXdFrom(json["upper_limits"]));

    for (Joint *joint: skeleton->getJoints())
    {
        joint->setActuatorType(Joint::FORCE);
        joint->setLimitEnforcement(true);
    }

    skeleton->setSelfCollisionCheck(false);
    skeleton->setAdjacentBodyCheck(false);

    kin_skeleton = skeleton->cloneSkeleton();

    period = (double)refMotion.size() / mocapFPS;
    phaseShift = 0;
    
    generator = default_random_engine(chrono::system_clock::now().time_since_epoch().count());
    uni_dist = uniform_real_distribution<double>(0.0, 0.9);
    norm_dist = normal_distribution<double>(0.0, 1.0);

    w_r = 0; // TODO: need to correct the root position

    reset();
}

void MimicReducedEnv::reset()
{
    world->reset();
    size_t idx = 0;
    if (enableRSI)
        phaseShift = uni_dist(generator);
    double intPart;
    phase = modf(getTime() / period + phaseShift, &intPart);
    frameIdx = (size_t)round(phase * refMotion.size());
    if (frameIdx >= refMotion.size())
        frameIdx -= refMotion.size();

    VectorXd initPos = refMotion[frameIdx];
    skeleton->setPositions(initPos);

    VectorXd initVel = skeleton->getPositionDifferences(refMotion[frameIdx + 1], refMotion[frameIdx]) * mocapFPS;
    skeleton->setVelocities(initVel);

    updateState();
}

void MimicReducedEnv::step()
{
    fallen = false;
    double intPart;
    phase = modf(getTime() / period + phaseShift, &intPart);
    frameIdx = (size_t)round(phase * refMotion.size());
    if (frameIdx >= refMotion.size())
        frameIdx -= refMotion.size();

    const VectorXd &target = refMotion[frameIdx].tail(skeleton->getNumDofs() - 6);
    VectorXd ref(skeleton->getNumDofs());
    ref << 0, 0, 0, 0, 0, 0, target + (action.array().min(1).max(-1) * scales.array()).matrix();

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

        /*
        dart::collision::CollisionResult result = world->getLastCollisionResult();
        for (size_t j = 0; j < fallenNodeIndices.size(); ++j)
        {
            const BodyNode *bn = skeleton->getBodyNode(fallenNodeIndices[j]);
            if (result.inCollision(bn))
            {
                fallen = true;
                //cout << bn->getName() << endl;
                break;
            }
        }
        */
    }
    updateState();
}

void MimicReducedEnv::updateState()
{
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    VectorXd s(bodyNodeIndices.size() * 12);
    const BodyNode *root = skeleton->getRootBodyNode();
    Isometry3d T = root->getTransform();
    Vector3d trans, rot;
    setTransNRot(T, trans, rot);
    s.segment(0, 3) = trans;
    s.segment(3, 3) = rot;
    s.segment(6, 3) = root->getLinearVelocity();
    s.segment(9, 3) = root->getAngularVelocity();
    for (size_t j = 1; j < bodyNodeIndices.size(); ++j)
    {
        const BodyNode *bn = skeleton->getBodyNode(bodyNodeIndices[j]);
        T = bn->getTransform(root, root);
        setTransNRot(T, trans, rot);
        s.segment(j * 12, 3) = trans;
        s.segment(j * 12 + 3, 3) = rot;
        s.segment(j * 12 + 6, 3) = bn->getLinearVelocity(Frame::World(), root);
        s.segment(j * 12 + 9, 3) = bn->getAngularVelocity(Frame::World(), root);
    }
    state << s, phase;
    double c = cost();
    reward = 40 - c - 0.5 * pow(action.norm(), 2);
    done = c > 12 || fallen;
}

double MimicReducedEnv::cost()
{
    kin_skeleton->setPositions(refMotion[frameIdx]);
    const vector<Joint*> &joints = skeleton->getJoints();
    const vector<Joint*> &kin_joints = kin_skeleton->getJoints();
    size_t n = jointIndices.size() - 1;

    double err_p = 0;
    for (size_t i = 1; i < jointIndices.size(); ++i)
    {
        size_t j = jointIndices[i];
        const Joint *joint = joints[j];
        const Joint *kin_joint = kin_joints[j];
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
        size_t idx = endEffectorIndices[i];
        const BodyNode *node = skeleton->getBodyNode(idx);
        const BodyNode *kin_node = kin_skeleton->getBodyNode(idx);
        Eigen::Vector3d p = node->getCOM();
        Eigen::Vector3d pr = kin_node->getCOM();
        err_e += fabs(p.z() - pr.z());
    }
    err_e /= n_ef;

    double err_b = 0;
    Eigen::Vector3d COM = skeleton->getCOM();
    Eigen::Vector3d COMr = kin_skeleton->getCOM();
    for (size_t i = 0; i < n_ef; ++i)
    {
        size_t idx = endEffectorIndices[i];
        const BodyNode *node = skeleton->getBodyNode(idx);
        const BodyNode *kin_node = kin_skeleton->getBodyNode(idx);
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
    //cout << w_p * err_p + w_r * err_r + w_e * err_e + w_b * err_b << endl;
    return w_p * err_p + w_r * err_r + w_e * err_e + w_b * err_b;
}

void MimicReducedEnv::print_info()
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
