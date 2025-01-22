#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/collision/ode/OdeCollisionDetector.hpp>
#include "SimCharacter.h"
#include "SimpleEnv.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

SimpleEnv::SimpleEnv(const char *cfgFilename)
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
    //skeleton->setGravity(Vector3d(0, -9.8, 0)); // Why it doesn't work?
    world->setGravity(Vector3d(0, -9.8, 0));
    //world->setGravity(Vector3d(0, 0, 0));
    world->addSkeleton(skeleton);
    world->setTimeStep(1.0 / forceRate);
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::FCLCollisionDetector::create());
    world->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::OdeCollisionDetector::create());
    //((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::DantzigBoxedLcpSolver()));
    ((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::PgsBoxedLcpSolver()));

    for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i)
    {
        BodyNode *bn = skeleton->getBodyNode(i);
        for (ShapeNode *sn: bn->getShapeNodes())
        {
            sn->getDynamicsAspect()->setFrictionCoeff(json["friction_coeff"].get<double>());
            sn->getDynamicsAspect()->setRestitutionCoeff(json["restitution_coeff"].get<double>());
        }
    }

    floor = Skeleton::create("floor");
    BodyNodePtr body = floor->createJointAndBodyNodePair<WeldJoint>(nullptr).second;
    // Deprecated
    //body->setFrictionCoeff(json["friction_coeff"].get<double>());
    //body->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    body->setName("floor");
    double floor_width = 1e2;
    double floor_height = 1;
    shared_ptr<BoxShape> box(new BoxShape(Vector3d(floor_width, floor_height, floor_width)));
    //auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect>(box);
    auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect, VisualAspect>(box);
    shapeNode->getDynamicsAspect()->setFrictionCoeff(json["friction_coeff"].get<double>());
    shapeNode->getDynamicsAspect()->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(1.0, 1.0, 1.0));
    Isometry3d tf(Isometry3d::Identity());
    tf.translation() = Vector3d(0.0, -0.502, 0.0);
    body->getParentJoint()->setTransformFromParentBodyNode(tf);
    world->addSkeleton(floor);

    scales = readVectorXdFrom(json["scales"]);
    //state = VectorXd(skeleton->getNumDofs() * 2);
    state = VectorXd(skeleton->getNumDofs() * 2 + 3);
    action = VectorXd(skeleton->getNumDofs() - 3);

    skeleton->setPositionLowerLimits(readVectorXdFrom(json["lower_limits"]));
    skeleton->setPositionUpperLimits(readVectorXdFrom(json["upper_limits"]));

    VectorXd limits = 1000 * VectorXd::Ones(skeleton->getNumDofs());
    skeleton->setVelocityLowerLimits(-limits);
    skeleton->setVelocityUpperLimits(limits);

    for (Joint *joint: skeleton->getJoints())
    {
        joint->setActuatorType(Joint::FORCE);
        joint->setLimitEnforcement(true);
    }

    double cfm = dart::constraint::JointLimitConstraint::getConstraintForceMixing();
    double erp = dart::constraint::JointLimitConstraint::getErrorReductionParameter();
    cfm = 0.1;
    erp = 0.2;
    dart::constraint::JointLimitConstraint::setConstraintForceMixing(cfm);
    dart::constraint::JointLimitConstraint::setErrorReductionParameter(erp);
    dart::constraint::ContactConstraint::setConstraintForceMixing(cfm);
    dart::constraint::ContactConstraint::setErrorReductionParameter(erp);
    cout << "CFM: " << cfm << endl;
    cout << "ERP: " << erp << endl;

    refMotion = readVectorXdListFrom(json["ref_motion"]);
    frameRate = json["frame_rate"].get<int>();
    kp = Eigen::VectorXd(skeleton->getNumDofs());
    kp << 0, 0, 0, readVectorXdFrom(json["kp"]);
    kd = Eigen::VectorXd(skeleton->getNumDofs());
    kd << 0, 0, 0, readVectorXdFrom(json["kd"]);
    mkp = MatrixXd::Zero(kp.size(), kp.size());
    mkd = MatrixXd::Zero(kd.size(), kd.size());
    mkp.diagonal() = kp;
    mkd.diagonal() = kd;
    period = (double)refMotion.size() / frameRate;

    reset();
}

SimpleEnv::~SimpleEnv()
{
}

void SimpleEnv::reset()
{
    world->reset();
    VectorXd initPos = VectorXd::Zero(skeleton->getNumDofs());
    initPos << 0, 0, 0, refMotion[0];
    skeleton->setPositions(initPos);
    VectorXd initVel = VectorXd::Zero(skeleton->getNumDofs());
    initVel[0] = 1.5;
    skeleton->setVelocities(initVel);
    prev_com = skeleton->getRootBodyNode()->getCOM();
    done = false;
    double intPart;
    phase = modf(getTime() / period, &intPart);
    frameIdx = (size_t)round(phase * refMotion.size());
    if (frameIdx >= refMotion.size())
        frameIdx -= refMotion.size();
    updateState();
}

void SimpleEnv::step()
{
    double intPart;
    phase = modf(getTime() / period, &intPart);
    frameIdx = (size_t)round(phase * refMotion.size());
    if (frameIdx >= refMotion.size())
        frameIdx -= refMotion.size();

    const VectorXd &target = refMotion[frameIdx];
    VectorXd ref(skeleton->getNumDofs());
    ref << 0, 0, 0, target + (action.array().min(1).max(-1) * scales.array()).matrix();

    prev_com = skeleton->getRootBodyNode()->getCOM();
    done = false;

    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        // stable PD
        VectorXd q = skeleton->getPositions();
        VectorXd dq = skeleton->getVelocities();
        MatrixXd invM = (skeleton->getMassMatrix() + mkd * skeleton->getTimeStep()).inverse();
        VectorXd p = -kp.array() * skeleton->getPositionDifferences(q + dq * skeleton->getTimeStep(), ref).array();
        VectorXd d = -kd.array() * dq.array();
        VectorXd qddot = invM * (-skeleton->getCoriolisAndGravityForces() + p + d + skeleton->getConstraintForces());
        VectorXd force = p + d - (kd.array() * qddot.array()).matrix() * world->getTimeStep();
        skeleton->setForces(force);
        world->step();
    }
    updateState();
    //cout << skeleton->getPositions().transpose() << endl;
}

void SimpleEnv::updateState()
{
    vector<BodyNode*> bns = skeleton->getBodyNodes();
    BodyNode *lFoot = bns[3], *rFoot = bns[6], *spNode;
    if (lFoot->getCOM().y() < rFoot->getCOM().y())
        spNode = lFoot;
    else
        spNode = rFoot;
    Vector3d r_IP = skeleton->getCOM() - spNode->getCOM();
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    state << q, dq, r_IP;
    Eigen::Vector3d curr_com = skeleton->getRootBodyNode()->getCOM();
    done = done || (curr_com.y() < 0.50 || curr_com.y() > 1.70);

    double r_com_vel, r_ref, r_norm;
    //r_com_vel = 1 * exp(-5 * abs((curr_com.x() - prev_com.x()) * actionRate - 1.2));
    r_com_vel = (curr_com.x() - prev_com.x()) * actionRate - 1.2;

    //double theta = sin(getTime() / 4 * M_PI) * M_PI / 6;
    //r_ref = 10 * (exp(-5 * abs(q[3] - theta)) + exp(-5 * abs(q[6] + theta)));
    VectorXd ref = refMotion[frameIdx];
    r_ref = 10 * exp(-2 * (q.tail(6) - ref).norm());

    r_norm = 2 * exp(-0.5 * action.norm());

    reward = r_com_vel + r_ref + r_norm;
    //cout << r_com_vel << " " << r_ref << " " << r_norm << endl;
    //cout << curr_com.transpose() << endl;

    /*
    if (isnan(reward) || state.array().isNaN().any())
    {
        //cout << state << endl;
        //cout << reward << endl;
        state.setOnes();
        reward = -100;
        done = true;
    }
    */
}
